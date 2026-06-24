from typing import Any

from agile.web import PagingResponse
from agile.web.common_result import gen_response_model, R
from fastapi import APIRouter

from infra.db.repos import user_repo
from domain.common.exceptions import UserNotFoundError
from services.graph import graph_ops
from shared.utils.json_utils import coerce_json_field
from interfaces.api.schemas.web_models import GraphFactsRequest, GraphFactEntitiesRequest, GraphEntityRelationsRequest, \
    GraphEntityTopicsRequest, GraphTopicMemoriesRequest, GraphExploreRequest, GraphFactsFilters

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post(
    "/facts",
    summary="查询用户关联事实",
    response_model=gen_response_model(
        "GraphFactsResponse",
        data_type=PagingResponse,
        data_desc="分页事实列表",
    ),
)
async def facts(req: GraphFactsRequest):
    user = await user_repo.get_user(req.user_identity, using_cache=True)
    if not user:
        raise UserNotFoundError(req.user_identity)

    total, records = graph_ops.find_user_facts_page(
        user.id,
        req.current or 1,
        req.size or 20,
        filters=req.filters
    )
    return R.success(
        data=PagingResponse(
            records=records,
            total=total,
            current=req.current or 1,
            size=req.size or 20)
    )


@router.post(
    "/fact/entities",
    summary="查询事实关联的规范化实体",
    response_model=gen_response_model(
        "GraphFactEntitiesResponse",
        data_type=PagingResponse,
        data_desc="分页规范化实体列表",
    ),
)
async def fact_entities(req: GraphFactEntitiesRequest):
    user = await user_repo.get_user(req.user_identity, using_cache=True)
    if not user:
        raise UserNotFoundError(req.user_identity)

    total, records = graph_ops.find_fact_canonical_entities_page(user.id, req.fact_id, req.current or 1, req.size or 20)
    return R.success(
        data=PagingResponse(
            records=records,
            total=total,
            current=req.current or 1,
            size=req.size or 20
        )
    )


@router.post(
    "/entity/relations",
    summary="查询实体关系",
    response_model=gen_response_model(
        "GraphEntityRelationsResponse",
        data_type=PagingResponse,
        data_desc="分页实体关系列表",
    ),
)
async def entity_relations(req: GraphEntityRelationsRequest):
    user = await user_repo.get_user(req.user_identity, using_cache=True)
    if not user:
        raise UserNotFoundError(req.user_identity)

    total, records = graph_ops.find_entity_relations_page(
        user.id,
        req.canonical_id,
        req.current or 1,
        req.size or 20,
        filters=req.filters
    )

    for row in records:
        fact_ids = coerce_json_field(row.get("fact_ids"), [])
        row["fact_ids"] = fact_ids if isinstance(fact_ids, list) else []

    return R.success(
        data=PagingResponse(
            records=records,
            total=total,
            current=req.current or 1,
            size=req.size or 20
        )
    )


@router.post(
    "/entity/topics",
    summary="查询实体关联话题",
    response_model=gen_response_model(
        "GraphEntityTopicsResponse",
        data_type=PagingResponse,
        data_desc="分页话题列表",
    ),
)
async def entity_topics(req: GraphEntityTopicsRequest):
    user = await user_repo.get_user(req.user_identity, using_cache=True)
    if not user:
        raise UserNotFoundError(req.user_identity)

    total, records = graph_ops.find_entity_topics_page(user.id, req.canonical_id, req.current or 1, req.size or 20)

    for row in records:
        keywords = coerce_json_field(row.get("keywords"), [])
        row["keywords"] = keywords if isinstance(keywords, list) else []
        dialogue_ids = coerce_json_field(row.get("dialogue_ids"), [])
        row["dialogue_ids"] = dialogue_ids if isinstance(dialogue_ids, list) else []
        row["fact_count"] = int(row.get("fact_count") or 0)
        row["dialogue_count"] = int(row.get("dialogue_count") or 0)

    return R.success(
        data=PagingResponse(
            records=records,
            total=total,
            current=req.current or 1,
            size=req.size or 20
        )
    )


@router.post(
    "/topic/memories",
    summary="查询话题关联记忆",
    response_model=gen_response_model(
        "GraphTopicMemoriesResponse",
        data_type=PagingResponse,
        data_desc="分页记忆列表",
    ),
)
async def topic_memories(req: GraphTopicMemoriesRequest):
    user = await user_repo.get_user(req.user_identity, using_cache=True)
    if not user:
        raise UserNotFoundError(req.user_identity)

    total, records = graph_ops.find_topic_memories_page(user.id, req.topic_id, req.current or 1, req.size or 20)
    records = _normalize_memories(records)
    return R.success(
        data=PagingResponse(
            records=records,
            total=total,
            current=req.current or 1,
            size=req.size or 20
        )
    )


@router.post(
    "/explore",
    summary="图探索聚合查询",
    response_model=gen_response_model(
        "GraphExploreResponse",
        data_type=dict[str, Any],
        data_desc="前端可渲染的 nodes/edges 数据",
    ),
)
async def explore(req: GraphExploreRequest):
    user = await user_repo.get_user(req.user_identity, using_cache=True)
    if not user:
        raise UserNotFoundError(req.user_identity)

    current = req.current or 1
    size = req.size or 20
    relation_size = req.relation_size or 20
    entity_size = req.entity_size or 20
    max_nodes = req.max_nodes
    max_edges = req.max_edges

    nodes_map: dict[str, dict[str, Any]] = {}
    edges_map: dict[str, dict[str, Any]] = {}
    node_limit_hit = False
    edge_limit_hit = False
    stopped_at: str | None = None
    current_stage = "seed"

    def add_node(node_id: str, node_type: str, label: str | None = None, data: dict[str, Any] | None = None) -> bool:
        nonlocal node_limit_hit, stopped_at
        if node_id in nodes_map:
            return True
        if max_nodes is not None and len(nodes_map) >= max_nodes:
            node_limit_hit = True
            if stopped_at is None:
                stopped_at = f"{current_stage}.node_limit"
            return False
        nodes_map[node_id] = {
            "id": node_id,
            "type": node_type,
            "label": label or node_id,
            "data": data or {},
        }
        return True

    def add_edge(source: str, target: str, relation: str, data: dict[str, Any] | None = None) -> bool:
        nonlocal edge_limit_hit, stopped_at
        if source not in nodes_map or target not in nodes_map:
            return False
        key = f"{source}|{target}|{relation}"
        if key in edges_map:
            return True
        if max_edges is not None and len(edges_map) >= max_edges:
            edge_limit_hit = True
            if stopped_at is None:
                stopped_at = f"{current_stage}.edge_limit"
            return False
        edges_map[key] = {
            "id": key,
            "source": source,
            "target": target,
            "relation": relation,
            "data": data or {},
        }
        return True

    def should_stop_expansion() -> bool:
        return node_limit_hit or edge_limit_hit

    if req.seed_type == "canonical":
        current_stage = "canonical.seed"
        seed_node_id = f"canonical:{req.seed_id}"
        add_node(seed_node_id, "canonical", req.seed_id, {"canonical_id": req.seed_id})
        current_stage = "canonical.relations"
        _, relations = graph_ops.find_entity_relations_page(user.id, req.seed_id, current=current, size=size)
        for relation in relations:
            if should_stop_expansion():
                break
            source_id = f"canonical:{relation['source_canonical_id']}"
            target_id = f"canonical:{relation['target_canonical_id']}"
            add_node(source_id, "canonical", relation["source_canonical_id"],
                     {"canonical_id": relation["source_canonical_id"]})
            add_node(target_id, "canonical", relation["target_canonical_id"],
                     {"canonical_id": relation["target_canonical_id"]})
            add_edge(source_id, target_id, relation["edge_relation"], relation)

    elif req.seed_type == "fact":
        current_stage = "fact.seed"
        seed_fact_id = req.seed_id
        seed_fact_node = f"fact:{seed_fact_id}"
        add_node(seed_fact_node, "fact", seed_fact_id, {"fact_id": seed_fact_id})

        if should_stop_expansion():
            truncated = node_limit_hit or edge_limit_hit
            return R.success(
                data={
                    "seed": {"type": req.seed_type, "id": req.seed_id},
                    "nodes": list(nodes_map.values()),
                    "edges": list(edges_map.values()),
                    "meta": {
                        "current": current,
                        "size": size,
                        "node_count": len(nodes_map),
                        "edge_count": len(edges_map),
                        "max_nodes": max_nodes,
                        "max_edges": max_edges,
                        "truncated": truncated,
                        "node_limit_hit": node_limit_hit,
                        "edge_limit_hit": edge_limit_hit,
                        "stopped_at": stopped_at,
                    },
                }
            )

        current_stage = "fact.entities"
        _, entities = graph_ops.find_fact_canonical_entities_page(
            user.id,
            seed_fact_id,
            current=current,
            size=entity_size,
        )
        for entity in entities:
            if should_stop_expansion():
                break
            canonical_id = entity.get("canonical_id")
            if not canonical_id:
                continue
            entity_node_id = f"canonical:{canonical_id}"
            add_node(entity_node_id, "canonical", entity.get("canonical_name") or canonical_id, entity)
            add_edge(seed_fact_node, entity_node_id, "HAS_ENTITY", {"fact_id": seed_fact_id})

            if should_stop_expansion():
                continue

            current_stage = "fact.entity_relations"
            _, relations = graph_ops.find_entity_relations_page(
                user.id,
                str(canonical_id),
                current=1,
                size=relation_size,
            )
            for relation in relations:
                if should_stop_expansion():
                    break
                source_id = f"canonical:{relation['source_canonical_id']}"
                target_id = f"canonical:{relation['target_canonical_id']}"
                add_node(source_id, "canonical", relation["source_canonical_id"],
                         {"canonical_id": relation["source_canonical_id"]})
                add_node(target_id, "canonical", relation["target_canonical_id"],
                         {"canonical_id": relation["target_canonical_id"]})
                add_edge(source_id, target_id, relation["edge_relation"], relation)

    elif req.seed_type == "topic":
        current_stage = "topic.seed"
        seed_topic_id = req.seed_id
        seed_topic_node = f"topic:{seed_topic_id}"
        add_node(seed_topic_node, "topic", seed_topic_id, {"topic_id": seed_topic_id})

        if should_stop_expansion():
            truncated = node_limit_hit or edge_limit_hit
            return R.success(
                data={
                    "seed": {"type": req.seed_type, "id": req.seed_id},
                    "nodes": list(nodes_map.values()),
                    "edges": list(edges_map.values()),
                    "meta": {
                        "current": current,
                        "size": size,
                        "node_count": len(nodes_map),
                        "edge_count": len(edges_map),
                        "max_nodes": max_nodes,
                        "max_edges": max_edges,
                        "truncated": truncated,
                        "node_limit_hit": node_limit_hit,
                        "edge_limit_hit": edge_limit_hit,
                        "stopped_at": stopped_at,
                    },
                }
            )

        current_stage = "topic.facts"
        _, facts = graph_ops.find_user_facts_page(
            user.id,
            current=current,
            size=size,
            filters=GraphFactsFilters(topic_id=seed_topic_id),
        )
        for fact in facts:
            if should_stop_expansion():
                break
            fact_id = fact["id"]
            fact_node = f"fact:{fact_id}"
            add_node(fact_node, "fact", fact_id, fact)
            add_edge(seed_topic_node, fact_node, "HAS_FACT", {"topic_id": seed_topic_id})

            if should_stop_expansion():
                continue

            current_stage = "topic.fact_entities"
            _, entities = graph_ops.find_fact_canonical_entities_page(
                user.id,
                fact_id,
                current=1,
                size=entity_size,
            )
            for entity in entities:
                if should_stop_expansion():
                    break
                canonical_id = entity.get("canonical_id")
                if not canonical_id:
                    continue
                entity_node_id = f"canonical:{canonical_id}"
                add_node(entity_node_id, "canonical", entity.get("canonical_name") or canonical_id, entity)
                add_edge(fact_node, entity_node_id, "HAS_ENTITY", {"fact_id": fact_id})

                if not req.include_relations_on_topic_entities or should_stop_expansion():
                    continue

                current_stage = "topic.entity_relations"
                _, relations = graph_ops.find_entity_relations_page(
                    user.id,
                    str(canonical_id),
                    current=1,
                    size=relation_size,
                )
                for relation in relations:
                    if should_stop_expansion():
                        break
                    source_id = f"canonical:{relation['source_canonical_id']}"
                    target_id = f"canonical:{relation['target_canonical_id']}"
                    add_node(source_id, "canonical", relation["source_canonical_id"],
                             {"canonical_id": relation["source_canonical_id"]})
                    add_node(target_id, "canonical", relation["target_canonical_id"],
                             {"canonical_id": relation["target_canonical_id"]})
                    add_edge(source_id, target_id, relation["edge_relation"], relation)

    else:
        raise ValueError(f"Unsupported seed_type: {req.seed_type}")

    truncated = node_limit_hit or edge_limit_hit

    return R.success(
        data={
            "seed": {"type": req.seed_type, "id": req.seed_id},
            "nodes": list(nodes_map.values()),
            "edges": list(edges_map.values()),
            "meta": {
                "current": current,
                "size": size,
                "node_count": len(nodes_map),
                "edge_count": len(edges_map),
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "truncated": truncated,
                "node_limit_hit": node_limit_hit,
                "edge_limit_hit": edge_limit_hit,
                "stopped_at": stopped_at,
            },
        }
    )


def _normalize_memories(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for memory in records:
        memory.pop("mean_vec", None)
        memory.pop("compressed_vec", None)

        sectors = memory.get("sectors")
        tags = memory.get("tags")
        meta = memory.get("meta")
        sectors = coerce_json_field(sectors, [])
        memory["sectors"] = sectors if isinstance(sectors, list) else []
        tags = coerce_json_field(tags, [])
        memory["tags"] = tags if isinstance(tags, list) else []
        meta = coerce_json_field(meta, {})
        memory["meta"] = meta if isinstance(meta, dict) else {}
        normalized.append(memory)

    return normalized
