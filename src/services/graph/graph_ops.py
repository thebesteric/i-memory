import asyncio
import datetime
import itertools
import uuid
from typing import Any

from agile.utils import LogHelper, timing
from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from services.memory.components import get_embed_model
from shared.config.settings import env
from infra.db.engine import get_session_factory
from infra.db.orm_models import (
    GraphCanonicalEntities,
    GraphEntities,
    GraphEntityRelations,
    GraphFactEntities,
    GraphFacts,
    GraphTopics,
    Memories,
)
from services.graph.entity_canonicalize import EntityCanonicalize
from services.graph.relation_inferencer import RelationInference, RelationInferenceOutput
from domain.graph.node_models import EdgeRelation
from services.graph.semantic_spliter import Topic
from domain.graph.models import Fact, Entity, EntityType, InferSource, RelationInferenceResult
from domain.memory.models import IMemoryUser
from shared.utils.json_utils import coerce_json_field
from interfaces.api.schemas.web_models import GraphEntityRelationFilters, GraphFactsFilters

logger = LogHelper.get_logger()

embed_model = get_embed_model()
session_factory = get_session_factory()
# 实体标准化器
entity_canonicalize = EntityCanonicalize(threshold=0.85)
# 关系推理器
relation_inference = RelationInference()


def _model_to_dict(model) -> dict[str, Any]:
    return {column.name: getattr(model, column.name) for column in model.__table__.columns}


def _mapping_to_dict(row) -> dict[str, Any]:
    return {str(k): v for k, v in row.items()}


def _get_session(conn=None):
    external_session = conn is not None
    session = conn if conn is not None else session_factory()
    return session, external_session


async def add_topic(user: IMemoryUser, topic: Topic, conn=None):
    """
    添加主题
    :param user:
    :param topic:
    :param conn:
    :return:
    """
    now = datetime.datetime.now()
    # 对 summary 进行向量化
    vector = await embed_model.embed(topic.summary)

    topic_id = str(uuid.uuid4())
    topic.set_id(topic_id)
    topic.set_vector(vector)

    external_session = conn is not None
    session = conn if conn is not None else session_factory()
    try:
        session.add(
            GraphTopics(  # type: ignore[arg-type]
                id=topic_id,
                user_id=user.id,
                name=topic.name,
                summary=topic.summary,
                vector=vector,
                keywords=topic.keywords or [],
                dialogue_ids=topic.dialogue_ids or [],
                created_at=now,
                updated_at=now,
            )
        )
        # 若调用方持有会话，需立即刷新，确保主题数据行已存在；避免同一事务内后续查询触发自动刷新时，连带执行依赖的事实插入操作。
        if external_session:
            session.flush()
        if not external_session:
            session.commit()
    finally:
        if not external_session:
            session.close()
    return topic


async def add_fact(user: IMemoryUser, fact: Fact, topic: Topic, conn=None) -> Fact:
    """
    添加事实
    :param user: 用户
    :param fact: 事实
    :param topic: 话题
    :param conn: 数据库连接对象
    :return:
    """
    now = datetime.datetime.now()
    fact_id = str(uuid.uuid4())
    fact.set_id(fact_id)

    # 事实的语义向量（由 5W 组合生成）
    parts = []
    if fact.what: parts.append(f"What: {fact.what}")
    if fact.who: parts.append(f"Who: {fact.who}")
    if fact.when: parts.append(f"When: {fact.when}")
    if fact.where: parts.append(f"Where: {fact.where}")
    if fact.why: parts.append(f"Why: {fact.why}")

    vector = await get_embed_model().embed(" | ".join(parts))

    external_session = conn is not None
    session = conn if conn is not None else session_factory()
    try:
        # noinspection PyTypeChecker
        session.add(
            GraphFacts(  # type: ignore[arg-type]
                id=fact_id,
                user_id=user.id,
                topic_id=topic.id,
                what=fact.what,
                when_=fact.when,
                where_=fact.where,
                who=fact.who,
                why=fact.why,
                confidence=fact.confidence,
                vector=vector,
                fact_kind=fact.fact_kind or "conversation",
                occurred_start=fact.occurred_start,
                occurred_end=fact.occurred_end,
                created_at=now,
                updated_at=now,
                processed_at=None,
            )
        )
        if not external_session:
            session.commit()
    finally:
        if not external_session:
            session.close()
    return fact


async def add_entity(user: IMemoryUser, entity: Entity, conn=None) -> Entity:
    """
    添加实体
    :param user: 用户
    :param entity: 实体
    :param conn: 数据库连接对象
    :return:
    """
    if entity.id is not None:
        return entity

    text = getattr(entity, "text", None)
    entity_type = getattr(entity.entity_type, 'name', EntityType.OTHER.name)

    # 如果存在则不添加实体（text 和 entity_type 相同，视为同一个实体）
    external_session = conn is not None
    session = conn if conn is not None else session_factory()
    try:
        existing = session.execute(
            select(GraphEntities.id, GraphEntities.user_id, GraphEntities.canonical_id)
            .where(GraphEntities.user_id == user.id, GraphEntities.text == text, GraphEntities.entity_type == entity_type)
            .limit(1)
        ).mappings().first()
        if existing:
            entity.set_id(existing["id"])
            entity.set_user_id(existing["user_id"])
            entity.set_canonical_id(existing["canonical_id"])
            return entity

        # 标准化实体
        canonical_entity = await entity_canonicalize.canonicalize(user, entity, conn=session)

        now = datetime.datetime.now()
        entity_id = str(uuid.uuid4())
        entity.set_id(entity_id)
        entity.set_user_id(user.id)
        if canonical_entity.id:
            entity.set_canonical_id(canonical_entity.id)

        # 添加实体
            # noinspection PyTypeChecker
            session.add(
            GraphEntities(  # type: ignore[arg-type]
                id=entity_id,
                user_id=user.id,
                text=text,
                entity_type=entity_type,
                canonical_id=canonical_entity.id or None,
                canonical_name=canonical_entity.name or None,
                created_at=now,
                updated_at=now,
            )
        )
        if not external_session:
            session.commit()
        return entity
    finally:
        if not external_session:
            session.close()


async def link_fact_entities(user: IMemoryUser, fact: Fact, conn=None) -> None:
    """
    关联事实和实体关系
    :param user: 用户
    :param fact: 事实
    :param conn: 数据库连接对象
    :return:
    """
    now = datetime.datetime.now()
    external_session = conn is not None
    session = conn if conn is not None else session_factory()

    try:
        for entity in fact.entities:
            if not entity.id:
                # 添加实体
                entity = await add_entity(user=user, entity=entity, conn=session)

        # 给 entity 增加 canonical_id 值
            if entity.id and not entity.canonical_id:
                entity_row = session.execute(
                    select(GraphEntities.canonical_id).where(GraphEntities.id == entity.id).limit(1)
                ).mappings().first()
                if entity_row and entity_row.get("canonical_id"):
                    entity.set_canonical_id(entity_row["canonical_id"])

        # 添加与 Fact 的关系映射
            _id = str(uuid.uuid4())
            stmt = pg_insert(GraphFactEntities).values(
                id=_id,
                user_id=user.id,
                fact_id=fact.id,
                entity_id=entity.id,
                canonical_id=entity.canonical_id,
                relation_to_user=entity.relation_to_user,
                created_at=now,
                updated_at=now,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[GraphFactEntities.user_id, GraphFactEntities.fact_id, GraphFactEntities.entity_id],
                set_={
                    "canonical_id": stmt.excluded.canonical_id,
                    "relation_to_user": stmt.excluded.relation_to_user,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            session.execute(stmt)

        if not external_session:
            session.commit()
    finally:
        if not external_session:
            session.close()


async def mark_memoires_to_fact_joined(m_ids: list[str], conn=None) -> int:
    """
    将记忆标记为已参与事实处理
    :param m_ids:
    :param conn:
    :return:
    """
    if not m_ids:
        return 0
    external_session = conn is not None
    session = conn if conn is not None else session_factory()
    try:
        result = session.execute(update(Memories).where(Memories.id.in_(m_ids)).values(fact_joined=True))
        if not external_session:
            session.commit()
        return int(getattr(result, "rowcount", 0) or 0)
    finally:
        if not external_session:
            session.close()


def _normalize_edge_relation(edge_relation_value: str | None) -> str:
    if not edge_relation_value:
        return EdgeRelation.CO_OCCURS_WITH.name
    normalized = str(edge_relation_value).strip().upper()
    return normalized if normalized in EdgeRelation.__members__ else EdgeRelation.RELATED_TO.name


def _extract_entity_meta_by_canonical_id(fact: Fact) -> dict[str, dict[str, Any]]:
    """
    抽取事实中的标准化实体的元信息，按照 canonical_id 聚合文本、实体类型和事实级上下文。
    :param fact: 事实
    :return:
    """
    canonical_meta: dict[str, dict[str, Any]] = {}
    for entity in fact.entities:
        if not entity.canonical_id:
            continue
        # 如果 key 不存在，则新建并设置默认值
        meta = canonical_meta.setdefault(
            entity.canonical_id,
            {
                "texts": set(),
                "entity_type": EntityType.OTHER.name,
            }
        )
        # 追加 entity.text 到文本集合，并记录实体类型（如果尚未设置为其他）
        if entity.text:
            meta["texts"].add(entity.text)

        # 顺序敏感，谁先出现，谁更可能决定最终类型；如果已经是其他类型了，就不覆盖了
        entity_type_name = getattr(entity.entity_type, "name", EntityType.OTHER.name)
        if meta["entity_type"] == EntityType.OTHER.name:
            meta["entity_type"] = entity_type_name

    for _, meta in canonical_meta.items():
        meta["texts"] = sorted(meta["texts"])
    return canonical_meta


def _infer_edge_by_rules(
        fact: Fact,
        source_meta: dict,
        target_meta: dict,
        enable: bool = True) -> tuple[str, str] | None:
    """
    基于规则进行边关系匹配
    :param fact: 事实
    :param source_meta: 源实体的元信息
    :param target_meta: 目标实体的元信息
    :param enable: 是否使用规则匹配
    :return:
    """
    if not enable:
        return None

    who = (fact.who or "").lower()
    what = (fact.what or "").lower()
    where = (fact.where or "").lower()
    why = (fact.why or "").lower()
    merged = f"{who} {what} {where} {why}"
    source_type = source_meta.get("entity_type", EntityType.OTHER.name)
    target_type = target_meta.get("entity_type", EntityType.OTHER.name)

    if source_type == EntityType.LOCATION.name and target_type == EntityType.LOCATION.name:
        if any(k in merged for k in ["位于", "属于", "inside", "within", " in ", " at "]):
            return EdgeRelation.LOCATED_IN.name, "Rule inferred hierarchical location relation"

    if any(k in merged for k in ["因为", "导致", "因此", "so that", "because", "caused by", "lead to"]):
        return EdgeRelation.CAUSES.name, "Rule inferred causal relation"

    if any(k in merged for k in ["之后", "随后", "然后", "before", "after", "later", "earlier"]):
        return EdgeRelation.PRECEDES.name, "Rule inferred temporal order relation"

    if source_type == EntityType.PERSON.name and target_type == EntityType.PERSON.name:
        if any(k in merged for k in
               ["同事", "合作", "团队", "项目", "coworker", "colleague", "teammate", "worked with"]):
            return EdgeRelation.WORKED_WITH.name, "Rule inferred collaboration relation"

    return None


async def _infer_edge_by_llm(
        source_canonical_id: str,
        target_canonical_id: str,
        fact: Fact,
        source_meta: dict,
        target_meta: dict
) -> RelationInferenceResult:
    """
    利用大模型进行边类型的推断
    :param fact: 事实
    :param source_meta: 源实体元数据
    :param target_meta: 目标实体元数据
    :return: RelationInferenceResult
    """
    try:
        output: RelationInferenceOutput = await relation_inference.invoke(
            fact=fact,
            source_meta=source_meta,
            target_meta=target_meta
        )
        edge_relation = _normalize_edge_relation(output.edge_relation)
        relation_evidence = output.relation_evidence or "LLM inferred relation"
        if output.confidence < env.GRAPH_RELATION_LLM_MIN_CONFIDENCE:
            logger.info(
                f"[GRAPH] LLM inferred relation confidence {output.confidence} is below threshold "
                f"{env.GRAPH_RELATION_LLM_MIN_CONFIDENCE}, keep evidence for fallback"
            )
        else:
            logger.info(
                f"[GRAPH] LLM inferred relation: {edge_relation} with confidence {output.confidence}, evidence: {relation_evidence}"
            )
        return RelationInferenceResult(
            source_canonical_id=source_canonical_id,
            target_canonical_id=target_canonical_id,
            edge_relation=edge_relation,
            relation_evidence=relation_evidence,
            infer_source="LLM",
            confidence=output.confidence,
        )
    except Exception as e:
        logger.warning(f"[GRAPH] LLM relation inference failed, fallback to co-occurrence: {e}")
        return RelationInferenceResult(
            source_canonical_id=source_canonical_id,
            target_canonical_id=target_canonical_id,
            edge_relation=EdgeRelation.CO_OCCURS_WITH.name,
            relation_evidence="LLM relation inference failed, fallback to co-occurrence",
            infer_source="FALLBACK",
            confidence=None,
        )


async def _infer_relation_for_pair(
        fact: Fact,
        source_canonical_id: str,
        target_canonical_id: str,
        source_meta: dict,
        target_meta: dict,
        llm_allowed_for_fact: bool,
        llm_semaphore: asyncio.Semaphore | None,
) -> RelationInferenceResult:
    """
    对单个 canonical 实体对做关系推断（规则 -> LLM -> fallback）。
    该阶段不访问数据库，便于并发执行。
    """
    llm_fallback_evidence = None
    llm_fallback_confidence = None
    llm_result = None

    infer_source: InferSource = "RULE"
    confidence = None
    relation = _infer_edge_by_rules(fact, source_meta, target_meta, enable=True)

    if relation is None and llm_allowed_for_fact:
        if llm_semaphore is None:
            llm_result = await _infer_edge_by_llm(
                source_canonical_id=source_canonical_id,
                target_canonical_id=target_canonical_id,
                fact=fact,
                source_meta=source_meta,
                target_meta=target_meta
            )
        else:
            async with llm_semaphore:
                llm_result = await _infer_edge_by_llm(
                    source_canonical_id=source_canonical_id,
                    target_canonical_id=target_canonical_id,
                    fact=fact,
                    source_meta=source_meta,
                    target_meta=target_meta
                )

        llm_edge_relation = llm_result.edge_relation
        llm_relation_evidence = llm_result.relation_evidence
        llm_confidence = llm_result.confidence

        if llm_confidence is not None and llm_confidence >= env.GRAPH_RELATION_LLM_MIN_CONFIDENCE:
            infer_source = "LLM"
            relation = (llm_edge_relation, llm_relation_evidence or "LLM inferred relation")
            confidence = llm_confidence
        else:
            llm_fallback_evidence = llm_relation_evidence if llm_confidence is not None else None
            llm_fallback_confidence = llm_confidence

    if relation is None:
        infer_source = "FALLBACK"
        fallback_evidence = "Entities co-occur in the same fact"
        if llm_fallback_evidence:
            fallback_evidence = f"Low-confidence LLM hint: {llm_fallback_evidence}"
        relation = (EdgeRelation.CO_OCCURS_WITH.name, fallback_evidence)
        confidence = llm_fallback_confidence

    edge_relation, relation_evidence = relation

    return RelationInferenceResult(
        source_canonical_id=source_canonical_id,
        target_canonical_id=target_canonical_id,
        edge_relation=edge_relation,
        relation_evidence=relation_evidence,
        infer_source=infer_source,
        confidence=confidence,
    )


@timing
async def infer_canonical_relations_for_fact(user: IMemoryUser, fact: Fact, conn=None) -> int:
    """
    基于单条 Fact 的 canonical 实体关系推断。

    流程：
    1) 先按 canonical_id 聚合实体并两两配对；
    2) 对每对实体按 rule -> llm -> fallback 决策 edge_relation；
    3) 以 (user_id, source_canonical_id, target_canonical_id, edge_relation) 去重写入；
    4) 通过 fact_ids 维护该关系的证据 fact 列表（去重追加）。
    """
    # 没有 fact 主键时无法作为关系证据，直接跳过
    if not fact.id:
        return 0

    # 将同义提及收敛到 canonical_id 维度，避免同一事实中重复实体造成重复建边
    canonical_meta = _extract_entity_meta_by_canonical_id(fact)
    canonical_ids = sorted(canonical_meta.keys())

    # 少于两个 canonical 实体时无法形成边
    if len(canonical_ids) < 2:
        return 0

    now = datetime.datetime.now()

    # LLM 触发门控：可配置总开关 + 单 fact 最大实体对数量，避免成本失控
    enable_llm = bool(env.GRAPH_RELATION_LLM_ENABLE)
    # LLM 最大允许处理的数量
    max_pairs_for_llm = int(env.GRAPH_RELATION_LLM_MAX_PAIRS or 0)
    # 当前事实的实体对数量（n 个实体形成 n*(n-1)/2 个对）
    pair_count = len(canonical_ids) * (len(canonical_ids) - 1) // 2
    # 是否开启 LLM 推断
    llm_allowed_for_fact = enable_llm and (max_pairs_for_llm <= 0 or pair_count <= max_pairs_for_llm)

    pairs = list(itertools.combinations(canonical_ids, 2))
    llm_concurrency = max(1, int(getattr(env, "GRAPH_RELATION_LLM_CONCURRENCY", 5) or 5))
    llm_semaphore = asyncio.Semaphore(llm_concurrency) if llm_allowed_for_fact else None

    # 并发执行推断阶段，数据库写入仍在后续串行执行
    infer_tasks = [
        _infer_relation_for_pair(
            fact=fact,
            source_canonical_id=source_canonical_id,
            target_canonical_id=target_canonical_id,
            source_meta=canonical_meta.get(source_canonical_id, {}),
            target_meta=canonical_meta.get(target_canonical_id, {}),
            llm_allowed_for_fact=llm_allowed_for_fact,
            llm_semaphore=llm_semaphore,
        )
        for source_canonical_id, target_canonical_id in pairs
    ]
    inferred_relations = await asyncio.gather(*infer_tasks)

    external_session = conn is not None
    session = conn if conn is not None else session_factory()
    affected = 0
    try:
        for inferred in inferred_relations:
            source_canonical_id = inferred.source_canonical_id
            target_canonical_id = inferred.target_canonical_id
            edge_relation = inferred.edge_relation
            relation_evidence = inferred.relation_evidence
            infer_source = inferred.infer_source
            confidence = inferred.confidence

            existing = session.execute(
                select(GraphEntityRelations.id, GraphEntityRelations.fact_ids)
                .where(
                    GraphEntityRelations.user_id == user.id,
                    GraphEntityRelations.source_canonical_id == source_canonical_id,
                    GraphEntityRelations.target_canonical_id == target_canonical_id,
                    GraphEntityRelations.edge_relation == edge_relation,
                )
                .limit(1)
            ).mappings().first()

            if existing:
                fact_ids = coerce_json_field(existing.get("fact_ids"), [])
                if not isinstance(fact_ids, list):
                    fact_ids = []

                if fact.id not in fact_ids:
                    fact_ids.append(fact.id)
                    result = session.execute(
                        update(GraphEntityRelations)
                        .where(GraphEntityRelations.id == existing["id"])
                        .values(
                            fact_ids=fact_ids,
                            relation_evidence=relation_evidence,
                            infer_source=infer_source,
                            confidence=confidence,
                            updated_at=now,
                        )
                    )
                    affected += int(getattr(result, "rowcount", 0) or 0)
                continue

            # 不存在则插入新边，首条证据为当前 fact.id
            session.add(
                GraphEntityRelations(  # type: ignore[arg-type]
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    source_canonical_id=source_canonical_id,
                    target_canonical_id=target_canonical_id,
                    edge_relation=edge_relation,
                    relation_evidence=relation_evidence,
                    infer_source=infer_source,
                    confidence=confidence,
                    fact_ids=[fact.id],
                    created_at=now,
                    updated_at=now,
                )
            )
            affected += 1

        if not external_session:
            session.commit()
        return affected
    finally:
        if not external_session:
            session.close()


def find_canonical_relations(user_id: str, canonical_id: str, limit: int = 100, conn=None) -> list[dict]:
    """
    查询某个 canonical entity 参与的关系边（双向）。
    """
    session, external_session = _get_session(conn)
    try:
        query = (
            select(GraphEntityRelations)
            .where(
                GraphEntityRelations.user_id == user_id,
                or_(
                    GraphEntityRelations.source_canonical_id == canonical_id,
                    GraphEntityRelations.target_canonical_id == canonical_id,
                ),
            )
            .order_by(
                GraphEntityRelations.updated_at.desc().nullslast(),
                GraphEntityRelations.created_at.desc().nullslast(),
            )
            .limit(limit)
        )
        rows = session.execute(query).scalars().all()
        return [_model_to_dict(row) for row in rows]
    finally:
        if not external_session:
            session.close()


def find_user_facts_page(
        user_id: str,
        current: int = 1,
        size: int = 20,
        filters: GraphFactsFilters | None = None,
        conn=None,
) -> tuple[int, list[dict]]:
    current = max(1, int(current or 1))
    size = max(1, int(size or 20))
    offset = (current - 1) * size

    filters_expr = [GraphFacts.user_id == user_id]

    if filters:
        topic_id = filters.topic_id
        if topic_id:
            filters_expr.append(GraphFacts.topic_id == topic_id)

        fact_kind = filters.fact_kind
        if fact_kind:
            filters_expr.append(GraphFacts.fact_kind == fact_kind)

        min_confidence = filters.min_confidence
        if min_confidence is not None:
            filters_expr.append(GraphFacts.confidence >= min_confidence)

        max_confidence = filters.max_confidence
        if max_confidence is not None:
            filters_expr.append(GraphFacts.confidence <= max_confidence)

        keyword = (filters.keyword or "").strip()
        if keyword:
            like = f"%{keyword}%"
            filters_expr.append(
                or_(
                    GraphFacts.what.ilike(like),
                    GraphFacts.who.ilike(like),
                    GraphFacts.where_.ilike(like),
                    GraphFacts.why.ilike(like),
                )
            )

    session, external_session = _get_session(conn)
    try:
        total = int(
            session.execute(select(func.count()).select_from(GraphFacts).where(*filters_expr)).scalar_one() or 0
        )
        query = (
            select(
                GraphFacts.id,
                GraphFacts.user_id,
                GraphFacts.topic_id,
                GraphFacts.what,
                GraphFacts.when_.label("when_text"),
                GraphFacts.where_.label("where_text"),
                GraphFacts.who,
                GraphFacts.why,
                GraphFacts.confidence,
                GraphFacts.fact_kind,
                GraphFacts.occurred_start,
                GraphFacts.occurred_end,
                GraphFacts.created_at,
                GraphFacts.updated_at,
                GraphFacts.processed_at,
            )
            .where(*filters_expr)
            .order_by(GraphFacts.updated_at.desc().nullslast(), GraphFacts.created_at.desc().nullslast())
            .limit(size)
            .offset(offset)
        )
        rows = session.execute(query).mappings().all()
        return total, [_mapping_to_dict(row) for row in rows]
    finally:
        if not external_session:
            session.close()


def find_fact_canonical_entities_page(
        user_id: str,
        fact_id: str,
        current: int = 1,
        size: int = 20,
        conn=None,
) -> tuple[int, list[dict]]:
    current = max(1, int(current or 1))
    size = max(1, int(size or 20))
    offset = (current - 1) * size
    session, external_session = _get_session(conn)
    try:
        base_filters = [
            GraphFactEntities.user_id == user_id,
            GraphFactEntities.fact_id == fact_id,
            GraphFactEntities.canonical_id.is_not(None),
        ]
        total = int(
            session.execute(select(func.count()).select_from(GraphFactEntities).where(*base_filters)).scalar_one() or 0
        )
        query = (
            select(
                GraphFactEntities.id,
                GraphFactEntities.user_id,
                GraphFactEntities.fact_id,
                GraphFactEntities.entity_id,
                GraphFactEntities.canonical_id,
                GraphFactEntities.relation_to_user,
                GraphFactEntities.created_at,
                GraphFactEntities.updated_at,
                GraphCanonicalEntities.name.label("canonical_name"),
                GraphCanonicalEntities.entity_type.label("canonical_entity_type"),
                GraphEntities.text.label("entity_text"),
                GraphEntities.entity_type.label("entity_type"),
            )
            .select_from(GraphFactEntities)
            .outerjoin(GraphCanonicalEntities, GraphFactEntities.canonical_id == GraphCanonicalEntities.id)
            .outerjoin(GraphEntities, GraphFactEntities.entity_id == GraphEntities.id)
            .where(*base_filters)
            .order_by(GraphFactEntities.updated_at.desc().nullslast(), GraphFactEntities.created_at.desc().nullslast())
            .limit(size)
            .offset(offset)
        )
        rows = session.execute(query).mappings().all()
        return total, [_mapping_to_dict(row) for row in rows]
    finally:
        if not external_session:
            session.close()


def find_entity_relations_page(
        user_id: str,
        canonical_id: str,
        current: int = 1,
        size: int = 20,
        filters: GraphEntityRelationFilters | None = None,
        conn=None,
) -> tuple[int, list[dict]]:
    current = max(1, int(current or 1))
    size = max(1, int(size or 20))
    offset = (current - 1) * size

    filters_expr = [
        GraphEntityRelations.user_id == user_id,
        or_(
            GraphEntityRelations.source_canonical_id == canonical_id,
            GraphEntityRelations.target_canonical_id == canonical_id,
        ),
    ]

    if filters:
        if filters.edge_relations:
            filters_expr.append(GraphEntityRelations.edge_relation.in_(filters.edge_relations))
        if filters.infer_sources:
            filters_expr.append(GraphEntityRelations.infer_source.in_(filters.infer_sources))
        if filters.min_confidence is not None:
            filters_expr.append(GraphEntityRelations.confidence >= filters.min_confidence)
        if filters.max_confidence is not None:
            filters_expr.append(GraphEntityRelations.confidence <= filters.max_confidence)
        if filters.related_canonical_id:
            filters_expr.append(
                or_(
                    GraphEntityRelations.source_canonical_id == filters.related_canonical_id,
                    GraphEntityRelations.target_canonical_id == filters.related_canonical_id,
                )
            )
        if filters.fact_id:
            filters_expr.append(GraphEntityRelations.fact_ids.contains([filters.fact_id]))

    session, external_session = _get_session(conn)
    try:
        total = int(
            session.execute(select(func.count()).select_from(GraphEntityRelations).where(*filters_expr)).scalar_one() or 0
        )
        query = (
            select(
                GraphEntityRelations.id,
                GraphEntityRelations.user_id,
                GraphEntityRelations.source_canonical_id,
                GraphEntityRelations.target_canonical_id,
                GraphEntityRelations.edge_relation,
                GraphEntityRelations.relation_evidence,
                GraphEntityRelations.infer_source,
                GraphEntityRelations.confidence,
                GraphEntityRelations.fact_ids,
                GraphEntityRelations.created_at,
                GraphEntityRelations.updated_at,
            )
            .where(*filters_expr)
            .order_by(
                GraphEntityRelations.updated_at.desc().nullslast(),
                GraphEntityRelations.created_at.desc().nullslast(),
            )
            .limit(size)
            .offset(offset)
        )
        rows = session.execute(query).mappings().all()
        return total, [_mapping_to_dict(row) for row in rows]
    finally:
        if not external_session:
            session.close()


def find_entity_topics_page(
        user_id: str,
        canonical_id: str,
        current: int = 1,
        size: int = 20,
        conn=None,
) -> tuple[int, list[dict]]:
    current = max(1, int(current or 1))
    size = max(1, int(size or 20))
    offset = (current - 1) * size
    session, external_session = _get_session(conn)
    try:
        base_query = (
            select(GraphTopics.id)
            .select_from(GraphFactEntities)
            .join(GraphFacts, GraphFactEntities.fact_id == GraphFacts.id)
            .join(GraphTopics, GraphTopics.id == GraphFacts.topic_id)
            .where(GraphFactEntities.user_id == user_id, GraphFactEntities.canonical_id == canonical_id)
        )
        total = int(
            session.execute(
                select(func.count(func.distinct(GraphTopics.id)))
                .select_from(GraphFactEntities)
                .join(GraphFacts, GraphFactEntities.fact_id == GraphFacts.id)
                .join(GraphTopics, GraphTopics.id == GraphFacts.topic_id)
                .where(GraphFactEntities.user_id == user_id, GraphFactEntities.canonical_id == canonical_id)
            ).scalar_one() or 0
        )
        query = (
            select(
                GraphTopics.id,
                GraphTopics.user_id,
                GraphTopics.name,
                GraphTopics.summary,
                GraphTopics.keywords,
                GraphTopics.dialogue_ids,
                GraphTopics.created_at,
                GraphTopics.updated_at,
                func.count(func.distinct(GraphFacts.id)).label("fact_count"),
                func.coalesce(func.jsonb_array_length(GraphTopics.dialogue_ids), 0).label("dialogue_count"),
            )
            .select_from(GraphFactEntities)
            .join(GraphFacts, GraphFactEntities.fact_id == GraphFacts.id)
            .join(GraphTopics, GraphTopics.id == GraphFacts.topic_id)
            .where(GraphFactEntities.user_id == user_id, GraphFactEntities.canonical_id == canonical_id)
            .group_by(
                GraphTopics.id,
                GraphTopics.user_id,
                GraphTopics.name,
                GraphTopics.summary,
                GraphTopics.keywords,
                GraphTopics.dialogue_ids,
                GraphTopics.created_at,
                GraphTopics.updated_at,
            )
            .order_by(GraphTopics.updated_at.desc().nullslast(), GraphTopics.created_at.desc().nullslast())
            .limit(size)
            .offset(offset)
        )
        rows = session.execute(query).mappings().all()
        return total, [_mapping_to_dict(row) for row in rows]
    finally:
        if not external_session:
            session.close()


def find_topic_memories_page(
        user_id: str,
        topic_id: str,
        current: int = 1,
        size: int = 20,
        conn=None,
) -> tuple[int, list[dict]]:
    current = max(1, int(current or 1))
    size = max(1, int(size or 20))
    offset = (current - 1) * size
    session, external_session = _get_session(conn)
    try:
        topic_dialogue_ids = session.execute(
            select(GraphTopics.dialogue_ids)
            .where(GraphTopics.user_id == user_id, GraphTopics.id == topic_id)
            .limit(1)
        ).scalar_one_or_none()
        raw_dialogue_ids = topic_dialogue_ids if isinstance(topic_dialogue_ids, list) else []
        dialogue_ids = [str(v) for v in raw_dialogue_ids if v]
        if not dialogue_ids:
            return 0, []

        total = int(
            session.execute(
                select(func.count()).select_from(Memories).where(Memories.user_id == user_id, Memories.id.in_(dialogue_ids))
            ).scalar_one() or 0
        )
        query = (
            select(Memories)
            .where(Memories.user_id == user_id, Memories.id.in_(dialogue_ids))
            .order_by(Memories.updated_at.desc().nullslast(), Memories.created_at.desc().nullslast())
            .limit(size)
            .offset(offset)
        )
        rows = session.execute(query).scalars().all()
        return total, [_model_to_dict(row) for row in rows]
    finally:
        if not external_session:
            session.close()

