from dataclasses import dataclass
from typing import Any, Iterable

from agile.utils import LogHelper
from sqlalchemy import or_, select

from infra.db.engine import get_session_factory
from infra.db.orm_models import GraphEntityRelations, GraphFactEntities, GraphFacts, GraphTopics, Memories
from domain.graph.node_models import EdgeRelation
from shared.utils.json_utils import coerce_json_field

logger = LogHelper.get_logger(title="[GRAPH_SEARCH]")
session_factory = get_session_factory()


def _to_str_set(values: Iterable[str | None]) -> set[str]:
    return {str(v) for v in values if v}


def _safe_limit(value: int | None, default: int = 1) -> int:
    try:
        return max(1, int(value if value is not None else default))
    except (TypeError, ValueError):
        return max(1, int(default))


def _safe_confidence(value: float | None, default: float = 0.5) -> float:
    try:
        conf = float(value if value is not None else default)
    except (TypeError, ValueError):
        conf = default
    return max(0.0, min(1.0, conf))


def _safe_hops(value: int | None, default: int = 1, min_hops: int = 1, max_hops: int = 4) -> int:
    try:
        hops = int(value if value is not None else default)
    except (TypeError, ValueError):
        hops = default
    return max(min_hops, min(max_hops, hops))


def _safe_decay(value: float | None, default: float = 0.8) -> float:
    try:
        decay = float(value if value is not None else default)
    except (TypeError, ValueError):
        decay = default
    return max(0.1, min(1.0, decay))


@dataclass(frozen=True)
class GraphExpansionCandidate:
    id: str
    score: float


def _normalize_json_list(payload: Any) -> list[Any]:
    parsed = coerce_json_field(payload, [])
    return parsed if isinstance(parsed, list) else []


EDGE_RELATION_WEIGHTS: dict[str, float] = {
    EdgeRelation.SAME_AS.value: 1.00,
    EdgeRelation.PART_OF.value: 0.92,
    EdgeRelation.LOCATED_IN.value: 0.84,
    EdgeRelation.WORKED_WITH.value: 0.80,
    EdgeRelation.KNOWS.value: 0.78,
    EdgeRelation.CAUSES.value: 0.74,
    EdgeRelation.PRECEDES.value: 0.66,
    EdgeRelation.CO_OCCURS_WITH.value: 0.50,
    EdgeRelation.RELATED_TO.value: 0.30,
}


def _find_seed_topic_ids(user_id: str, seed_ids: set[str], limit: int) -> list[str]:
    if not seed_ids:
        return []
    dialogue_matchers = [GraphTopics.dialogue_ids.op("?")(seed_id) for seed_id in sorted(seed_ids)]
    if not dialogue_matchers:
        return []
    with session_factory() as session:
        rows = session.execute(
            select(GraphTopics.id)
            .where(GraphTopics.user_id == user_id, or_(*dialogue_matchers))
            .order_by(
                GraphTopics.updated_at.desc().nullslast(),
                GraphTopics.created_at.desc().nullslast(),
                GraphTopics.id.asc(),
            )
            .limit(limit)
        ).all()
    return [str(row[0]) for row in rows if row and row[0]]


def _find_seed_fact_ids(user_id: str, topic_ids: list[str], limit: int) -> list[str]:
    if not topic_ids:
        return []
    with session_factory() as session:
        rows = session.execute(
            select(GraphFacts.id)
            .where(GraphFacts.user_id == user_id, GraphFacts.topic_id.in_(topic_ids))
            .order_by(
                GraphFacts.confidence.desc().nullslast(),
                GraphFacts.updated_at.desc().nullslast(),
                GraphFacts.created_at.desc().nullslast(),
                GraphFacts.id.asc(),
            )
            .limit(limit)
        ).all()
    return [str(row[0]) for row in rows if row and row[0]]


def _find_seed_canonical_ids(user_id: str, fact_ids: list[str]) -> list[str]:
    if not fact_ids:
        return []
    with session_factory() as session:
        rows = session.execute(
            select(GraphFactEntities.canonical_id)
            .where(
                GraphFactEntities.user_id == user_id,
                GraphFactEntities.fact_id.in_(fact_ids),
                GraphFactEntities.canonical_id.is_not(None),
            )
            .distinct()
        ).all()
    return [str(row[0]) for row in rows if row and row[0]]


def _fetch_relation_edges(user_id: str, canonical_ids: set[str], min_confidence: float) -> list[dict[str, Any]]:
    if not canonical_ids:
        return []
    with session_factory() as session:
        rows = session.execute(
            select(
                GraphEntityRelations.source_canonical_id,
                GraphEntityRelations.target_canonical_id,
                GraphEntityRelations.edge_relation,
                GraphEntityRelations.confidence,
            )
            .where(
                GraphEntityRelations.user_id == user_id,
                or_(
                    GraphEntityRelations.source_canonical_id.in_(canonical_ids),
                    GraphEntityRelations.target_canonical_id.in_(canonical_ids),
                ),
                or_(
                    GraphEntityRelations.confidence.is_(None),
                    GraphEntityRelations.confidence >= min_confidence,
                ),
                GraphEntityRelations.source_canonical_id != GraphEntityRelations.target_canonical_id,
            )
        ).mappings().all()
    return [{str(k): v for k, v in row.items()} for row in rows]


def _walk_related_canonical_scores(
        user_id: str,
        seed_canonical_ids: list[str],
        *,
        max_hops: int,
        hop_decay: float,
        min_relation_confidence: float,
        per_hop_limit: int,
        min_walk_score: float,
) -> dict[str, float]:
    if not seed_canonical_ids:
        return {}

    # 状态包含：当前实体、已走路径（用于去环）、累计分数
    active_states: list[tuple[str, tuple[str, ...], float]] = [
        (canonical_id, (canonical_id,), 1.0)
        for canonical_id in seed_canonical_ids
        if canonical_id
    ]
    aggregated_scores: dict[str, float] = {}

    # 按 hop 逐层扩散，避免一次性全图扫描
    for hop in range(1, max_hops + 1):
        current_nodes = {canonical_id for canonical_id, _, _ in active_states}
        edges = _fetch_relation_edges(user_id, current_nodes, min_relation_confidence)
        if not edges:
            break

        next_states: list[tuple[str, tuple[str, ...], float]] = []
        for canonical_id, path, cum_score in active_states:
            for edge in edges:
                src = str(edge.get("source_canonical_id") or "")
                dst = str(edge.get("target_canonical_id") or "")
                if canonical_id not in {src, dst}:
                    continue
                next_id = dst if canonical_id == src else src
                # 跳过空节点与环路
                if not next_id or next_id in path:
                    continue

                edge_relation = str(edge.get("edge_relation") or EdgeRelation.RELATED_TO.value)
                relation_weight = EDGE_RELATION_WEIGHTS.get(edge_relation, EDGE_RELATION_WEIGHTS[EdgeRelation.RELATED_TO.value])
                confidence = float(edge.get("confidence") or 0.0)
                # 评分：保留历史分 + 本跳关系分（关系类型权重 + 置信度 + hop 衰减）
                score = cum_score * hop_decay + confidence * relation_weight * (hop_decay ** hop)
                if score < min_walk_score:
                    continue
                next_states.append((next_id, path + (next_id,), score))

        if not next_states:
            break

        # 每跳仅保留高分候选，控制分支爆炸
        next_states.sort(key=lambda item: (-item[2], item[0]))
        active_states = next_states[:per_hop_limit]
        for next_id, _, score in active_states:
            aggregated_scores[next_id] = max(aggregated_scores.get(next_id, 0.0), score)

    return aggregated_scores


def _fetch_related_fact_scores(user_id: str, canonical_scores: dict[str, float]) -> dict[str, float]:
    if not canonical_scores:
        return {}
    with session_factory() as session:
        rows = session.execute(
            select(GraphFactEntities.fact_id, GraphFactEntities.canonical_id)
            .where(
                GraphFactEntities.user_id == user_id,
                GraphFactEntities.canonical_id.in_(list(canonical_scores.keys())),
            )
        ).all()
    fact_scores: dict[str, float] = {}
    for fact_id, canonical_id in rows:
        if not fact_id or not canonical_id:
            continue
        fact_key = str(fact_id)
        canonical_key = str(canonical_id)
        fact_scores[fact_key] = max(fact_scores.get(fact_key, 0.0), canonical_scores.get(canonical_key, 0.0))
    return fact_scores


def _fetch_related_topic_scores(user_id: str, fact_scores: dict[str, float]) -> dict[str, float]:
    if not fact_scores:
        return {}
    with session_factory() as session:
        rows = session.execute(
            select(GraphFacts.id, GraphFacts.topic_id)
            .where(
                GraphFacts.user_id == user_id,
                GraphFacts.id.in_(list(fact_scores.keys())),
                GraphFacts.topic_id.is_not(None),
            )
        ).all()
    topic_scores: dict[str, float] = {}
    for fact_id, topic_id in rows:
        if not fact_id or not topic_id:
            continue
        topic_key = str(topic_id)
        topic_scores[topic_key] = max(topic_scores.get(topic_key, 0.0), fact_scores.get(str(fact_id), 0.0))
    return topic_scores


def _collect_related_memory_scores(user_id: str, topic_scores: dict[str, float]) -> dict[str, float]:
    if not topic_scores:
        return {}
    with session_factory() as session:
        topic_rows = session.execute(
            select(GraphTopics.id, GraphTopics.dialogue_ids)
            .where(GraphTopics.user_id == user_id, GraphTopics.id.in_(list(topic_scores.keys())))
        ).all()
        memory_scores: dict[str, float] = {}
        for topic_id, dialogue_ids in topic_rows:
            topic_key = str(topic_id)
            for memory_id in _normalize_json_list(dialogue_ids):
                mem_key = str(memory_id)
                if not mem_key:
                    continue
                memory_scores[mem_key] = max(memory_scores.get(mem_key, 0.0), topic_scores.get(topic_key, 0.0))

        if not memory_scores:
            return {}

        existing_memory_ids = {
            str(row[0])
            for row in session.execute(
                select(Memories.id)
                .where(Memories.user_id == user_id, Memories.id.in_(list(memory_scores.keys())))
            ).all()
            if row and row[0]
        }
    return {memory_id: memory_scores[memory_id] for memory_id in existing_memory_ids}


def expand_candidate_ids_via_graph(
        user_id: str,
        seed_memory_ids: set[str],
        limit: int,
        min_relation_confidence: float = 0.5,
        max_hops: int = 1,
        hop_decay: float = 0.8,
        per_hop_limit: int = 200,
        min_walk_score: float = 0.05,
) -> list[GraphExpansionCandidate]:
    """
    通过 Topic/Fact/Entity 图谱链路扩展候选记忆：
    memory -> topic -> fact -> canonical entity -> related canonical entity -> fact -> topic -> memory
    @param user_id: 用户 ID
    @param seed_memory_ids: 初始记忆 IDs 种子
    @param limit: 扩展的最大记忆数量
    @param min_relation_confidence: 关系置信度阈值，默认 0.5
    @param max_hops: 实体关系游走最大跳数，默认 1
    @param hop_decay: 每跳衰减系数，默认 0.8
    @param per_hop_limit: 每跳保留候选上限，默认 200
    @param min_walk_score: 游走最小累计分数阈值，默认 0.05
    @return: 扩展后的候选结构列表（按 score 降序）
    """

    if not user_id:
        logger.warning("Skip expansion due to empty user_id")
        return []

    seed_ids = _to_str_set(seed_memory_ids or set())
    # 空种子直接返回空集
    if not seed_ids:
        return []

    max_limit = _safe_limit(limit)
    min_confidence = _safe_confidence(min_relation_confidence)
    safe_max_hops = _safe_hops(max_hops)
    safe_hop_decay = _safe_decay(hop_decay)
    safe_per_hop_limit = _safe_limit(per_hop_limit, default=200)
    safe_min_walk_score = _safe_confidence(min_walk_score, default=0.05)

    logger.info(
        f"Start expansion user={user_id}, seed_count={len(seed_ids)}, limit={max_limit}, "
        f"min_conf={min_confidence}, max_hops={safe_max_hops}, hop_decay={safe_hop_decay}, "
        f"per_hop_limit={safe_per_hop_limit}, min_walk_score={safe_min_walk_score}"
    )

    # 第一段：memory -> topic -> fact -> canonical entity，定位扩展起点
    seed_topic_ids = _find_seed_topic_ids(user_id, seed_ids, max_limit)
    seed_fact_ids = _find_seed_fact_ids(user_id, seed_topic_ids, max_limit * 2)
    seed_canonical_ids = _find_seed_canonical_ids(user_id, seed_fact_ids)
    # 第二段：在实体关系图上做受限游走，得到相关实体分数
    canonical_scores = _walk_related_canonical_scores(
        user_id,
        seed_canonical_ids,
        max_hops=safe_max_hops,
        hop_decay=safe_hop_decay,
        min_relation_confidence=min_confidence,
        per_hop_limit=safe_per_hop_limit,
        min_walk_score=safe_min_walk_score,
    )
    # 第三段：canonical entity -> fact -> topic -> memory，回投到可召回记忆
    fact_scores = _fetch_related_fact_scores(user_id, canonical_scores)
    topic_scores = _fetch_related_topic_scores(user_id, fact_scores)
    memory_scores = _collect_related_memory_scores(user_id, topic_scores)

    candidates = [
        GraphExpansionCandidate(id=memory_id, score=max(0.0, min(1.0, float(score or 0.0))))
        for memory_id, score in sorted(memory_scores.items(), key=lambda item: (-item[1], item[0]))
        if memory_id not in seed_ids
    ]

    if candidates:
        logger.info(f"Expanded memory candidates by graph: +{len(candidates)}, top_score={candidates[0].score:.3f}")
    else:
        logger.info("expansion finished with no new memories")
    return candidates
