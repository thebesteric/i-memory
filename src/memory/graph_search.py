import json
from dataclasses import dataclass
from typing import Any, Iterable

from agile.utils import LogHelper

from src.core.db import get_db
from src.memory.graph.graph_node_models import EdgeRelation

logger = LogHelper.get_logger()
db = get_db()


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


@dataclass(frozen=True)
class GraphExpansionCandidate:
    id: str
    score: float


def _collect_topic_dialogue_ids(user_id: str, topic_ids: list[str], limit: int) -> set[str]:
    if not topic_ids:
        return set()
    rows = db.fetchall(
        """
        SELECT dialogue_ids
        FROM graph_topics
        WHERE user_id = %s
          AND id = ANY (%s)
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, id ASC
        LIMIT %s
        """,
        (user_id, topic_ids, _safe_limit(limit)),
    )
    dialogue_ids: set[str] = set()
    for row in rows:
        payload = row.get("dialogue_ids")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = []
        if isinstance(payload, list):
            dialogue_ids.update(_to_str_set(payload))
    return dialogue_ids


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


def _edge_relation_weights_sql() -> str:
    rows = ",\n            ".join(
        f"('{relation}', {weight})" for relation, weight in EDGE_RELATION_WEIGHTS.items()
    )
    return f"""
        SELECT *
        FROM (VALUES
            {rows}
        ) AS rw(edge_relation, relation_weight)
    """


def _build_graph_expansion_sql(*,
                               seed_memory_params: tuple[str, list[str]],
                               seed_topic_params: tuple[str, int],
                               seed_fact_params: tuple[str, int],
                               seed_canonical_params: tuple[str, ...],
                               related_edge_hits_params: tuple[str, float],
                               related_fact_params: tuple[str, ...],
                               related_topic_params: tuple[str, ...],
                               related_memory_refs_params: tuple[str, str, int]) -> tuple[str, tuple[Any, ...]]:
    sql = f"""
        WITH seed_memory AS (
            SELECT id
            FROM memories
            WHERE user_id = %s
              AND id = ANY (%s)
        ),
        seed_topic AS (
            SELECT gt.id
            FROM graph_topics gt
            WHERE gt.user_id = %s
              AND EXISTS (
                  SELECT 1
                  FROM seed_memory sm
                  WHERE gt.dialogue_ids ? sm.id
              )
            ORDER BY gt.updated_at DESC NULLS LAST, gt.created_at DESC NULLS LAST, gt.id ASC
            LIMIT %s
        ),
        seed_fact AS (
            SELECT gf.id
            FROM graph_facts gf
            JOIN seed_topic st ON st.id = gf.topic_id
            WHERE gf.user_id = %s
            ORDER BY gf.confidence DESC NULLS LAST, gf.updated_at DESC NULLS LAST, gf.created_at DESC NULLS LAST, gf.id ASC
            LIMIT %s
        ),
        seed_canonical AS (
            SELECT DISTINCT gfe.canonical_id
            FROM graph_fact_entities gfe
            JOIN seed_fact sf ON sf.id = gfe.fact_id
            WHERE gfe.user_id = %s
              AND gfe.canonical_id IS NOT NULL
        ),
        relation_weights AS (
            {_edge_relation_weights_sql()}
        ),
        related_edge_hits AS (
            SELECT
                CASE
                    WHEN gre.source_canonical_id = sc.canonical_id THEN gre.target_canonical_id
                    ELSE gre.source_canonical_id
                END AS canonical_id,
                gre.edge_relation,
                gre.confidence,
                gre.updated_at,
                gre.created_at,
                gre.id,
                rw.relation_weight,
                COALESCE(gre.confidence, 0.0) * rw.relation_weight AS graph_score
            FROM graph_entity_relations gre
            JOIN seed_canonical sc
              ON gre.source_canonical_id = sc.canonical_id
              OR gre.target_canonical_id = sc.canonical_id
            JOIN relation_weights rw
              ON rw.edge_relation = gre.edge_relation
            WHERE gre.user_id = %s
              AND (gre.confidence IS NULL OR gre.confidence >= %s)
              AND gre.source_canonical_id <> gre.target_canonical_id
        ),
        related_canonical AS (
            SELECT canonical_id, MAX(graph_score) AS graph_score
            FROM related_edge_hits
            WHERE canonical_id IS NOT NULL
            GROUP BY canonical_id
        ),
        related_fact AS (
            SELECT gfe.fact_id, MAX(rc.graph_score) AS graph_score
            FROM graph_fact_entities gfe
            JOIN related_canonical rc ON rc.canonical_id = gfe.canonical_id
            WHERE gfe.user_id = %s
            GROUP BY gfe.fact_id
        ),
        related_topic AS (
            SELECT gf.topic_id, MAX(rf.graph_score) AS graph_score
            FROM graph_facts gf
            JOIN related_fact rf ON rf.fact_id = gf.id
            WHERE gf.user_id = %s
              AND gf.topic_id IS NOT NULL
            GROUP BY gf.topic_id
        ),
        related_memory_refs AS (
            SELECT m.id AS memory_id, MAX(rt.graph_score) AS graph_score
            FROM graph_topics gt
            JOIN related_topic rt ON rt.topic_id = gt.id
            JOIN LATERAL jsonb_array_elements_text(COALESCE(gt.dialogue_ids, '[]'::jsonb)) d(memory_id) ON TRUE
            JOIN memories m ON m.id = d.memory_id
            WHERE gt.user_id = %s
              AND m.user_id = %s
            GROUP BY m.id
        )
        SELECT rmr.memory_id AS id, COALESCE(rmr.graph_score, 0.0) AS graph_score
        FROM related_memory_refs rmr
        WHERE NOT EXISTS (
            SELECT 1
            FROM seed_memory sm
            WHERE sm.id = rmr.memory_id
        )
        ORDER BY graph_score DESC NULLS LAST, id ASC
        LIMIT %s
    """
    params: tuple[Any, ...] = (
        *seed_memory_params,
        *seed_topic_params,
        *seed_fact_params,
        *seed_canonical_params,
        *related_edge_hits_params,
        *related_fact_params,
        *related_topic_params,
        *related_memory_refs_params,
    )
    return sql, params


def expand_candidate_ids_via_graph(
        user_id: str,
        seed_memory_ids: set[str],
        limit: int,
        min_relation_confidence: float = 0.5,
) -> list[GraphExpansionCandidate]:
    """
    通过 Topic/Fact/Entity 图谱链路扩展候选记忆：
    memory -> topic -> fact -> canonical entity -> related canonical entity -> fact -> topic -> memory
    @param user_id: 用户 ID
    @param seed_memory_ids: 初始记忆 IDs 种子
    @param limit: 扩展的最大记忆数量
    @param min_relation_confidence: 关系置信度阈值，默认 0.5
    @return: 扩展后的候选结构列表（按 score 降序）
    """

    if not user_id:
        logger.warning("[GRAPH_SEARCH] skip expansion due to empty user_id")
        return []

    seed_ids = _to_str_set(seed_memory_ids or set())
    # 空种子直接返回空集
    if not seed_ids:
        return []

    max_limit = _safe_limit(limit)
    min_confidence = _safe_confidence(min_relation_confidence)

    logger.info(
        f"[GRAPH_SEARCH] start expansion user={user_id}, seed_count={len(seed_ids)}, limit={max_limit}, min_conf={min_confidence}"
    )
    sql, params = _build_graph_expansion_sql(
        seed_memory_params=(user_id, sorted(seed_ids)),
        seed_topic_params=(user_id, max_limit),
        seed_fact_params=(user_id, max_limit * 2),
        seed_canonical_params=(user_id,),
        related_edge_hits_params=(user_id, min_confidence),
        related_fact_params=(user_id,),
        related_topic_params=(user_id,),
        related_memory_refs_params=(user_id, user_id, max_limit * 3),
    )
    rows = db.fetchall(sql, params)

    candidates = [
        GraphExpansionCandidate(
            id=str(row.get("id")),
            score=max(0.0, min(1.0, float(row.get("graph_score") or 0.0))),
        )
        for row in rows
        if row.get("id") and str(row.get("id")) not in seed_ids
    ]

    if candidates:
        logger.info(
            f"[GRAPH_SEARCH] Expanded memory candidates by graph: +{len(candidates)}, top_score={candidates[0].score:.3f}"
        )
    else:
        logger.info("[GRAPH_SEARCH] expansion finished with no new memories")
    return candidates
