import json
from dataclasses import dataclass
from typing import Any, Iterable

from agile.utils import LogHelper

from src.core.db import get_db
from src.memory.graph.graph_node_models import EdgeRelation

logger = LogHelper.get_logger(title="[GRAPH_SEARCH]")
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
    rows = ",".join(f"('{relation}', {weight})" for relation, weight in EDGE_RELATION_WEIGHTS.items())
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
                               walk_params: tuple[float, float, str, int, float, float, int],
                               related_fact_params: tuple[str, ...],
                               related_topic_params: tuple[str, ...],
                               related_memory_refs_params: tuple[str, str, int]) -> tuple[str, tuple[Any, ...]]:
    """
    构建“种子记忆 → 主题 → 事实 → canonical 实体 → 多跳实体游走 → 反向召回记忆”的图扩展 SQL。

    流程说明：
    - `seed_memory`：把输入的记忆 ID 作为种子集合，只保留当前用户的数据
    - `seed_topic`：找出包含种子记忆的主题（`graph_topics.dialogue_ids` 命中）
    - `seed_fact`：在这些主题下聚合出相关事实（Fact）
    - `seed_canonical`：从事实映射到 canonical 实体，作为递归游走的起点
    - `relation_weights`：为不同 `edge_relation` 提供预定义权重
    - `walk`：按 `max_hops` 递归扩展实体关系，并结合 `hop_decay` / `min_relation_confidence` / `min_walk_score` / `per_hop_limit` 做剪枝
    - `related_canonical`：对同一 canonical 的多条路径做分数聚合
    - `related_fact` / `related_topic` / `related_memory_refs`：把相关 canonical 反向映射回事实、主题和记忆 ID
    - 最终 SELECT：排除种子记忆本身，按 `graph_score` 降序返回候选记忆

    额外约束：
    - 通过 `path` 数组避免回环
    - 通过 `max_hops` 控制最大游走深度
    - 通过 `per_hop_limit` 控制每一跳的扩展规模
    - 通过 `hop_decay` 控制越深层路径的分数衰减
    """
    sql = f"""
        WITH RECURSIVE seed_memory AS (
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
        walk AS (
            SELECT
                0 AS hop, -- 初始跳数 0（种子）
                sc.canonical_id, -- 种子实体
                ARRAY[sc.canonical_id]::text[] AS path, -- 记录游走路径，防环
                1.0::float8 AS cum_score -- 初始分数
            FROM seed_canonical sc
            WHERE sc.canonical_id IS NOT NULL

            UNION ALL
            -- 递归扩展下一跳
            SELECT
                w.hop + 1 AS hop, -- 当前跳数
                next_node.next_canonical_id AS canonical_id, -- 下一跳 canonical
                w.path || next_node.next_canonical_id AS path, -- 追加路径
                (
                    w.cum_score * %s
                    + COALESCE(gre.confidence, 0.0) * rw.relation_weight * POWER(%s, w.hop + 1)
                ) AS cum_score -- 分数 = 衰减系数 * 上一跳分数 + 新关系得分
            FROM graph_entity_relations gre
            JOIN walk w
              ON gre.source_canonical_id = w.canonical_id
              OR gre.target_canonical_id = w.canonical_id
            JOIN LATERAL (
                SELECT
                    CASE
                        WHEN gre.source_canonical_id = w.canonical_id THEN gre.target_canonical_id
                        ELSE gre.source_canonical_id
                    END AS next_canonical_id
            ) AS next_node ON TRUE
            JOIN relation_weights rw
              ON rw.edge_relation = gre.edge_relation
            WHERE gre.user_id = %s
              AND w.hop < %s
              AND (gre.confidence IS NULL OR gre.confidence >= %s)
              AND gre.source_canonical_id <> gre.target_canonical_id
              AND next_node.next_canonical_id IS NOT NULL
              AND NOT (next_node.next_canonical_id = ANY (w.path))
        ),
        walk_ranked AS (
            -- 按跳数截断候选
            SELECT canonical_id, cum_score
            FROM (
                SELECT
                    w.hop,
                    w.canonical_id,
                    w.cum_score,
                    ROW_NUMBER() OVER (
                        PARTITION BY w.hop
                        ORDER BY w.cum_score DESC NULLS LAST, w.canonical_id ASC
                    ) AS rn -- 跳内排序
                FROM walk w
                WHERE w.hop > 0
                  AND w.canonical_id IS NOT NULL
                  AND w.cum_score >= %s
            ) ranked
            WHERE rn <= %s
        ),
        related_canonical AS (
            -- 按 canonical 聚合多路径分数
            SELECT canonical_id, MAX(cum_score) AS graph_score
            FROM walk_ranked
            WHERE canonical_id IS NOT NULL
            GROUP BY canonical_id
        ),
        related_fact AS (
            -- canonical 反查 fact
            SELECT gfe.fact_id, MAX(rc.graph_score) AS graph_score
            FROM graph_fact_entities gfe
            JOIN related_canonical rc ON rc.canonical_id = gfe.canonical_id
            WHERE gfe.user_id = %s
            GROUP BY gfe.fact_id
        ),
        related_topic AS (
            -- fact 反查 topic
            SELECT gf.topic_id, MAX(rf.graph_score) AS graph_score
            FROM graph_facts gf
            JOIN related_fact rf ON rf.fact_id = gf.id
            WHERE gf.user_id = %s
              AND gf.topic_id IS NOT NULL
            GROUP BY gf.topic_id
        ),
        related_memory_refs AS (
            -- topic 反查记忆
            SELECT DISTINCT m.id AS memory_id, MAX(rt.graph_score) AS graph_score
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
        *walk_params,
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

    sql, params = _build_graph_expansion_sql(
        seed_memory_params=(user_id, sorted(seed_ids)),
        seed_topic_params=(user_id, max_limit),
        seed_fact_params=(user_id, max_limit * 2),
        seed_canonical_params=(user_id,),
        walk_params=(
            safe_hop_decay,
            safe_hop_decay,
            user_id,
            safe_max_hops,
            min_confidence,
            safe_min_walk_score,
            safe_per_hop_limit,
        ),
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
        logger.info(f"Expanded memory candidates by graph: +{len(candidates)}, top_score={candidates[0].score:.3f}")
    else:
        logger.info("expansion finished with no new memories")
    return candidates
