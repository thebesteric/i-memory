import asyncio
import datetime
import json
import math
import time
import uuid
from typing import Any, Dict, List

from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.search import BM25Searcher
from agile.utils import LogHelper, timing

from src.core.components import get_sector_classifier, get_vector_store, MEMORIES_CACHE, get_embed_model
from src.core.config import env
from src.core.constants import SECTOR_RELATIONSHIPS, HYBRID_PARAMS, get_dynamic_sector_weights
from src.core.db import get_db
from src.core.mem_ops import mem_ops
from src.core.extract_essence import ExtractEssence
from src.core.score import compute_tag_match_score, compute_hybrid_score
from src.core.sector_classify import SECTOR_CONFIGS, ClassifyResult, SectorClassifier
from src.core.vector.base_vector_store import VectorSearch, BaseVectorStore
from src.core.waypoints import Waypoints, Expansion
from src.core import user_ops
from src.exceptions.exceptions import UserNotFoundError
from src.memory.decay import Decay
from src.memory.embed import embed_multi_sector, calc_mean_vec, embed, embed_batch
from src.memory import graph_search
from src.memory.memory_models import IMemoryFilters, IMemoryItemDebugInfo, IMemoryItemInfo, IMemoryUserIdentity, \
    IMemoryUser, QARole, \
    IMemoryFiltersConfig, IMemorySearchResult, IMemoryGraphConfig
from src.core.user_summary import update_user_summary
from src.memory.profile import user_profile_ops
from src.memory.session import session_ops
from src.memory.session.session_models import Sessions
from src.ops.dynamic_memory import calc_cross_sector_resonance_score, apply_retrieval_trace_reinforcement_to_memory, \
    propagate_associative_reinforcement_to_linked_nodes
from src.tools.chunking import chunk_text
from src.tools.keyword import compute_keyword_overlap, compute_token_overlap
from src.tools.text import canonical_token_set
from src.tools.vectors import vec_to_buf, cos_sim

logger = LogHelper.get_logger(title="[HSG]")
waypoints = Waypoints()
db = get_db()
decay = Decay(reinforce_on_query=True, regeneration_enabled=True)
vector_store: BaseVectorStore = get_vector_store()
sector_classifier: SectorClassifier = get_sector_classifier()
embed_model: BaseEmbedModel = get_embed_model()
bm25_searcher = BM25Searcher(id_field="id", content_field="content")


@timing
async def embed_query_for_all_sectors(query: str, sectors: List[str]) -> Dict[str, List[float]]:
    if not sectors: return {}
    try:
        vectors = await embed_batch(query, sectors)
        return {sector: vector for sector, vector in zip(sectors, vectors)}
    except Exception as e:
        raise RuntimeError(f"[HSG] embed_batch failed, fallback to per-sector embedding: {e}")


def compress_vec_for_storage(vec: List[float], target_dim: int) -> List[float]:
    """
    将一个高维浮点向量压缩为指定维度（target_dim）的低维向量，常用于向量存储降维
    @param vec: 输入的高维浮点向量
    @param target_dim: 目标维度
    @return: 压缩后的低维向量
    """
    # 如果原始向量长度小于等于目标维度，直接返回原向量
    if len(vec) <= target_dim: return vec
    comp = [0.0] * target_dim
    # 计算每个桶的大小（原始向量长度除以目标维度）
    bucket_size = len(vec) / target_dim
    for i in range(target_dim):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        s = 0.0
        c = 0
        for j in range(start, min(end, len(vec))):
            s += vec[j]
            c += 1
        # 每个桶内的元素求平均，得到新向量的每一维
        comp[i] = s / c if c > 0 else 0.0
    # 对压缩后的向量做归一化（L2范数），保证数值稳定
    n = math.sqrt(sum(x * x for x in comp))
    if n > 0:
        for i in range(target_dim): comp[i] /= n
    return comp


async def add_hsg_memory(user_identity: IMemoryUserIdentity,
                         content: str,
                         tags: List[str] = None,
                         metadata: Any = None,
                         qa_role: QARole | None = None,
                         batch_id: str | None = None) -> Dict[str, Any]:
    """
    添加一条 Hierarchical Semantic Graph 记忆（数据库 + 向量存储、按扇区（sectors）分层组织记忆）
    :param content: 记忆内容
    :param tags: 标签
    :param metadata: 元数据
    :param user_identity: 用户身份
    :param qa_role: QA 角色（human/assistant）
    :param batch_id: 批次 ID
    :return:
    """
    # 用户合法性检查
    user_identity.check_legality()

    # 获取用户，若不存在则创建一条新用户记录
    user: IMemoryUser | None = await user_ops.get_user(user_identity=user_identity)
    if not user:
        raise UserNotFoundError(user_identity)

    # 角色合法性检查
    if qa_role and qa_role not in ("human", "assistant"):
        raise ValueError("qa_role must be one of: human, assistant")

    # 需传 qa_role，问答配对（尝试复用最近一条未配对 human 的 qa_pair_id）
    qa_pair_id = _resolve_auto_qa_linking(user, qa_role)

    # 若未显式传入 batch_id，则自动继承 qa_pair_id，使每轮问答天然可批量删除
    if not batch_id and qa_pair_id:
        batch_id = qa_pair_id

    # 生成内容的嵌入向量
    vec = await embed_model.embed(content)

    # 构建 SQL 查询，查询该用户的记忆，包含租户和项目过滤（如果提供了租户和项目信息），并且只查询有向量的记忆
    user_memories = mem_ops.find_mem_by_user(user, order_by=["t.salience DESC", "t.last_seen_at DESC"], limit=100)

    # 初始化最佳相似记忆（相似度，记忆记录）
    best_sim_mem_similarity = tuple()
    if user_memories:
        for user_memory in user_memories:
            v_list_json = json.loads(user_memory["v"])
            v = [float(s) for s in v_list_json]
            similarity = embed_model.similarity(vec, v)
            # 找到相似度最高的记忆
            if not best_sim_mem_similarity or similarity > best_sim_mem_similarity[0]:
                best_sim_mem_similarity = (similarity, user_memory)

    if best_sim_mem_similarity:
        best_sim_content = best_sim_mem_similarity[1]["content"]
        best_sim_content = best_sim_content if len(best_sim_content) <= 20 else best_sim_content[:20] + "..."
        logger.info(f"Maybe best similar memory: Sim: {best_sim_mem_similarity[0]}, Content: {best_sim_content}")

    # 当前时间
    now = datetime.datetime.now()

    # 存在相似记忆 && 相似度 >= 0.95
    if best_sim_mem_similarity and best_sim_mem_similarity[0] >= env.SIMILARITY_THRESHOLD:
        """
        如果发现内容高度相似（相似度 >= 0.95）
        不会新建一条记忆，而是提升已有记忆的显著性，并更新时间戳，表示这条记忆再次被“关注”或“激活”
        """
        best_sim_mem = best_sim_mem_similarity[1]
        content = best_sim_mem["content"]
        content = content if len(content) <= 20 else content[:20] + "..."
        logger.info(
            f"Found similar memory {best_sim_mem['id']} with {best_sim_mem_similarity[0]} for User: {user.id}, Content: {content}"
        )
        # 提升显著性，但不超过 1.0
        boost = min(1.0, (best_sim_mem["salience"] or 0) + 0.15)
        # 更新最后访问时间和显著性
        db.execute("UPDATE memories SET last_seen_at=%s, salience=%s, updated_at=%s WHERE id=%s",
                   (now, boost, now, best_sim_mem['id']))
        db.commit()
        return {
            "id": best_sim_mem["id"],
            "primary_sector": best_sim_mem["primary_sector"],
            "sectors": [best_sim_mem["primary_sector"]],
            "deduplicated": True
        }

    # 对内容分段，判断是否需要分块存储（当内容过长时），并统计总的令牌数
    chunks, total_token = chunk_text(content)
    use_chunks = len(chunks) > 1

    # 更新字符数和估计的令牌数
    metadata.update({
        "char_count": len(content),
        "estimated_tokens": total_token,
    })

    # 文本分类：判断内容所属的主/辅 sector（语义、情感、程序、事件、反思等）
    cls_ret = await sector_classifier.classify(content=content, metadata=metadata)
    all_secs = [cls_ret.primary] + cls_ret.additional

    try:
        current_seg_result = db.fetchone("SELECT current_segment FROM segment FOR UPDATE")
        cur_seg = current_seg_result["current_segment"] if current_seg_result else 0
        # 获取当前 segment 中的记忆总数量
        cnt_res = db.fetchone("SELECT count(*) as c FROM memories WHERE segment=%s", (cur_seg,))
        # 如果当前 SECTOR 记忆数达到上限，则切换到下一个 SECTOR
        if cnt_res["c"] >= env.SECTOR_SIZE:
            cur_seg += 1
            db.execute("UPDATE segment SET current_segment=%s, updated_at=NOW()", (cur_seg,))
            logger.info(f"Rotated to segment [{cur_seg}]")

        # 调用 extract_essence，生成摘要（内容长度 > 1000，模型调用）
        essence = await ExtractEssence(content=content, max_len=env.SUMMARY_MAX_LENGTH).extract()

        # 获取主 sector 的配置
        sec_cfg = SECTOR_CONFIGS[cls_ret.primary]
        # 始化记忆的显著性（salience）分数
        # 基础分为 0.4，每多一个辅扇区（cls_ret.additional），显著性加 0.1
        # 最终显著性限定在 0.0~1.0 之间，防止过高或为负
        init_sal = max(0.0, min(1.0, 0.4 + 0.1 * len(cls_ret.additional)))

        # 调用 mem_ops.ins_mem，将记忆内容、摘要、sector、标签、元数据等插入数据库
        mid = str(uuid.uuid4())
        mem_ops.ins_mem(
            id=mid,
            user_id=user.id,
            segment=cur_seg,
            content=essence,
            primary_sector=cls_ret.primary,
            sectors=json.dumps(all_secs or [], ensure_ascii=False),
            tags=json.dumps(tags or [], ensure_ascii=False),
            meta=json.dumps(metadata or {}, ensure_ascii=False),
            created_at=now,
            updated_at=now,
            last_seen_at=now,
            salience=init_sal,
            decay_lambda=sec_cfg.decay_lambda,
            version=1,
            mean_dim=None,
            mean_vec=None,
            compressed_vec=None,
            feedback_score=0,
            qa_role=qa_role,
            qa_pair_id=qa_pair_id,
            batch_id=batch_id
        )

        # 调用 embed_multi_sector，对内容进行多 sector 嵌入，生成向量
        emb_res: List[Dict[str, Any]] = await embed_multi_sector(
            user_id=user.id,
            mem_id=mid,
            txt=content,
            secs=all_secs,
            chunks=chunks if use_chunks else None
        )
        tasks = []
        for r in emb_res:
            # 存储每个 sector 的向量到向量库
            task = vector_store.store_vector(mid, r["sector"], r["vector"], r["dim"], user)
            tasks.append(task)

        # 并发执行所有任务
        if tasks:
            await asyncio.gather(*tasks)

        # 计算所有 sector 的均值向量
        mean_vec = calc_mean_vec(emb_res, all_secs)
        # 将一个浮点数列表（向量）序列化为二进制字节流（bytes）
        mean_buf = vec_to_buf(mean_vec)
        # 更新记忆的均值向量和维度
        db.execute("UPDATE memories SET mean_dim=%s, mean_vec=%s WHERE id=%s", (len(mean_vec), mean_buf, mid))

        # 若向量维度大于 128，则压缩存储
        if len(mean_vec) > 128:
            # 压缩均值向量到 128 维
            comp = compress_vec_for_storage(mean_vec, 128)
            # 将压缩向量序列化为二进制字节流（bytes）
            comp_mean_buf = vec_to_buf(comp)
            # 更新记忆的压缩向量
            db.execute("UPDATE memories SET compressed_vec=%s WHERE id=%s", (comp_mean_buf, mid))

        # 建立 Waypoint 关联，为新记忆建立与其他记忆的关联
        await waypoints.create_single_waypoint(mid, mean_vec, now, user)
        # 更新用户摘要
        await update_user_summary(user)

        logger.info(
            f"Added memory {mid} for User: {user.id}, Primary Sector: {cls_ret.primary}, Additional Sectors: {cls_ret.additional}, Salience: {init_sal}"
        )

        # 返回新记忆的 id、内容、sector、分段数、salience 等信息
        return {
            "id": mid,
            "content": content,
            "primary_sector": cls_ret.primary,
            "sectors": all_secs,
            "chunks": len(chunks),
            "salience": init_sal,
            "qa_role": qa_role,
            "qa_pair_id": qa_pair_id
        }
    except Exception as e:
        raise e


@timing
async def reinforce_memories(effective_k_list):
    """
    强化记忆
    :param effective_k_list: 需要强化的记忆列表
    :return:
    """
    for _item in effective_k_list:
        try:
            reinforcement_sal = await apply_retrieval_trace_reinforcement_to_memory(_item.id, _item.salience)
            now = datetime.datetime.now()
            db.execute("UPDATE memories SET salience=%s, last_seen_at = %s WHERE id = %s",
                       (reinforcement_sal, now, _item.id))

            # 有图遍历路径时
            if len(_item.path) > 1:
                wps_rows = db.fetchall("SELECT dst_id, weight FROM waypoints WHERE src_id=%s", (_item.id,))
                wps = [{"target_id": row["dst_id"], "weight": row["weight"]} for row in wps_rows]

                # 向关联节点传播（得到关联节点新的显著性）
                pru = await propagate_associative_reinforcement_to_linked_nodes(_item.id, reinforcement_sal, wps)
                for u in pru:
                    # 获取关联记忆
                    linked_mem = mem_ops.get_mem(u["node_id"])
                    if linked_mem:
                        # “当前时间” 与 “关联记忆最后访问时间” 的间隔天数
                        time_diff = (now - linked_mem["last_seen_at"]).total_seconds() / 86400.0
                        # 自然指数函数 math.exp 生成一个衰减系数 decay_fact（衰减因子），核心作用是将「关联记忆最后访问时间与当前时间的间隔天数」转化为 0~1 之间的权重值
                        # 时间间隔越久，衰减因子越小，对应记忆的权重 / 影响力越低
                        decay_fact = math.exp(-0.02 * time_diff)
                        # 上下文增强系数：基于记忆显著性差异和时间衰减的得分调整项，用于精细化修正基础匹配得分，放大优质记忆的优势、降低低质记忆的权重
                        ctx_boost = HYBRID_PARAMS["gamma"] * (
                                reinforcement_sal - (linked_mem["salience"] or 0)) * decay_fact
                        # 更新关联记忆的显著性
                        new_sal = max(0.0, min(1.0, (linked_mem["salience"] or 0) + ctx_boost))
                        # 更新关联记忆的显著性和最后访问时间
                        db.execute("UPDATE memories SET salience = %s, last_seen_at = %s WHERE id = %s",
                                   (new_sal, now, u["node_id"]))

            # 记录查询命中事件并触发动态更新
            await decay.on_query_hit(_item.id, _item.primary_sector, lambda t: embed(t, _item.primary_sector))
        except Exception as e:
            logger.warning(f"Reinforce_memories failed for memory {_item.id}: {e}")


@timing
async def query_hsg_memories(query: str, top_k: int = 10, filters: IMemoryFilters = None) -> IMemorySearchResult:
    """
    基于混合评分机制（内容相似度、关键词重叠、路标关联、图检索、时间衰减等）进行记忆检索
    @param query: 查询文本
    @param top_k: 返回的记忆数量
    @param filters: 过滤和调节参数
    @return: 返回符合查询条件的记忆列表
    """
    start_q = time.time()
    decay.inc_q()
    filters = filters or IMemoryFilters(user_identity=IMemoryUserIdentity())
    filters.config = filters.config or IMemoryFiltersConfig()
    graph_cfg = filters.config.graph
    if graph_cfg.type == "recall":
        graph_cfg = IMemoryGraphConfig.recall_first()
    elif graph_cfg.type == "precision":
        graph_cfg = IMemoryGraphConfig.precision_first()

    user_identity = filters.user_identity
    # 用户合法性检查
    user_identity.check_legality()

    user = await user_ops.get_user(user_identity=user_identity, using_cache=True)
    if not user:
        raise UserNotFoundError(user_identity)

    try:
        # 检查 60 秒内的查询缓存，命中则直接返回
        cache_key = f"{query}:{top_k}:{filters.model_dump_json() if filters else '{}'}"
        entry = MEMORIES_CACHE.get(cache_key)
        if isinstance(entry, IMemorySearchResult):
            return entry
        if entry is not None:
            logger.warning(f"Ignore invalid cache payload for key={cache_key}: {type(entry)}")

        # 判断查询属于哪个扇区
        query_classify: ClassifyResult = await sector_classifier.classify(content=query)
        # 提取查询关键词 token 集合
        query_tokens = canonical_token_set(query)
        # 确定检索扇区范围
        sectors = (filters.sectors if filters and filters.sectors else None) or list(SECTOR_CONFIGS.keys())
        if not sectors:
            sectors = ["semantic"]
        # 为各扇区生成嵌入向量
        query_embed_with_sectors = await embed_query_for_all_sectors(query, sectors)

        # 动态权重调
        primary_sector = query_classify.primary
        dynamic_sector_weights = get_dynamic_sector_weights(primary_sector=primary_sector)

        # 召回的记忆 ID 集合
        ids = set()

        # =========== 第一路召回（可选）：BM25 召回（基于关键词匹配的传统检索方法，补充向量召回可能遗漏的相关记忆）===========
        bm25_ids = set()
        # 是否开启了 BM25 检索
        if filters.config.bm25_enable:
            bm25_memories = mem_ops.find_mem_by_user(
                user=user,
                order_by=["t.created_at DESC"],
                limit=2000
            )
            bm25_search_timing_wrapped = timing(bm25_searcher.search)
            bm25_ids = set(bm25_search_timing_wrapped(query, docs=bm25_memories, top_k=top_k * 3))

        # =========== 第二路召回：向量召回（相似度匹配逻辑）===========
        # 存储各扇区检索结果
        sector_result = {}
        vector_search_ids = set()
        for sector in sectors:
            # 获取该扇区的查询向量
            query_vector = query_embed_with_sectors[sector]
            # 每个扇区返回 top_k × 3 个候选
            res: List[VectorSearch] = await vector_store.search(user, query_vector, sector, top_k * 3)
            vector_search_ids.update([r.id for r in res])
            sector_result[sector] = res

        # 合并召回 ID，去重
        for sector, result in sector_result.items():
            for vector_search in result:
                ids.add(vector_search.id)
        ids.update(bm25_ids)

        # 收集所有候选的相似度
        all_sims = []
        # 候选记忆 ID 集合
        for sector, result in sector_result.items():
            for vector_search in result:
                all_sims.append(vector_search.similarity)

        # 计算平均相似度，判断检索质量
        avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0
        # 自适应扩展规模，平均相似度越低，扩展规模越大，以提高召回率
        adapt_exp = math.ceil(0.3 * top_k * (1 - avg_sim))
        # 实际的的查询数量：最终检索规模
        effective_k = top_k + adapt_exp

        # =========== 第三路召回：摘要召回（L2）===========
        sessions: Sessions = await session_ops.session_search(user, query, top_k=effective_k)
        session_search_dialogue_ids = set()
        for session in sessions.sessions:
            session_search_dialogue_ids.update(session.dialogue_ids)
        ids.update(session_search_dialogue_ids)

        # =========== 第四路召回：路标扩展召回（按需）===========
        expansion: List[Expansion] = []
        waypoint_expanded_ids = set()
        # 若平均相似度 < 0.55（低置信），触发路标遍历扩展
        if avg_sim < 0.55:
            # 通过 waypoint 关系扩展
            expansion = await waypoints.expand_via_waypoints(list(ids), effective_k * 2)
            # 加入候选集
            for exp in expansion:
                waypoint_expanded_ids.add(exp.id)
        ids.update(waypoint_expanded_ids)

        # =========== 第五路召回：图扩展召回（可选）===========
        graph_expanded_ids = set()
        graph_candidate_scores = {}
        if graph_cfg.enable:
            seed_ids = set(ids)
            # 通过 topic/fact/entity 关系链进行扩展
            graph_candidates = graph_search.expand_candidate_ids_via_graph(
                user_id=user.id,
                seed_memory_ids=seed_ids,
                limit=effective_k * 2,
                min_relation_confidence=graph_cfg.min_relation_confidence,
                max_hops=graph_cfg.max_hops,
                hop_decay=graph_cfg.hop_decay,
                per_hop_limit=graph_cfg.per_hop_limit,
                min_walk_score=graph_cfg.min_walk_score,
            )
            graph_expanded_ids = {c.id for c in graph_candidates}
            graph_candidate_scores = {c.id: c.score for c in graph_candidates}
            ids.update(graph_expanded_ids)

        # 获取候选记忆内容
        memories = mem_ops.find_mem_by_ids(list(ids))

        # 计算每个候选的关键词重叠分数
        kw_scores = {}
        # 记录记忆首次召回来源
        mem_from = {}
        # 记忆来源
        from_rules = [
            (bm25_ids, "bm25"),
            (vector_search_ids, "vector"),
            (session_search_dialogue_ids, "session"),
            (waypoint_expanded_ids, "waypoint"),
            (graph_expanded_ids, "graph"),
        ]
        for mem in memories:
            mem_id = mem["id"]
            # 计算每个候选的关键词重叠分数
            overlap = compute_keyword_overlap(query_tokens, mem["content"])
            # 关键词重叠分数占比 15%
            kw_scores[mem_id] = overlap * 0.15
            # 记录记忆首次召回来源
            mem_from[mem_id] = next(
                (name for rule_ids, name in from_rules if mem_id in rule_ids),
                "unknown"
            )

        # 数据准备与过滤
        res_list: List[IMemoryItemInfo] = []
        for mem in memories:
            # 记忆 ID
            mid = mem["id"]
            # 显著性 < 过滤阈值，跳过
            if filters.min_salience and mem["salience"] < filters.min_salience:
                continue

            # 如果记忆的用户 ID 与查询用户 ID 不匹配，则跳过该记忆
            if mem["user_id"] != user.id:
                continue

            # 多向量融合相似度评分（该评分反映了记忆在多个维度（扇区）上与查询的相关性）
            mvf = await calc_multi_vec_fusion_score(mid, query_embed_with_sectors, dynamic_sector_weights)
            # 跨扇区共振分数（根据记忆类型间的相互作用强度计算激活程度）
            csr = await calc_cross_sector_resonance_score(mem["primary_sector"], query_classify.primary, mvf)

            # 取最高相似度
            best_sim = csr
            for s, rlist in sector_result.items():
                for vector_search in rlist:
                    if vector_search.id == mid and vector_search.similarity > best_sim:
                        best_sim = vector_search.similarity

            # 记忆的扇区
            mem_sec = mem["primary_sector"]
            # 实际文本分类扇区
            query_sec = query_classify.primary
            # 惩罚值
            penalty = 1.0
            # 扇区不匹配，应用惩罚
            if mem_sec != query_sec:
                penalty = SECTOR_RELATIONSHIPS.get(query_sec, {}).get(mem_sec, 0.3)

            # 相似度惩罚调整
            sim_adjust = best_sim * penalty
            # 扩展的记忆项
            expansion_mem = next((e for e in expansion if e.id == mid), None)
            # waypoint 权重
            waypoint_weight = min(1.0, max(0.0, expansion_mem.weight if expansion_mem else 0.0))

            # 计算最后一次访问距今的天数
            last_seen_at = mem["last_seen_at"]
            now = datetime.datetime.now()
            time_diff = now - last_seen_at
            days = time_diff.total_seconds() / 86400.0

            # 计算记忆衰减后的显著性（艾宾浩斯遗忘曲线）
            salience = decay.calc_decay(mem["primary_sector"], mem["salience"], days)
            # 标准化文本内容 token 集合
            memory_tokens = canonical_token_set(mem["content"])
            # 关键词重叠
            token_overlap = compute_token_overlap(query_tokens, memory_tokens)
            # 时效性分数
            rec_sc = decay.calc_recency_score_decay(last_seen_at)
            # 标签匹配得分
            tag_match_score = await compute_tag_match_score(mem, query_tokens)
            # 图扩展分数：作为额外排序信号，避免高质量图候选被完全淹没
            graph_score = graph_candidate_scores.get(mid, 0.0)
            graph_bonus = min(0.12, graph_score * 0.12)

            # 组合权重最终得分: 相似度(35%) × 扇区惩罚，关键词重叠(20%)，Waypoint权重(15%)，时效性(10%)，标签匹配(20%)
            final_score = compute_hybrid_score(sim=sim_adjust,
                                               tok_ov=token_overlap,
                                               wp_wt=waypoint_weight,
                                               rec_sc=rec_sc,
                                               kw_score=kw_scores.get(mid, 0),
                                               tag_match=tag_match_score)
            final_score = max(0.0, min(1.0, final_score + graph_bonus))

            metadata = json.loads(mem["meta"]) or {}
            metadata.update({
                "type": "memory",
                "from": mem_from.get(mid, "unknown"),
                "graph_score": graph_score,
                "graph_bonus": graph_bonus,
            })

            # 构建结果项
            item = IMemoryItemInfo(
                id=mid,
                content=mem["content"],
                score=final_score,
                primary_sector=mem["primary_sector"],
                path=expansion_mem.path if expansion_mem else [mid],
                salience=salience,
                last_seen_at=mem["last_seen_at"],
                created_at=mem["created_at"],
                tags=json.loads(mem["tags"] or "[]"),
                qa_role=mem.get("qa_role"),
                qa_pair_id=mem.get("qa_pair_id"),
                metadata=metadata
            )
            # 调试信息
            if filters.config.debug:
                item.debug = IMemoryItemDebugInfo(
                    sim_adjust=sim_adjust,
                    token_overlap=token_overlap,
                    recency_score=rec_sc,
                    waypoint_weight=waypoint_weight,
                    tag_match_score=tag_match_score,
                    penalty_score=penalty
                )
            # 加入结果列表
            res_list.append(item)

        # QA 模式：prefer/qa 时尝试配对提升
        if filters.query_mode and filters.query_mode in ("qa", "prefer"):
            res_list = _promote_qa_assistant_answer(res_list, query_classify.primary)

        # 按分数降序
        res_list.sort(key=lambda x: x.score, reverse=True)
        # 取 effective_k 条记录
        effective_k_list: List[IMemoryItemInfo] = res_list[:effective_k]

        # 命中记忆强化异步执行
        try:
            asyncio.create_task(reinforce_memories(effective_k_list))
        except Exception as e:
            logger.warning(f"Failed to schedule reinforce_memories async task: {e}")

        logger.info(
            f"Query processed in {time.time() - start_q:.3f} seconds. Query: '{query}' Expected: {effective_k}, Actual: {len(effective_k_list)} returned."
        )

        # 查询用户画像
        user_profile = None
        if filters.config and filters.config.user_profile_enable:
            user_profile = await user_profile_ops.get_user_profile(user, query_cache=True)

        # 是否启用会话摘要
        effective_k_list = extract_sessions_if_necessary(sessions, filters, res_list, effective_k_list)

        # 封装为 IMemorySearchResult，并返回结果
        memory_search_result = IMemorySearchResult(user_profile=user_profile, memories=effective_k_list)
        MEMORIES_CACHE.set(cache_key, memory_search_result)
        return memory_search_result

    finally:
        decay.dec_q()


def extract_sessions_if_necessary(sessions: Sessions, filters: IMemoryFilters, memories: list[IMemoryItemInfo],
                                  effective_k_list: List[IMemoryItemInfo]) -> List[IMemoryItemInfo]:
    """
    从 sessions 中提取会话级记忆，并根据配置决定是否将会话记忆加入结果列表，以及是否进行会话去重处理
    :param sessions:
    :param filters:
    :param memories:
    :param effective_k_list:
    :return:
    """
    if filters.config and filters.config.session_summary_enable:
        session_memories: list[IMemoryItemInfo] = []
        for session in sessions.sessions:
            session_dialogue_ids = session.dialogue_ids or []
            session_similarity = max(0.0, min(1.0, float(session.similarity or 0.0)))
            session_dialogue_count = len(session_dialogue_ids)
            # 计算 session 平均得分
            session_total_score = 0.0
            session_total_salience = 0.0
            session_tags = set()
            session_mem_hits = 0
            for m in memories:
                if m.id in session_dialogue_ids:
                    session_total_score += m.score
                    session_total_salience += m.salience
                    session_tags.update(m.tags)
                    session_mem_hits += 1
            # 平均得分 = 会话内命中记忆得分总和 / 会话内记忆命中数
            mem_avg_score = max(0.0, (session_total_score / session_mem_hits) if session_mem_hits > 0 else 0.0)
            # 平均显著性 = 会话内命中记忆显著性总和 / 会话内记忆命中数
            session_avg_salience = max(
                0.0,
                (session_total_salience / session_mem_hits) if session_mem_hits > 0 else 0.0
            )
            # 覆盖率 = 当前候选中实际命中的原始记忆数 / 该 session 总 dialogue 数
            coverage_ratio = (session_mem_hits / session_dialogue_count) if session_dialogue_count > 0 else 0.0
            # session 分数融合：以摘要向量相似度为主，原始 mem 平均分和覆盖率为辅
            session_score = max(
                0.0,
                min(1.0, 0.55 * session_similarity + 0.30 * mem_avg_score + 0.15 * coverage_ratio)
            )

            # 构建 session 级的记忆
            session_mem = IMemoryItemInfo(
                id=f"session:{session.id}",
                content=session.summary,
                score=session_score,
                primary_sector="semantic",
                path=session_dialogue_ids,
                salience=session_avg_salience,
                tags=list(session_tags),
                metadata={
                    "type": "session",
                    "from": "session",
                    "session_id": session.id,
                    "session_similarity": session_similarity,
                    "session_mem_hits": session_mem_hits,
                    "session_dialogue_count": session_dialogue_count,
                    "session_coverage_ratio": coverage_ratio,
                    "session_mem_avg_score": mem_avg_score,
                }
            )
            session_memories.append(session_mem)

            # 会话去重
            if filters.config.session_dedup_enable:
                # 判断 dialogue_id 是否在 effective_k_list 中，如果在则移除，确保会话记忆与其包含的对话记忆不重复
                effective_k_list = [
                    mem for mem in effective_k_list
                    if mem.id not in session.dialogue_ids
                ]

        # 按 score 从高到低排序
        effective_k_list.extend(session_memories)
        effective_k_list.sort(key=lambda x: x.score, reverse=True)

    return effective_k_list


async def calc_multi_vec_fusion_score(mid: str, qe: Dict[str, List[float]], weights: Dict[str, float]) -> float:
    """
    计算多向量融合相似度评分
    通过对记忆的各扇区向量与查询向量进行余弦相似度计算，并根据权重进行加权平均，得到最终融合相似度评分
    该评分反映了记忆在多个维度（扇区）上与查询的相关性
    :param mid: 记忆 ID
    :param qe: 查询向量字典，键为扇区名称，值为对应的向量列表
    :param weights: 权重字典，键为权重名称，值为对应的权重值
    :return: 融合相似度评分，浮点数
    """

    # 获取记忆的所有向量
    vecs = await vector_store.get_vectors_by_id(mid)
    # 加权相似度总和（分子）
    s = 0.0
    # 有效权重总和（分母），用于归一化
    tot = 0.0
    # 权重映射，将权重名称映射到对应的扇区名称
    weights_mapping = {
        "semantic": weights.get("semantic_dimension_weight", 0),
        "emotional": weights.get("emotional_dimension_weight", 0),
        "procedural": weights.get("procedural_dimension_weight", 0),
        "episodic": weights.get("episodic_dimension_weight", 0),
        "reflective": weights.get("reflective_dimension_weight", 0),
    }

    for v in vecs:
        # 获取记忆在该扇区的向量
        qv = qe.get(v.sector)
        if not qv:
            continue
        # 计算记忆向量与查询向量的余弦相似度
        sim = cos_sim(v.vector, qv)
        # 获取该扇区的权重，默认为 0.5
        wgt = weights_mapping.get(v.sector, 0.5)
        # 加权累加相似度
        s += sim * wgt
        # 累加权重
        tot += wgt

    # 返回归一化的融合相似度评分
    return s / tot if tot > 0 else 0.0


def _promote_qa_assistant_answer(items: List[IMemoryItemInfo], query_sector: str) -> List[IMemoryItemInfo]:
    """
    在 prefer/qa 模式下，优先把与高分问题匹配的 assistant 回答提升到顶部。
    配对关系仅依赖写入时存储的 qa_pair_id，查询方无需传入会话标识。
    """
    if not items:
        return items

    # 从召回结果中找分数最高的 human 记忆
    best_human = next(
        (it for it in sorted(items, key=lambda x: x.score, reverse=True)
         if it.qa_role == "human"),
        None
    )
    if not best_human:
        return items

    # 尝试查找与该 human 记忆配对的 assistant 记忆
    pair_row = None
    if best_human.qa_pair_id:
        pair_row = db.fetchone(
            """
            SELECT *
            FROM memories
            WHERE qa_pair_id = %s
              AND qa_role = 'assistant'
            ORDER BY created_at DESC LIMIT 1
            """,
            (best_human.qa_pair_id,)
        )

    if not pair_row:
        logger.info(
            f"No paired assistant answer found for human memory {best_human.id} with qa_pair_id {best_human.qa_pair_id}"
        )
        return items

    # 已在候选列表中：直接提升分数
    for item in items:
        if item.id == pair_row["id"]:
            item.score = max(item.score, best_human.score + 0.2)
            item.primary_sector = item.primary_sector or query_sector
            item_content_preview = item.content if len(item.content) <= 20 else item.content[:20] + "..."
            logger.info(
                f"Promoted paired assistant memory[{item.id}] content: {item_content_preview} with score {item.score} based on human memory[{best_human.id}]"
            )
            return items

    # 不在候选列表中：动态追加
    pair_id = pair_row.get("id")
    items.append(IMemoryItemInfo(
        id=pair_id,
        content=pair_row.get("content"),
        score=best_human.score + 0.2,
        primary_sector=pair_row.get("primary_sector") or query_sector,
        path=[str(pair_id)] if pair_id else [],
        salience=pair_row.get("salience") or 0.0,
        last_seen_at=pair_row.get("last_seen_at"),
        tags=json.loads(pair_row.get("tags") or "[]"),
        qa_role=pair_row.get("qa_role"),
        qa_pair_id=pair_row.get("qa_pair_id"),
        metadata=json.loads(pair_row.get("meta") or "{}")
    ))
    return items


def _resolve_auto_qa_linking(user: IMemoryUser, qa_role: QARole | None) -> str | None:
    """
    自动补齐 QA 配对字段：
    - human：生成新的 qa_pair_id
    - assistant：复用最近一条“未被 assistant 配对”的 human.qa_pair_id
    """
    if qa_role not in ("human", "assistant"):
        return None

    if qa_role == "human":
        return str(uuid.uuid4())

    # assistant 自动配对：匹配同身份下最近未配对的 human
    sql = """
          SELECT h.qa_pair_id
          FROM memories h
          WHERE h.qa_role = 'human'
            AND h.user_id = %s
            AND h.qa_pair_id IS NOT NULL
            AND NOT EXISTS (SELECT 1
                            FROM memories a
                            WHERE a.qa_role = 'assistant'
                              AND a.qa_pair_id = h.qa_pair_id
                              AND a.user_id = h.user_id)
          ORDER BY h.created_at DESC LIMIT 1
          """
    row = db.fetchone(sql, (user.id,))
    if not row:
        return str(uuid.uuid4())

    return row.get("qa_pair_id")
