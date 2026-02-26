import datetime
import json
import math
import time
import uuid
from typing import Optional, Any, Dict, List

from src.ai.embed.base_embed_model import BaseEmbedModel
from src.ai.model_provider import get_embed_model
from src.core.cache.memory_cache import MemoryCache
from src.core.config import env
from src.core.constants import SECTOR_RELATIONSHIPS, HYBRID_PARAMS, CACHE_TTL, CACHE_SIZE, CACHE_TTL_TIME_UNIT
from src.core.db import get_db
from src.core.dml_ops import dml_ops
from src.core.extract_essence import ExtractEssence
from src.core.score import compute_tag_match_score, compute_hybrid_score
from src.core.sector_classify import SECTOR_CONFIGS, SectorClassifier, ClassifyResult
from src.core.vector.base_vector_store import vector_store, VectorSearch
from src.core.waypoints import Waypoints, Expansion
from src.memory import user_ops
from src.memory.decay import Decay
from src.memory.embed import embed_multi_sector, calc_mean_vec, embed
from src.memory.models.memory_models import IMemoryFilters, IMemoryItemDebugInfo, IMemoryItemInfo, IMemoryUserIdentity, IMemoryUser
from src.memory.user_summary import update_user_summary
from src.ops.dynamic_memory import calculate_cross_sector_resonance_score, apply_retrieval_trace_reinforcement_to_memory, \
    propagate_associative_reinforcement_to_linked_nodes
from src.tools.chunking import chunk_text
from src.tools.keyword import compute_keyword_overlap, compute_token_overlap
from src.tools.text import canonical_token_set
from src.tools.vectors import vec_to_buf, cos_sim
from src.utils.log_helper import LogHelper
from src.utils.time_unit import TimeUnit

logger = LogHelper.get_logger()
waypoints = Waypoints()
db = get_db()
decay = Decay(reinforce_on_query=True, regeneration_enabled=True)


async def embed_query_for_all_sectors(query: str, sectors: List[str]) -> Dict[str, List[float]]:
    res = {}
    for s in sectors:
        res[s] = await embed(query, s)
    return res


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


async def add_hsg_memory(content: str, tags: List[str] = None, metadata: Any = None, user_identity: IMemoryUserIdentity = None) -> Dict[str, Any]:
    """
    添加一条 Hierarchical Semantic Graph 记忆（数据库 + 向量存储、按扇区（sectors）分层组织记忆）
    :param content: 记忆内容
    :param tags: 标签
    :param metadata: 元数据
    :param user_identity: 用户身份
    :return:
    """
    # 获取嵌入模型
    embed_model: BaseEmbedModel = get_embed_model()
    # 生成内容的嵌入向量
    vec = await embed_model.embed(content)

    # 构建 SQL 查询，查询该用户的记忆，包含租户和项目过滤（如果提供了租户和项目信息），并且只查询有向量的记忆
    sql_parts = [
        """
        SELECT *
        FROM memories t
                 LEFT JOIN vectors v on t.id = v.id
        WHERE t.user_id = %s
          AND v.v IS NOT NULL
        """,
    ]

    user_id = user_identity.user_id
    tenant_id = user_identity.tenant_id
    project_id = user_identity.project_id

    # 查询参数列表，初始包含 user_id
    params = [user_id]

    # 判断租户是否存在
    if tenant_id:
        sql_parts.append("AND t.tenant_id = %s")
        params.append(tenant_id)

    # 判断项目是否存在
    if project_id:
        sql_parts.append("AND t.project_id = %s")
        params.append(project_id)

    # 拼接排序和分页
    sql_parts.append("ORDER BY t.salience DESC, t.last_seen_at DESC LIMIT 100")
    final_sql = " ".join(sql_parts)

    # 找到该用户最相似的已有记忆
    user_memories = db.fetchall(final_sql, tuple(params))

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
        logger.debug(f"[HSG] Maybe best similar memory: Sim: {best_sim_mem_similarity[0]}, Content: {best_sim_content}")

    # 当前时间
    now = datetime.datetime.now()

    # 存在相似记忆 && 相似度 >= 0.9
    if best_sim_mem_similarity and best_sim_mem_similarity[0] >= env.SIMILARITY_THRESHOLD:
        """
        如果发现内容高度相似（相似度 >= 0.95）
        不会新建一条记忆，而是提升已有记忆的显著性，并更新时间戳，表示这条记忆再次被“关注”或“激活”
        """
        best_sim_mem = best_sim_mem_similarity[1]
        content = best_sim_mem["content"]
        content = content if len(content) <= 20 else content[:20] + "..."
        logger.info(f"[HSG] Found similar memory {best_sim_mem['id']} with {best_sim_mem_similarity[0]} for User: {user_id}, Content: {content}")
        # 提升显著性，但不超过 1.0
        boost = min(1.0, (best_sim_mem["salience"] or 0) + 0.15)
        # 更新最后访问时间和显著性
        db.execute("UPDATE memories SET last_seen_at=%s, salience=%s, updated_at=%s WHERE id=%s", (now, boost, now, best_sim_mem['id']))
        db.commit()
        return {
            "id": best_sim_mem["id"],
            "primary_sector": best_sim_mem["primary_sector"],
            "sectors": [best_sim_mem["primary_sector"]],
            "deduplicated": True
        }

    # 判断是否有用户
    if user_id:
        user: IMemoryUser = await user_ops.get_user(user_identity=user_identity)
        # 没有该用户则创建一条新用户记录
        if not user:
            await user_ops.add_user(user_identity=user_identity)

    # 对内容分段，判断是否需要分块存储
    chunks, total_token = chunk_text(content)
    use_chunks = len(chunks) > 1

    # 更新字符数和估计的令牌数
    metadata.update({
        "char_count": len(content),
        "estimated_tokens": total_token,
    })

    # 文本分类：判断内容所属的主/辅 sector（语义、情感、程序、事件、反思等）
    cls_ret = await SectorClassifier(content=content, metadata=metadata).classify()
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
            logger.info(f"[HSG] Rotated to segment [{cur_seg}]")

        # 调用 extract_essence，生成摘要
        essence = await ExtractEssence(content=content, max_len=env.SUMMARY_MAX_LENGTH).extract()
        # 获取主 sector 的配置
        sec_cfg = SECTOR_CONFIGS[cls_ret.primary]
        # 始化记忆的显著性（salience）分数
        # 基础分为 0.4，每多一个辅扇区（cls_ret.additional），显著性加 0.1
        # 最终显著性限定在 0.0~1.0 之间，防止过高或为负
        init_sal = max(0.0, min(1.0, 0.4 + 0.1 * len(cls_ret.additional)))

        # 调用 dml_ops.ins_mem，将记忆内容、摘要、sector、标签、元数据等插入数据库
        mid = str(uuid.uuid4())
        dml_ops.ins_mem(
            id=mid,
            user_id=user_id,
            tenant_id=tenant_id or None,
            project_id=project_id or None,
            segment=cur_seg,
            content=essence,
            primary_sector=cls_ret.primary,
            sectors=json.dumps(all_secs or []),
            tags=json.dumps(tags or []),
            meta=json.dumps(metadata or {}),
            created_at=now,
            updated_at=now,
            last_seen_at=now,
            salience=init_sal,
            decay_lambda=sec_cfg.decay_lambda,
            version=1,
            mean_dim=None,
            mean_vec=None,
            compressed_vec=None,
            feedback_score=0
        )

        # 调用 embed_multi_sector，对内容进行多 sector 嵌入，生成向量
        emb_res = await embed_multi_sector(mid, content, all_secs, chunks if use_chunks else None)
        for r in emb_res:
            # 存储每个 sector 的向量到向量库
            await vector_store.store_vector(mid, r["sector"], r["vector"], r["dim"], user_identity)

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
            # 更新记忆的压缩向量
            db.execute("UPDATE memories SET compressed_vec=%s WHERE id=%s", (vec_to_buf(comp), mid))

        # 建立 Waypoint 关联，为新记忆建立与其他记忆的关联
        await waypoints.create_single_waypoint(mid, mean_vec, now, user_identity)
        if user_id:
            # 更新用户摘要
            await update_user_summary(user_identity)

        # 返回新记忆的 id、内容、sector、分段数、salience 等信息
        return {
            "id": mid,
            "content": content,
            "primary_sector": cls_ret.primary,
            "sectors": all_secs,
            "chunks": len(chunks),
            "salience": init_sal
        }
    except Exception as e:
        raise e


async def calc_multi_vec_fusion_score(mid: str, qe: Dict[str, List[float]], w: Dict[str, float]) -> float:
    """
    计算多向量融合相似度评分
    通过对记忆的各扇区向量与查询向量进行余弦相似度计算，并根据权重进行加权平均，得到最终的融合相似度评分
    该评分反映了记忆在多个维度（扇区）上与查询的相关性
    :param mid: 记忆 ID
    :param qe: 查询向量字典，键为扇区名称，值为对应的向量列表
    :param w: 权重字典，键为权重名称，值为对应的权重值
    :return: 融合相似度评分，浮点数
    """

    # 获取记忆的所有向量
    vecs = await vector_store.get_vectors_by_id(mid)
    # 加权相似度总和（分子）
    s = 0.0
    # 有效权重总和（分母），用于归一化
    tot = 0.0
    # 权重映射，将权重名称映射到对应的扇区名称
    weight_mapping = {
        "semantic": w.get("semantic_dimension_weight", 0),
        "emotional": w.get("emotional_dimension_weight", 0),
        "procedural": w.get("procedural_dimension_weight", 0),
        "episodic": w.get("temporal_dimension_weight", 0),
        "reflective": w.get("reflective_dimension_weight", 0),
    }

    for v in vecs:
        # 获取记忆在该扇区的向量
        qv = qe.get(v.sector)
        if not qv:
            continue
        # 计算记忆向量与查询向量的余弦相似度
        sim = cos_sim(v.vector, qv)
        # 获取该扇区的权重，默认为 0.5
        wgt = weight_mapping.get(v.sector, 0.5)
        # 加权累加相似度
        s += sim * wgt
        # 累加权重
        tot += wgt

    # 返回归一化的融合相似度评分
    return s / tot if tot > 0 else 0.0


# 查询缓存
CACHE = MemoryCache(maxsize=CACHE_SIZE, default_ttl=CACHE_TTL, time_unit=CACHE_TTL_TIME_UNIT)


async def hsg_query(query: str, top_k: int = 10, filters: IMemoryFilters = None) -> List[IMemoryItemInfo]:
    """
    基于混合评分机制（内容相似度、关键词重叠、路标关联、时间衰减等）进行记忆检索
    @param query: 查询文本
    @param top_k: 返回的记忆数量
    @param filters: 过滤和调节参数
    @return: 返回符合查询条件的记忆列表
    """
    start_q = time.time()
    decay.inc_q()
    filters = filters or IMemoryFilters(user_identity=IMemoryUserIdentity())
    try:
        # 检查 60 秒内的查询缓存，命中则直接返回
        cache_key = f"{query}:{top_k}:{filters.model_dump_json() if filters else '{}'}"
        entry = CACHE.get(cache_key)
        if entry:
            return entry

        # 判断查询属于哪个扇区
        query_classify: ClassifyResult = await SectorClassifier(content=query).classify()
        # 提取查询关键词 token 集合
        query_tokens = canonical_token_set(query)
        # 确定检索扇区范围
        sectors = (filters.sectors if filters and filters.sectors else None) or list(SECTOR_CONFIGS.keys())
        if not sectors:
            sectors = ["semantic"]
        # 为各扇区生成嵌入向量
        query_embed = await embed_query_for_all_sectors(query, sectors)

        # 动态权重调
        primary_classify = query_classify.primary
        weight = {
            "semantic_dimension_weight": 1.2 if primary_classify == "semantic" else 0.8,
            "emotional_dimension_weight": 1.5 if primary_classify == "emotional" else 0.6,
            "procedural_dimension_weight": 1.3 if primary_classify == "procedural" else 0.7,
            "temporal_dimension_weight": 1.4 if primary_classify == "episodic" else 0.7,
            "reflective_dimension_weight": 1.1 if primary_classify == "reflective" else 0.5,
        }

        # 存储各扇区检索结果
        sector_result = {}
        for sector in sectors:
            # 获取该扇区的查询向量
            query_vector = query_embed[sector]
            # 每个扇区返回 top_k × 3 个候选
            res: List[VectorSearch] = await vector_store.search(query_vector, sector, top_k * 3, filters)
            sector_result[sector] = res

        # 收集所有候选的相似度
        all_sims = []
        # 候选记忆 ID 集合
        ids = set()
        for sector, result in sector_result.items():
            for vector_search in result:
                all_sims.append(vector_search.similarity)
                ids.add(vector_search.id)

        # 计算平均相似度，判断检索质量
        avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0
        # 自适应扩展规模，平均相似度越低，扩展规模越大，以提高召回率
        adapt_exp = math.ceil(0.3 * top_k * (1 - avg_sim))
        # 实际的的查询数量：最终检索规模
        effective_k = top_k + adapt_exp
        # 是否置信度高
        high_conf = avg_sim >= 0.55

        # 若平均相似度 < 0.55（低置信），触发图遍历扩展
        expansion: List[Expansion] = []
        if not high_conf:
            # 通过 waypoint 关系扩展
            expansion = await waypoints.expand_via_waypoints(list(ids), effective_k * 2)
            # 加入候选集
            for exp in expansion:
                ids.add(exp.id)

        # 获取候选记忆内容
        memories = dml_ops.find_mem(ids)

        # 提前计算每个候选的关键词重叠分数
        res_list: List[IMemoryItemInfo] = []
        kw_scores = {}
        for mem in memories:
            overlap = compute_keyword_overlap(query_tokens, mem["content"])
            kw_scores[mem["id"]] = overlap * 0.15

        # 数据准备与过滤
        for mem in memories:
            # 记忆 ID
            mid = mem["id"]
            # 显著性 < 过滤阈值，跳过
            if filters.min_salience and mem["salience"] < filters.min_salience:
                continue

            # 用户信息不匹配，跳过
            user_identity = filters.user_identity
            user_id = user_identity.user_id
            tenant_id = user_identity.tenant_id
            project_id = user_identity.project_id
            # 如果记忆的用户 ID 与查询用户 ID 不匹配，则跳过该记忆
            # 或者（如果提供了租户 ID）记忆的租户 ID 与查询租户 ID 不匹配，或者（如果提供了项目 ID）记忆的项目 ID 与查询项目 ID 不匹配，则跳过该记忆
            if mem["user_id"] != user_id or (tenant_id and mem["tenant_id"] != tenant_id) or (project_id and mem["project_id"] != project_id):
                continue

            # 多向量融合相似度
            mvf = await calc_multi_vec_fusion_score(mid, query_embed, weight)
            # 跨扇区共振分数
            csr = await calculate_cross_sector_resonance_score(mem["primary_sector"], query_classify.primary, mvf)

            # 取最高相似度
            best_sim = csr
            for s, rlist in sector_result.items():
                for vector_search in rlist:
                    if vector_search.id == mem and vector_search.similarity > best_sim:
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

            # 组合权重最终得分: 相似度(35%) × 扇区惩罚，关键词重叠(20%)，Waypoint权重(15%)，时效性(10%)，标签匹配(20%)
            final_score = compute_hybrid_score(sim=sim_adjust,
                                               tok_ov=token_overlap,
                                               wp_wt=waypoint_weight,
                                               rec_sc=rec_sc,
                                               kw_score=kw_scores.get(mid, 0),
                                               tag_match=tag_match_score)
            # 构建结果项
            item = IMemoryItemInfo(
                id=mid,
                content=mem["content"],
                score=final_score,
                primary_sector=mem["primary_sector"],
                path=expansion_mem.path if expansion_mem else [mid],
                salience=salience,
                last_seen_at=mem["last_seen_at"],
                tags=json.loads(mem["tags"] or "[]"),
                metadata=json.loads(mem["meta"] or {})
            )
            # 调试信息
            if filters and filters.debug:
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

        # 按分数降序
        res_list.sort(key=lambda x: x.score, reverse=True)
        # 取 effective_k 条记录
        effective_k_list: List[IMemoryItemInfo] = res_list[:effective_k]

        # 命中记忆强化
        for _item in effective_k_list:
            # 应用检索迹强化：提升成功检索到的记忆节点的显著性
            reinforcement_sal = await apply_retrieval_trace_reinforcement_to_memory(_item.id, _item.salience)
            now = datetime.datetime.now()
            db.execute("UPDATE memories SET salience=%s, last_seen_at = %s WHERE id = %s", (reinforcement_sal, now, _item.id))

            # 有图遍历路径时
            if len(_item.path) > 1:
                wps_rows = db.fetchall("SELECT dst_id, weight FROM waypoints WHERE src_id=%s", (_item.id,))
                wps = [{"target_id": row["dst_id"], "weight": row["weight"]} for row in wps_rows]

                # 向关联节点传播（得到关联节点新的显著性）
                pru = await propagate_associative_reinforcement_to_linked_nodes(_item.id, reinforcement_sal, wps)
                for u in pru:
                    # 获取关联记忆
                    linked_mem = dml_ops.get_mem(u["node_id"])
                    if linked_mem:
                        # “当前时间”与“关联记忆最后访问时间”的间隔天数
                        time_diff = (now - linked_mem["last_seen_at"]) / 86400.0
                        # 自然指数函数 math.exp 生成一个衰减系数 decay_fact（衰减因子），核心作用是将「关联记忆最后访问时间与当前时间的间隔天数」转化为 0~1 之间的权重值
                        # 时间间隔越久，衰减因子越小，对应记忆的权重 / 影响力越低
                        decay_fact = math.exp(-0.02 * time_diff)
                        # 上下文增强系数：基于记忆显著性差异和时间衰减的得分调整项，用于精细化修正基础匹配得分，放大优质记忆的优势、降低低质记忆的权重
                        ctx_boost = HYBRID_PARAMS["gamma"] * (reinforcement_sal - (linked_mem["salience"] or 0)) * decay_fact
                        # 更新关联记忆的显著性
                        new_sal = max(0.0, min(1.0, (linked_mem["salience"] or 0) + ctx_boost))
                        # 更新关联记忆的显著性和最后访问时间
                        db.execute("UPDATE memories SET salience = %s, last_seen_at = %s WHERE id = %s", (new_sal, now, u["node_id"]))

            # 记录查询命中事件并触发动态更新
            await decay.on_query_hit(_item.id, _item.primary_sector, lambda t: embed(t, _item.primary_sector))

        # 存入查询缓存
        CACHE.set(cache_key, effective_k_list)
        logger.info(
            f"[HSG] Query processed in {time.time() - start_q:.3f} seconds. Query: '{query}' Expected: {effective_k}, Actual: {len(effective_k_list)} returned.")
        return effective_k_list
    finally:
        decay.dec_q()
