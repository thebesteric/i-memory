import json
import math
import re
import time
import uuid
from typing import Optional, Any, Dict, List

from src.ai.embed.base_embed_model import get_embed_model, BaseEmbedModel
from src.core.config import env
from src.core.db import db
from src.core.dml_ops import dml_ops
from src.core.sector_classify import SECTOR_CONFIGS, SectorClassifier
from src.core.vector.base_vector_store import vector_store
from src.memory.embed import embed_multi_sector, calc_mean_vec
from src.memory.user_summary import update_user_summary
from src.tools.chunking import chunk_text
from src.tools.text import canonical_token_set
from src.tools.vectors import vec_to_buf, buf_to_vec, cos_sim
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


def compute_simhash(text: str) -> str:
    """
    为一段文本生成 simhash 指纹，用于内容去重和相似性检测。
    """

    # 标准化文本，得到 token 集合
    tokens = canonical_token_set(text)
    # 存储每个 token 的哈希值
    hashes = []

    # 对每个 token 计算哈希（两次循环，确保分布均匀）
    for t in tokens:
        h = 0
        for c in t:
            h = (h << 5) - h + ord(c)
            h = h & 0xffffffff
        h = 0
        for c in t:
            val = (h << 5) - h + ord(c)
            val = val & 0xffffffff
            if val & 0x80000000: val = -((val ^ 0xffffffff) + 1)
            h = val
        hashes.append(h)

    # 64 维向量
    vec = [0] * 64

    # 64 维向量累加：每个 token 的哈希在每一位上为 1 则加 1，否则减 1
    for h in hashes:
        for i in range(64):
            bit = 1 << (i % 32)
            if h & bit:
                vec[i] += 1
            else:
                vec[i] -= 1

    # 生成 simhash
    res_hash = ""

    # 每 4 位合成一个十六进制数，拼成 16 位字符串
    for i in range(0, 64, 4):
        nibble = 0
        if vec[i] > 0: nibble += 8
        if vec[i + 1] > 0: nibble += 4
        if vec[i + 2] > 0: nibble += 2
        if vec[i + 3] > 0: nibble += 1
        res_hash += format(nibble, 'x')

    # 返回最终的 simhash 字符串
    return res_hash


def compute_hamming_distance(h1: str, h2: str) -> int:
    """
    计算两个 simhash 指纹之间的汉明距离
    :param h1: 第一个 simhash 字符串
    :param h2: 第二个 simhash 字符串
    :return: 汉明距离（整数）
    """
    dist = 0
    for i in range(len(h1)):
        x = int(h1[i], 16) ^ int(h2[i], 16)
        if x & 8: dist += 1
        if x & 4: dist += 1
        if x & 2: dist += 1
        if x & 1: dist += 1
    return dist


def extract_essence(raw: str, sec: str, max_len: int) -> str:
    """
    从一段文本中提取精华内容，优先保留重要句子和信息密集的部分，以适应记忆存储的长度限制。
    @param raw: 原始文本内容
    @param sec: 记忆扇区（如语义、情感、程序、事件、反思等），可用于调整提取策略
    @param max_len: 提取内容的最大长度限制
    """

    # 如果环境变量 env.USE_SUMMARY_ONLY 为假，或原文长度不超过 max_len，直接返回原文，不做摘要
    if not env.USE_SUMMARY_ONLY or len(raw) <= max_len:
        return raw

    # 用正则将原文按句号、问号、感叹号等分割成句子，并去除过短的句子（小于10个字符）
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if len(s.strip()) > 10]

    # 如果没有有效句子，返回原文的前 max_len 个字符
    if not sents: return raw[:max_len]

    scored = []
    # 遍历每个句子，按多种启发式规则累加分数
    for idx, s in enumerate(sents):
        sc = 0
        # 首句（加10分）
        if idx == 0: sc += 10
        # 次句（加5分）
        if idx == 1: sc += 5
        # if re.match(r"^#+\s", s) or re.match(r"^[A-Z][A-Z\s]+:", s): sc += 8
        # 是否为“人名:内容”格式（加6分）
        if re.match(r"^[A-Z][a-z]+:", s): sc += 6
        # 包含 yyyy-MM-dd 格式（加7分）
        if re.search(r"\d{4}-\d{2}-\d{2}", s): sc += 7
        # 包含月份（加5分）
        if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+", s, re.I): sc += 5
        # 包含日期、金额、年、月、公里等（加4分）
        if re.search(r"\$\d+|\d+\s*(miles|dollars|years|months|km)", s): sc += 4
        # 包含人名、地名、机构名等信息密集的短语（加3分）
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", s): sc += 3
        # 包含动词（加4分）
        if re.search(
                r"\b(bought|purchased|serviced|visited|went|got|received|paid|earned|learned|discovered|found|saw|met|completed|finished|fixed|implemented|created|updated|added|removed|resolved)\b",
                s, re.I): sc += 4
        # 包含疑问词（加2分）
        if re.search(r"\b(who|what|when|where|why|how)\b", s, re.I): sc += 2
        # 短句（少于80字符，加2分）
        if len(s) < 80: sc += 2
        # 包含第一人称（加1分）
        if re.search(r"\b(I|my|me)\b", s): sc += 1
        scored.append({"text": s, "score": sc, "idx": idx})

    # 按得分从高到低排序
    scored.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    curr_len = 0
    # 优先选首句（如果长度合适），再依次选高分句，直到总长度接近 max_len
    first = next((x for x in scored if x["idx"] == 0), None)
    if first and len(first["text"]) < max_len:
        selected.append(first)
        curr_len += len(first["text"])

    # 选其他句子
    for item in scored:
        if item["idx"] == 0: continue
        if curr_len + len(item["text"]) + 2 <= max_len:
            selected.append(item)
            curr_len += len(item["text"]) + 2

    # 按原文顺序排序选中的句子
    selected.sort(key=lambda x: x["idx"])
    # 按原文顺序拼接选中的句子，作为最终摘要返回
    return " ".join(x["text"] for x in selected)


async def create_single_waypoint(new_id: str, new_mean: List[float], ts: int, user_id: str = "anonymous"):
    """
    用于为新记忆（new_id）在所有记忆中寻找最相似的“均值向量”，并在数据库中建立 waypoint（路标）关联
    该函数会遍历当前用户的所有记忆，计算每个记忆的均值向量与新记忆均值向量的余弦相似度，
    找到相似度最高的记忆，并在数据库中插入一条 waypoint 记录，表示新记忆指向该最相似记忆。
    如果没有找到任何相似记忆，则创建一条自指向的 waypoint 记录。
    该函数最终会提交数据库事务以保存更改。

    @param new_id: 新记忆的唯一标识符
    @param new_mean: 新记忆的均值向量（浮点数列表）
    @param ts: 当前时间戳（毫秒）
    @param user_id: 用户标识符（可选，默认为 "anonymous"）
    """
    # 获取当前用户的所有记忆
    memories = dml_ops.all_mem_by_user(user_id, 1000, 0) if user_id else dml_ops.all_mem(1000, 0)
    best = None
    best_sim = -1.0

    import numpy as np
    # 将新记忆的均值向量转换为 numpy 数组
    nm = np.array(new_mean, dtype=np.float32)

    # 遍历所有记忆，计算与新记忆均值向量的余弦相似度
    for mem in memories:
        # 跳过自身或没有均值向量的记忆
        if mem["id"] == new_id or not mem["mean_vec"]: continue
        # 将现有记忆的均值向量转换为 numpy 数组
        ex_mean = np.array(buf_to_vec(mem["mean_vec"]), dtype=np.float32)
        # 计算余弦相似度
        sim = cos_sim(nm, ex_mean)
        # 如果相似度超过当前最佳相似度，更新最佳记忆 ID 和相似度
        if sim > best_sim:
            best_sim = sim
            best = mem["id"]

    # 如果找到最佳匹配，创建 waypoint 关联
    # Use Postgres UPSERT syntax (ON CONFLICT) — waypoints has PRIMARY KEY (src_id, dst_id)
    insert_sql = (
        """
        INSERT INTO waypoints(src_id, dst_id, user_id, weight, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (src_id,dst_id) DO
        UPDATE SET
            user_id = EXCLUDED.user_id, weight = EXCLUDED.weight, created_at = EXCLUDED.created_at, updated_at = EXCLUDED.updated_at
        """
    )
    if best:
        db.execute(insert_sql, (new_id, best, user_id, float(best_sim), ts, ts))
    # 否则创建自指向的 waypoint
    else:
        db.execute(insert_sql, (new_id, new_id, user_id, 1.0, ts, ts))
    db.commit()


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


async def add_hsg_memory(content: str, tags: Optional[str] = None, metadata: Any = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    添加一条 Hierarchical Semantic Graph 记忆（数据库 + 向量存储、按扇区（sectors）分层组织记忆）
    :param content: 记忆内容
    :param tags: 标签
    :param metadata: 元数据
    :param user_id: 用户标识符
    :return:
    """
    # 获取嵌入模型
    embed_model: BaseEmbedModel = get_embed_model()
    # 生成内容的嵌入向量
    vec = await embed_model.embed(content)

    # 找到该用户最相似的已有记忆
    user_memories = db.fetchall("""
                                SELECT *
                                FROM memories t
                                         LEFT JOIN vectors v on t.id = v.id
                                WHERE t.user_id = %s
                                  AND v.v IS NOT NULL
                                ORDER BY t.salience DESC, t.last_seen_at DESC LIMIT 100
                                """,
                                (user_id,))
    # 初始化最佳相似记忆（相似度，记忆记录）
    best_sim_mem_similarity = tuple()
    if user_memories:
        for user_memory in user_memories:
            v_list_json = json.loads(user_memory["v"])
            v = [float(s) for s in v_list_json]
            similarity = embed_model.similarity(vec, v)
            # 找到最小的汉明距离
            if not best_sim_mem_similarity or similarity > best_sim_mem_similarity[0]:
                best_sim_mem_similarity = (similarity, user_memory)

    if best_sim_mem_similarity:
        content = best_sim_mem_similarity[1]["content"]
        content = content if len(content) <= 20 else content[:20] + "..."
        logger.debug(f"[HSG] Best similar memory: Sim: {best_sim_mem_similarity[0]}, Content: {content}")

    # 当前时间戳（毫秒）
    now = int(time.time() * 1000)

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
        user = db.fetchone("SELECT * FROM users WHERE user_id=%s", (user_id,))
        # 没有该用户则创建一条新用户记录
        if not user:
            db.execute(
                "INSERT INTO users(user_id,summary,reflection_count,created_at,updated_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (user_id) DO NOTHING",
                (user_id, "User profile initializing...", 0, now, now)
            )
            db.commit()

    # 对内容分段，判断是否需要分块存储
    chunks = chunk_text(content)
    use_chunks = len(chunks) > 1

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

        # 调用 extract_essence，根据 sector 和配置，生成摘要
        stored = extract_essence(content, cls_ret.primary, env.SUMMARY_MAX_LENGTH)
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
            user_id=user_id or "anonymous",
            segment=cur_seg,
            content=stored,
            primary_sector=cls_ret.primary,
            tags=tags,
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
            await vector_store.store_vector(mid, r["sector"], r["vector"], r["dim"], user_id or "anonymous")

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
        await create_single_waypoint(mid, mean_vec, now, user_id)
        if user_id:
            # 更新用户摘要
            await update_user_summary(user_id)

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
