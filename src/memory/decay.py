import asyncio
import datetime
import math
import random
import time
from typing import Optional, Any

from agile.utils import LogHelper, singleton, timing

from src.core.components import get_vector_store
from src.core.config import env
from src.core.constants import HYBRID_PARAMS
from src.core.db import get_db, DB
from src.core.mem_ops import mem_ops
from src.core.sector_classify import SECTOR_CONFIGS, SectorCfg
from src.core.vector.base_vector_store import BaseVectorStore
from src.tools.text import canonical_tokens_from_text

logger = LogHelper.get_logger(title="[DECAY]")


@singleton
class DecayCfg:
    def __init__(self, *, reinforce_on_query: bool = True, regeneration_enabled: bool = True):
        self.threads = env.DECAY_THREADS or 3
        self.cold_threshold = env.DECAY_COLD_THRESHOLD or 0.25
        self.reinforce_on_query = reinforce_on_query
        self.regeneration_enabled = regeneration_enabled
        self.max_vec_dim = env.VECTOR_MAX_DIM or 1536
        self.min_vec_dim = env.VECTOR_MIN_DIM or 64
        self.summary_layers = min(3, max(1, env.SUMMARY_LAYERS or 3))
        self.lambda_hot = 0.005
        self.lambda_warm = 0.02
        self.lambda_cold = 0.05
        # 一天的毫秒数
        self.time_unit_ms = 86_400_000


class Decay:
    active_q = 0
    last_decay = 0

    # 内部防抖冷却时间（毫秒）；实例初始化时会根据 DECAY_INTERVAL_SECONDS 自动覆盖
    cooldown = 60000

    @staticmethod
    def _derive_cooldown_ms(interval_seconds: int) -> int:
        # 比调度间隔略小：默认少 10%，最少少 2 秒，最多少 30 秒
        delta_sec = max(2, min(30, int(interval_seconds * 0.1)))
        cooldown_sec = max(1, interval_seconds - delta_sec)
        return cooldown_sec * 1000

    def __init__(self, reinforce_on_query: bool = True, regeneration_enabled: bool = True):
        """
        衰减管理器
        记录当前活跃的查询数，用于控制衰减操作的频率
        :param reinforce_on_query: 是否在查询命中时增强记忆显著性
        :param regeneration_enabled: 是否启用记忆向量的再生成机制
        """
        self.cfg = DecayCfg(
            reinforce_on_query=reinforce_on_query,
            regeneration_enabled=regeneration_enabled
        )
        interval_sec = max(1, int(getattr(env, "DECAY_INTERVAL_SECONDS", 60) or 60))
        self.cooldown = self._derive_cooldown_ms(interval_sec)
        self.vector_store: BaseVectorStore = get_vector_store()
        self.db: DB = get_db()

    @classmethod
    def inc_q(cls):
        cls.active_q += 1

    @classmethod
    def dec_q(cls):
        cls.active_q = max(0, cls.active_q - 1)

    @classmethod
    def pick_tier(cls, m: dict, now_ts: int) -> str:
        """
        记忆分层，把记忆分成 hot/warm/cold，决定衰减强度
        :param m: 记忆数据
        :param now_ts: 当前时间戳
        :return:
        """
        # 距上次访问的时间（按天归一）
        last_touch = cls.to_ms(m.get("last_seen_at") or m.get("updated_at"), now_ts)
        dt = max(0, now_ts - last_touch)
        recent = dt < 6 * 86_400_000

        high = (m.get("coactivations") or 0) > 5 or (m["salience"] or 0) > 0.7
        # 近期且高激活/高显著性 -> hot
        if recent and high:
            return "hot"
        # 近期或中显著性 -> warm
        if recent or (m["salience"] or 0) > 0.4:
            return "warm"
        # 否则 cold
        return "cold"

    @staticmethod
    def to_ms(v: Any, default_ms: int) -> int:
        """Normalize DB time values to milliseconds for safe arithmetic."""
        if v is None:
            return default_ms

        if isinstance(v, datetime.datetime):
            return int(v.timestamp() * 1000)

        if isinstance(v, (int, float)):
            # Heuristic: values below 1e11 are likely in seconds.
            return int(v * 1000) if v < 100_000_000_000 else int(v)

        if isinstance(v, str):
            txt = v.strip()
            if not txt:
                return default_ms
            try:
                parsed = datetime.datetime.fromisoformat(txt.replace("Z", "+00:00"))
                return int(parsed.timestamp() * 1000)
            except ValueError:
                try:
                    num = float(txt)
                    return int(num * 1000) if num < 100_000_000_000 else int(num)
                except ValueError:
                    return default_ms

        return default_ms

    @classmethod
    def mean(cls, arr: list[float]) -> float:
        return sum(arr) / len(arr) if arr else 0

    @classmethod
    def normalize(cls, v: list[float]):
        n = math.sqrt(sum(x * x for x in v))
        if n > 0:
            for i in range(len(v)): v[i] /= n

    @classmethod
    def compress_vector(cls, vec: list[float], f: float, min_dim=64, max_dim=1536) -> list[float]:
        src = vec if vec else [1.0]
        tgt_dim = max(min_dim, min(max_dim, math.floor(len(src) * max(0.0, min(1.0, f)))))
        dim = max(min_dim, min(len(src), tgt_dim))

        if dim >= len(src): return list(src)

        pooled = []
        bucket = math.ceil(len(src) / dim)
        for i in range(0, len(src), bucket):
            sub = src[i: i + bucket]
            pooled.append(cls.mean(sub))

        cls.normalize(pooled)
        return pooled

    @classmethod
    def hash_to_vec(cls, s: str, d=32) -> list[float]:
        h = 2166136261
        for c in s:
            h ^= ord(c)
            h = (h * 16777619) & 0xffffffff

        out = [0.0] * max(2, d)
        x = h or 1
        for i in range(len(out)):
            x ^= (x << 13) & 0xffffffff
            x ^= (x >> 17) & 0xffffffff
            x ^= (x << 5) & 0xffffffff
            out[i] = ((x / 0xffffffff) * 2 - 1)

        cls.normalize(out)
        return out

    @classmethod
    def top_keywords(cls, t: str, k=5) -> list[str]:
        words = canonical_tokens_from_text(t)
        freq = {}
        for w in words: freq[w] = freq.get(w, 0) + 1
        items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in items[:k]]

    @classmethod
    def fingerprint_mem(cls, m: dict) -> dict[str, Any]:
        """
        生成指纹向量和关键词摘要
        :param m:
        :return:
        """
        base = f"{m['id']}|{m.get('summary') or m['content'] or ''}".strip()
        vec = cls.hash_to_vec(base, 32)
        summary = " ".join(cls.top_keywords(m.get('summary') or m['content'] or "", 3))
        return {"vector": vec, "summary": summary}

    @staticmethod
    def calc_decay(sec: str, init_sal: float, days_since: float, seg_idx: Optional[int] = None, max_seg: Optional[int] = None) -> float:
        """
        计算记忆衰减后的显著性（艾宾浩斯遗忘曲线）
        公式：decayed_sal = init_sal * exp(-lambda * days_since) + alpha_reinforce * (1 - exp(-lambda * days_since))
        其中，lambda 根据记忆所属的 sector 和分段位置进行调整
        :param sec: 记忆所属的扇区
        :param init_sal: 初始显著性值（0-1 之间,表示记忆的重要程度）
        :param days_since: 距离上次访问该记忆的天数
        :param seg_idx: 当前记忆在分段中的索引位置（可选）
        :param max_seg: 总分段数（可选）
        :return: 衰减后的显著性
        """
        # 获取扇区配置
        sector_cfg: SectorCfg = SECTOR_CONFIGS.get(sec)

        # 如果没有配置则返回初始显著性（不衰减）
        if not sector_cfg:
            return init_sal

        # 获取基础衰减率
        lam = sector_cfg.decay_lambda
        # 根据分段位置调整衰减率，如果记忆在靠后的分段，衰减会变慢（越往后的记忆越重要,衰减越慢）
        if seg_idx is not None and max_seg is not None and max_seg > 0:
            # 让重要位置的记忆衰减更慢
            seg_ratio = math.sqrt(seg_idx / max_seg)
            lam = lam * (1.0 - seg_ratio)

        # 计算衰减部分（艾宾浩斯遗忘曲线，负责自然遗忘）：使用指数衰减公式，时间越久记忆越弱（但永远不会降到 0）
        decayed = init_sal * math.exp(-lam * days_since)
        # 计算增强部分（保证不完全遗忘）：添加一个“基础记忆强度”，防止完全遗忘：时间越长，增强项越大，最终稳定在 alpha_reinforce
        reinforce = HYBRID_PARAMS["alpha_reinforce"] * (1 - math.exp(-lam * days_since))

        # 返回最终显著性，确保在 0 到 1 之间
        return max(0.0, min(1.0, decayed + reinforce))

    @staticmethod
    def calc_recency_score_decay(last_seen: datetime.datetime) -> float:
        """
        计算最近的分数：距离最近一次访问的时间越久（hours 越大），得分越低；时间越近（hours 越小），得分越高
        :param last_seen: 最近一次访问时间
        :return:
        """
        if not last_seen:
            return 0.0
        now = datetime.datetime.now()
        time_diff = now - last_seen
        dt = max(0.0, time_diff.total_seconds())
        hours = dt / 3600.0
        # 单调递减指数函数，时间越久分数越低
        return math.exp(-0.05 * hours)

    async def on_query_hit(self, mem_id: str, sector: str, reembed_fn=None):
        """
        记录查询命中事件并触发动态更新
        @param mem_id: 记忆 ID
        @param sector: 命中记忆的主扇区（语义/情感/程序/事件/反思等）
        @param reembed_fn: 嵌入函数，用于将查询文本转换为向量
        """
        if not self.cfg.regeneration_enabled and not self.cfg.reinforce_on_query:
            return

        m = mem_ops.get_mem(mem_id)
        if not m: return

        updated = False
        # 是否允许重新生成嵌入向量 并且 提供了 reembed_fn 函数
        if self.cfg.regeneration_enabled and reembed_fn:
            # 获取当前记忆以及指定 sector 的向量
            vec_row = await self.vector_store.get_vector(mem_id, sector)
            # 向量存在且维度较小（<=64），则重新生成嵌入向量
            if vec_row and vec_row.vector and len(vec_row.vector) <= 64:
                try:
                    base = m["summary"] or m["content"] or ""
                    new_vec = await reembed_fn(base)
                    await self.vector_store.store_vector(mem_id, sector, new_vec, len(new_vec))
                    updated = True
                except Exception as ex:
                    logger.warning("Error re-embedding memory on query hit: %s", ex)

        # 是否允许在查询命中时增强记忆显著性
        if self.cfg.reinforce_on_query:
            # 计算新的显著性，增加 0.15，最大不超过 1.0
            new_sal = min(1.0, (m["salience"] or 0.5) + 0.15)
            self.db.execute("UPDATE memories SET salience = %s, last_seen_at = %s WHERE id = %s", (new_sal, datetime.datetime.now(), mem_id))
            self.db.commit()
            updated = True

        if updated:
            logger.info("Memory %s reinforced on query hit", mem_id)

    @timing
    async def apply_decay(self):
        """
        批处理衰减器，按 segment 抽样一部分记忆，计算“遗忘程度”，并在必要时做向量压缩或指纹化降级
        核心：分段抽样 + 渐进衰减 + 分级降级
        :return:
        """
        # 有活跃查询（“读优先”策略，保障在线响应），则跳过
        if self.active_q > 0:
            logger.info(f"Skipped - {self.active_q} active queries")
            return

        # 当前衰减时间小于 cooldown 定义的时间（避免短时间重复跑批），则跳过
        now_ts = int(time.time() * 1000)
        if now_ts - self.last_decay < self.cooldown:
            rem = (self.cooldown - (now_ts - self.last_decay)) / 1000
            logger.info(f"Skipped - cooldown active ({rem:.0f}s left)")
            return

        # 记录本次衰减时间戳
        self.last_decay = now_ts

        t0 = time.time()

        # 读取 segment 列表，按照 segment 降序（越新的 segment 越先处理），每次处理一个 segment 的一部分记忆
        segments_rows = self.db.fetchall("SELECT DISTINCT segment FROM memories ORDER BY segment DESC")
        segments = [r["segment"] for r in segments_rows]

        # 处理规模
        tot_proc = 0
        # 修改数
        tot_chg = 0
        # 压缩次数
        tot_comp = 0
        # 指纹化次数
        tot_fp = 0
        # tier 分布
        tier_counts = {"hot": 0, "warm": 0, "cold": 0}

        for seg in segments:
            # feedback_score as coactivations 把反馈分当作激活强度；后续公式会用它减缓衰减
            rows = self.db.fetchall(
                """
                SELECT id,
                       content,
                       summary,
                       salience,
                       decay_lambda,
                       last_seen_at,
                       updated_at,
                       primary_sector,
                       feedback_score as coactivations
                FROM memories
                WHERE segment = %s
                """,
                (seg,))

            # 衰减比例
            decay_ratio = env.DECAY_RATIO or 0.03
            # 批次大小，最少 1 条，控制每个 segment 每轮只处理一部分记忆，形成“渐进式衰减”
            batch_sz = max(1, int(len(rows) * decay_ratio))
            if not rows:
                continue

            if batch_sz >= len(rows):
                batch = rows
            else:
                # True random sample avoids contiguous-window bias.
                batch = random.sample(rows, batch_sz)

            for m in batch:
                # 将记忆转换为字段
                dict_m = dict(m)
                # 单条记忆分层，返回记忆热度：hot/warm/cold
                m_tier = self.pick_tier(dict_m, now_ts)
                tier_counts[m_tier] += 1

                # 根据记忆热度，获取衰减系数，越热的记忆，衰减越小
                lam = self.cfg.lambda_hot if m_tier == "hot" else (self.cfg.lambda_warm if m_tier == "warm" else self.cfg.lambda_cold)

                # 距上次访问的时间（按天归一）
                last_touch = self.to_ms(dict_m.get("last_seen_at") or dict_m.get("updated_at"), now_ts)
                dt = max(0.0, (now_ts - last_touch) / self.cfg.time_unit_ms)
                # 激活强度
                act = max(0, dict_m.get("coactivations") or dict_m.get("feedback_score") or 0)
                # 显著性增强值：act 增大时，sal 增大，从而减慢衰减
                sal = max(0.0, min(1.0, (dict_m["salience"] or 0.5) * (1 + math.log1p(act))))

                # 公式：时间越久、tier 越冷（lam 越大）则衰减越强；显著性高则衰减更慢
                f = math.exp(-lam * (dt / (sal + 0.1)))
                # 计算最新的显著性值
                new_sal = max(0.0, min(1.0, sal * f))
                # 差值 > 0.001 才算变化，避免无意义写库
                changed = abs(new_sal - (dict_m["salience"] or 0)) > 0.001

                # 分级降级策略（随遗忘加深）
                # 当遗忘较明显时，把向量降维（compress_vector），减少存储与检索开销
                # 关键点：这是“信息保留 + 成本下降”的中间态，不直接删除语义
                if f < 0.7:
                    sector = dict_m["primary_sector"] or "semantic"
                    # 取当前 sector 向量
                    vec_row = await self.vector_store.get_vector(dict_m["id"], sector)
                    if vec_row and vec_row.vector:
                        vec = vec_row.vector
                        if len(vec) > 0:
                            # 计算新向量
                            new_vec = self.compress_vector(vec, f, self.cfg.min_vec_dim, self.cfg.max_vec_dim)
                            # 仅当维度变小才回写
                            if len(new_vec) < len(vec):
                                # 更新向量表
                                await self.vector_store.store_vector(dict_m["id"], sector, new_vec, len(new_vec))
                                # 压缩次数 +1
                                tot_comp += 1
                                changed = True

                # 更冷的记忆降级为“指纹向量 + 关键词摘要”
                # 关键点：这是强降级，召回精度会下降，用关键词重写 summary，保留最小可检索性
                if f < max(0.3, self.cfg.cold_threshold):
                    sector = dict_m["primary_sector"] or "semantic"
                    # 生成 32 维 hash 向量和短摘要
                    fp = self.fingerprint_mem(dict_m)
                    # 更新向量表
                    await self.vector_store.store_vector(dict_m["id"], sector, fp["vector"], len(fp["vector"]))
                    # 更新记忆摘要
                    self.db.execute(
                        "UPDATE memories SET summary = %s WHERE id = %s",
                        (fp["summary"], dict_m["id"])
                    )
                    # 指纹化次数 +1
                    tot_fp += 1
                    changed = True

                # 更新数据库
                if changed:
                    self.db.execute(
                        "UPDATE memories SET salience = %s, updated_at = %s WHERE id = %s",
                        (new_sal, datetime.datetime.now(), dict_m["id"])
                    )
                    tot_chg += 1

                # 处理规模 +1
                tot_proc += 1
                # sleep(0) 能提升协作公平性，避免长循环饿死其他协程
                await asyncio.sleep(0)

        self.db.commit()

        # 计算耗时
        dur = (time.time() - t0) * 1000
        logger.info(f"{tot_chg}/{tot_proc} | tiers: {tier_counts} | comp_count: {tot_comp} fp_count: {tot_fp} | {dur:.1f}ms")
