import asyncio
import datetime
import math
import random
import time
from typing import Optional, Any

from agile.utils import LogHelper, singleton

from src.core.config import env
from src.core.constants import HYBRID_PARAMS
from src.core.db import get_db, DB
from src.core.dml_ops import dml_ops
from src.core.sector_classify import SECTOR_CONFIGS, SectorCfg
from src.core.vector.base_vector_store import get_vector_store, BaseVectorStore
from src.tools.text import canonical_tokens_from_text

logger = LogHelper.get_logger()


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
    COOLDOWN = 60000

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
        dt = max(0, now_ts - (m["last_seen_at"] or m["updated_at"] or now_ts))
        recent = dt < 6 * 86_400_000
        high = (m.get("coactivations") or 0) > 5 or (m["salience"] or 0) > 0.7
        if recent and high: return "hot"
        if recent or (m["salience"] or 0) > 0.4: return "warm"
        return "cold"

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

        m = dml_ops.get_mem(mem_id)
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
            logger.info("[decay] Memory %s reinforced on query hit", mem_id)

    async def apply_decay(self):
        if self.active_q > 0:
            logger.info(f"[decay] skipped - {self.active_q} active queries")
            return

        now_ts = int(time.time() * 1000)
        if now_ts - self.last_decay < self.COOLDOWN:
            rem = (self.COOLDOWN - (now_ts - self.last_decay)) / 1000
            print(f"[decay] skipped - cooldown active ({rem:.0f}s left)")
            return

        last_decay = now_ts
        t0 = time.time()
        segments_rows = self.db.fetchall("SELECT DISTINCT segment FROM memories ORDER BY segment DESC")
        segments = [r["segment"] for r in segments_rows]

        tot_proc = 0
        tot_chg = 0
        tot_comp = 0
        tot_fp = 0
        tier_counts = {"hot": 0, "warm": 0, "cold": 0}

        for seg in segments:
            rows = self.db.fetchall(
                "SELECT id, content, summary, salience, decay_lambda, last_seen_at, updated_at, primary_sector, feedback_score as coactivations FROM memories WHERE segment = ?",
                (seg,))

            decay_ratio = env.decay_ratio or 0.03
            batch_sz = max(1, int(len(rows) * decay_ratio))
            if not rows: continue

            start_idx = random.randint(0, max(0, len(rows) - batch_sz))
            batch = rows[start_idx: start_idx + batch_sz]

            for m in batch:
                dict_m = dict(m)
                m_tier = self.pick_tier(dict_m, now_ts)
                tier_counts[m_tier] += 1

                lam = self.cfg.lambda_hot if m_tier == "hot" else (self.cfg.lambda_warm if m_tier == "warm" else self.cfg.lambda_cold)
                dt = max(0, (now_ts - (dict_m["last_seen_at"] or dict_m["updated_at"] or 0)) / self.cfg.time_unit_ms)
                act = max(0, dict_m.get("coactivations") or dict_m.get("feedback_score") or 0)
                sal = max(0.0, min(1.0, (dict_m["salience"] or 0.5) * (1 + math.log1p(act))))

                f = math.exp(-lam * (dt / (sal + 0.1)))
                new_sal = max(0.0, min(1.0, sal * f))
                changed = abs(new_sal - (dict_m["salience"] or 0)) > 0.001
                if f < 0.7:
                    sector = dict_m["primary_sector"] or "semantic"
                    vec_row = await self.vector_store.get_vector(dict_m["id"], sector)
                    if vec_row and vec_row.vector:
                        vec = vec_row.vector
                        if len(vec) > 0:
                            new_vec = self.compress_vector(vec, f, self.cfg.min_vec_dim, self.cfg.max_vec_dim)

                            if len(new_vec) < len(vec):
                                await self.vector_store.store_vector(dict_m["id"], sector, new_vec, len(new_vec))
                                tot_comp += 1
                                changed = True
                if f < max(0.3, self.cfg.cold_threshold):
                    sector = dict_m["primary_sector"] or "semantic"
                    fp = self.fingerprint_mem(dict_m)
                    await self.vector_store.store_vector(dict_m["id"], sector, fp["vector"], len(fp["vector"]))
                    self.db.execute("UPDATE memories SET summary = ? WHERE id = ?", (fp["summary"], dict_m["id"]))
                    tot_fp += 1
                    changed = True

                if changed:
                    self.db.execute("UPDATE memories SET salience=?, updated_at=? WHERE id=?", (new_sal, int(time.time() * 1000), dict_m["id"]))
                    tot_chg += 1

                tot_proc += 1
                await asyncio.sleep(0)

        self.db.commit()
        dur = (time.time() - t0) * 1000
        logger.info(f"[decay] {tot_chg}/{tot_proc} | tiers: {tier_counts} | comp={tot_comp} fp={tot_fp} | {dur:.1f}ms")
