import math
from typing import List
import numpy as np
from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import singleton

from src.core.sector_classify import SEC_WTS
from src.tools.text import canonicalize_token, synonyms_for, canonical_tokens_from_text


@singleton
class SyntheticEmbed(BaseEmbedModel):

    def __init__(self, dim: int = 1536):
        """
        初始化合成向量模型，设置输出向量维度
        :param dim: 向量维度
        """
        super().__init__(dim=dim)
        self.model = "synthetic"

    async def embed(self, text: str, sector: str = None, dim: int = None) -> List[float]:
        """
        生成单条文本的合成向量，默认扇区为 semantic
        :param text: 文本
        :param sector: 扇区
        :param dim: 向量维度
        :return:
        """
        return self._gen_syn_emb(text, sector or "semantic")

    async def embed_batch(self, texts: List[str], sector: str = None, dim: int = None) -> List[List[float]]:
        """批量生成合成向量，所有文本共享同一个扇区上下文。"""
        return [self._gen_syn_emb(t, sector or "semantic") for t in texts]

    @staticmethod
    def _fnv1a(v: str) -> int:
        """FNV-1a 哈希：用于把特征稳定映射到向量索引。"""
        h = 0x811c9dc5
        for c in v:
            h = (h ^ ord(c)) * 16777619
            h &= 0xffffffff
        return h

    def _murmurish(self, v: str, seed: int) -> int:
        """轻量 Murmur 风格哈希：作为第二路哈希降低单点冲突。"""
        h = seed
        for c in v:
            h = (h ^ ord(c)) * 0x5bd1e995
            h &= 0xffffffff
            h = (h >> 13) ^ h
            h &= 0xffffffff
        return h

    def _add_feat(self, vec: np.ndarray, k: str, w: float):
        """将一个离散特征以带符号哈希方式写入向量（两路索引）。"""
        h = self._fnv1a(k)
        h2 = self._murmurish(k, 0xdeadbeef)
        val = w * (1.0 - float((h & 1) << 1))

        if (self.dim & (self.dim - 1)) == 0:
            vec[h & (self.dim - 1)] += val
            vec[h2 & (self.dim - 1)] += val * 0.5
        else:
            vec[h % self.dim] += val
            vec[h2 % self.dim] += val * 0.5

    def _add_pos_feat(self, vec: np.ndarray, pos: int, w: float):
        """写入位置特征，用正余弦编码补充顺序信息。"""
        idx = pos % self.dim
        ang = pos / pow(10000, (2 * idx) / self.dim)
        vec[idx] += w * math.sin(ang)
        vec[(idx + 1) % self.dim] += w * math.cos(ang)

    def _gen_syn_emb(self, t: str, s: str) -> List[float]:
        """核心合成嵌入流程：分词扩展、特征哈希、位置编码、归一化。"""
        v = np.zeros(self.dim, dtype=np.float32)
        ct = canonical_tokens_from_text(t)

        if not ct:
            return (np.ones(self.dim, dtype=np.float32) / math.sqrt(self.dim)).tolist()

        et = []
        for tok in ct:
            et.append(tok)
            syns = synonyms_for(tok)
            if syns:
                for syn in syns: et.append(canonicalize_token(syn))

        el = len(et)
        if el == 0: return (np.ones(self.dim, dtype=np.float32) / math.sqrt(self.dim)).tolist()

        tc = {}
        for tok in et: tc[tok] = tc.get(tok, 0) + 1

        sw = SEC_WTS.get(s, 1.0)

        for tok, c in tc.items():
            tf = c / el
            idf = math.log(1 + el / c)
            w = (tf * idf + 1) * sw
            self._add_feat(v, f"{s}|tok|{tok}", w)
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    self._add_feat(v, f"{s}|c3|{tok[i:i + 3]}", w * 0.4)

        for i in range(len(ct) - 1):
            a, b = ct[i], ct[i + 1]
            pw = 1.0 / (1.0 + i * 0.1)
            self._add_feat(v, f"{s}|bi|{a}_{b}", 1.4 * sw * pw)

        dl = math.log(1 + el)
        for i in range(min(len(ct), 50)):
            self._add_pos_feat(v, i, (0.5 * sw) / dl)

        n = np.linalg.norm(v)
        if n > 0: v /= n
        return v.tolist()
