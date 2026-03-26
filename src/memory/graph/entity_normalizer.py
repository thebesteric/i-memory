import string

import numpy as np
from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import singleton
from sklearn.metrics.pairwise import cosine_similarity

from src.core.components import get_embed_model
from src.core.db import get_db
from src.memory.models.graph_models import Entity
from src.memory.models.memory_models import IMemoryUserIdentity


@singleton
class EntityNormalizer:

    def __init__(self, threshold: float = 0.85):
        # 数据库对象
        self.db = get_db()
        # 嵌入模型
        self.embed_model: BaseEmbedModel = get_embed_model()
        # 相似度阈值，超过该值则认为是同一实体
        self.threshold = threshold
        # canonical_text -> embedding
        self.canonical_embeddings: dict[str, list[float]] = {}
        # text -> canonical_text
        self.canonical_cache: dict[str, str] = {}

    @staticmethod
    def _get_cache_key(*, user_identity: IMemoryUserIdentity, entity: Entity) -> str:
        """
        生成缓存键
        """
        return f"{user_identity.id}:{entity.entity_type.value}:{entity.text}"

    @staticmethod
    def _get_embedding_key(canonical_id: str, entity: Entity) -> str:
        """
        生成向量存储键
        """
        return f"{canonical_id}:{entity.entity_type.value}"

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        """
        return cosine_similarity([vec1], [vec2])[0][0]

    async def load_canonical_entities(self):
        rows = self.db.fetchall(
            """
            SELECT id, name, entity_type, entity_label, vector
            FROM canonical_entities
            WHERE vector IS NOT NULL
            """)

    async def normalize(self, *, user_identity: IMemoryUserIdentity, entity: Entity, threshold: float = 0.85) -> str:
        # 1. 清理文本，并转换为小写
        text = self.clean_text(entity.text).lower()


    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗文本：去除所有空格 + 中英文标点符号
        :param text: 原始文本
        :return: 清洗后的纯文本
        """
        # 1. 去除所有空格（包括半角、全角空格）
        text = text.replace(" ", "").replace("　", "")

        # 2. 定义中英文标点集合
        # 英文标点
        en_punctuation = set(string.punctuation)
        # 中文标点（常用全覆盖）
        cn_punctuation = "，。！？；：“”‘’（）【】《》…—·、『』「」°"

        # 3. 遍历去除所有标点
        for p in en_punctuation:
            text = text.replace(p, "")
        for p in cn_punctuation:
            text = text.replace(p, "")

        return text
