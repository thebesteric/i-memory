import numpy as np
from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import singleton
from sklearn.metrics.pairwise import cosine_similarity

from src.core.components import get_embed_model
from src.core.db import get_db
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
    def _get_cache_key(*, user_identity: IMemoryUserIdentity, entity_type: str, text: str) -> str:
        """
        生成缓存键
        """
        return f"{user_identity.id}:{entity_type}:{text}"

    @staticmethod
    def _get_embedding_key(canonical_id: str, entity_type: str) -> str:
        """
        生成向量存储键
        """
        return f"{canonical_id}:{entity_type}"

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

    async def normalize(self, *, user_identity: IMemoryUserIdentity, entity_type: str, text: str, threshold: float = 0.85) -> str:
        pass
