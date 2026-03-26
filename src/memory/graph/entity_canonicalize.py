import string

import numpy as np
from agile.cache import MemoryCache
from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import singleton
from sklearn.metrics.pairwise import cosine_similarity

from src.core.components import get_embed_model
from src.core.db import get_db
from src.memory.models.graph_models import Entity, CanonicalEntity
from src.memory.models.memory_models import IMemoryUser


@singleton
class EntityCanonicalize:

    def __init__(self, threshold: float = 0.85):
        # 数据库对象
        self.db = get_db()
        # 嵌入模型
        self.embed_model: BaseEmbedModel = get_embed_model()
        # 加载标准化实体
        self.canonical_entity_cache: MemoryCache = self.load_canonical_entities()
        # 相似度阈值，超过该值则认为是同一实体
        self.threshold = threshold
        # canonical_text -> embedding
        self.canonical_embeddings: dict[str, list[float]] = {}

    def load_canonical_entities(self) -> MemoryCache:
        canonical_entity_cache = MemoryCache()
        rows = self.db.fetchall(
            """
            SELECT id,
                   name,
                   entity_type,
                   entity_label,
                   vector,
                   occurrence_count,
                   first_seen_at,
                   last_seen_at,
                   created_at,
                   updated_at
            FROM graph_canonical_entities
            WHERE vector IS NOT NULL
            """)

        # 加入缓存
        for row in rows or []:
            canonical_entity_cache.set(row["id"], CanonicalEntity.from_dict(row))

        return canonical_entity_cache

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        """
        return cosine_similarity([vec1], [vec2])[0][0]

    @classmethod
    def _generate_embedding_text(cls, entity: Entity) -> str:
        # 清理文本，并转换为小写
        text = cls.clean_text(entity.text)
        return f"{entity.entity_type.name}|{text}"

    async def canonicalize(self, *, user: IMemoryUser, entity: Entity, threshold: float = 0.85) -> CanonicalEntity:
        """
        规范化
        1. 实体抽取后，将原始实体文本小写化，去除多余空格，作为 canonical_name
        2. 查找候选实体：用小写化的名称做模糊匹配（包含、trigram、ILIKE）
        3. 打分消歧：结合名称相似度、上下文共现、时间接近性，选出最优候选
        4. 唯一性保证：新建实体时用 ON CONFLICT 保证同名（小写）只会有一个实体
        5. 返回标准实体
        :param user: 用户
        :param entity: 实体
        :param threshold: 阈值
        :return:
        """

        # 查询缓存
        if entity.canonical_id:
            return self.canonical_entity_cache.get(entity.canonical_id)

        # TODO 将 entity 的 text 和 entity_type 进行 embedding，然后去 canonical_entities 中计算相似度，找到最符合的，找到就返回
        # 向量化
        vector = await self.embed_model.embed(self._generate_embedding_text(entity))

        # 获取所有的标准化实体
        canonical_entities: list[CanonicalEntity] = self.canonical_entity_cache.values()

        # 相似度最高的标准化实体
        best_similarity: tuple[CanonicalEntity, float] | None = None
        for canonical_entity in canonical_entities:
            # 计算相似度
            similarity: float = self.embed_model.similarity(vector, canonical_entity.vector)
            # 找到相似度最高的
            if best_similarity is None or similarity > best_similarity[1]:
                best_similarity = (canonical_entity, similarity)

        if best_similarity:
            pass

        return None

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
        # 中文标点
        cn_punctuation = "，。！？；：“”‘’（）【】《》…—·、『』「」°"

        # 3. 遍历去除所有标点
        for p in en_punctuation:
            text = text.replace(p, "")
        for p in cn_punctuation:
            text = text.replace(p, "")

        # 转换为小写
        return text.strip().lower()
