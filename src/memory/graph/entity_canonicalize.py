import datetime
import string
import uuid

import numpy as np
from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import singleton, LogHelper
from sklearn.metrics.pairwise import cosine_similarity

from src.core.components import get_embed_model
from src.core.db import get_db
from src.memory.graph.graph_models import Entity, CanonicalEntity
from src.memory.memory_models import IMemoryUser

logger = LogHelper.get_logger()


@singleton
class EntityCanonicalize:

    def __init__(self, threshold: float = 0.85):
        # 数据库对象
        self.db = get_db()
        # 嵌入模型
        self.embed_model: BaseEmbedModel = get_embed_model()
        # 加载标准化实体
        self.canonical_entity_cache: dict[str, dict[str, CanonicalEntity]] = self.load_canonical_entities()
        # 相似度阈值，超过该值则认为是同一实体
        self.threshold = threshold
        # canonical_text -> embedding
        self.canonical_embeddings: dict[str, list[float]] = {}

    def load_canonical_entities(self) -> dict[str, dict[str, CanonicalEntity]]:
        canonical_entity_cache: dict[str, dict[str, CanonicalEntity]] = {}
        rows = self.db.fetchall(
            """
            SELECT id,
                   user_id,
                   name,
                   entity_type,
                   entity_label,
                   vector,
                   occurrence_count,
                   first_seen_at,
                   last_seen_at,
                   is_active,
                   created_at,
                   updated_at
            FROM graph_canonical_entities
            WHERE vector IS NOT NULL
              AND is_active = TRUE
            """)

        # 加入缓存
        for row in rows or []:
            entity_id = row["id"]
            user_id = row["user_id"]
            if user_id not in canonical_entity_cache:
                canonical_entity_cache[user_id] = {}
            canonical_entity_cache[user_id][entity_id] = CanonicalEntity.from_dict(row)

        return canonical_entity_cache

    def _canonicalize_entity(self, user: IMemoryUser, entity: Entity, vector: list[float], conn=None) -> CanonicalEntity:
        """
        标准化实体对象
        :param user: 用户
        :param entity: 实体
        :param vector: 向量
        :param conn: 数据库连接对象
        :return:
        """
        now = datetime.datetime.now()
        _id = str(uuid.uuid4())
        entity_type = entity.entity_type
        # 写入数据库
        self.db.execute(
            """
            INSERT INTO graph_canonical_entities (id, user_id, name, entity_type, entity_label, vector, occurrence_count, first_seen_at, last_seen_at,
                                                  created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, name, entity_type) DO NOTHING
            """,
            (_id, user.id, entity.text, entity_type.name, entity_type.label, vector, 1, now, now, now, now),
            conn=conn
        )
        return CanonicalEntity(
            id=_id,
            name=entity.text,
            entity_type=entity.entity_type,
            vector=vector,
            occurrence_count=1,
            first_seen_at=now,
            last_seen_at=now,
            created_at=now,
            updated_at=now
        )

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

    async def canonicalize(self, user: IMemoryUser, entity: Entity, conn=None) -> CanonicalEntity:
        """
        规范化实体：将输入的实体与标准化实体进行匹配，找到最相似的标准化实体，如果相似度超过阈值则返回该标准化实体，否则返回 None
        :param user: 用户
        :param entity: 实体
        :param conn: 数据库连接对象
        :return:
        """
        # 查询缓存
        if entity.canonical_id:
            return self.canonical_entity_cache.get(user.id, {}).get(entity.canonical_id)

        # 将 entity 的 text 和 entity_type 进行 embedding，然后去 canonical_entities 中计算相似度，找到最符合的，找到就返回
        # 实体向量化
        vector = await self.embed_model.embed(self._generate_embedding_text(entity))

        # 获取所有的标准化实体
        canonical_entities: list[CanonicalEntity] = list(self.canonical_entity_cache.get(user.id, {}).values())

        # 相似度最高的标准化实体
        best_similarity: tuple[CanonicalEntity, float] | None = None
        for canonical_entity in canonical_entities:
            # 计算相似度
            similarity: float = self.embed_model.similarity(vector, canonical_entity.vector)
            # 找到相似度最高的
            if best_similarity is None or similarity > best_similarity[1]:
                best_similarity = (canonical_entity, similarity)

        # 相似度最高的实体的阈值 >= 阈值
        if best_similarity and best_similarity[1] >= self.threshold:
            logger.info(f"[GRAPH] Entity '{entity.text}' matched with canonical entity '{best_similarity[0].name}', similarity: {best_similarity[1]} ")
            return best_similarity[0]

        # 将实体标准化操作
        canonical_entity = self._canonicalize_entity(user, entity, vector, conn=conn)
        # 加入缓存
        self.canonical_entity_cache.setdefault(user.id, {})[entity.canonical_id] = canonical_entity

        return canonical_entity

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
