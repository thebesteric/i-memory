from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.config import env
from src.memory.models.memory_models import IMemoryFilters
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


class VectorRow:

    def __init__(self, id: str, sector: str, vector: List[float], dim: int):
        self.id = id
        self.sector = sector
        self.vector = vector
        self.dim = dim


class VectorSearch:

    def __init__(self, id: str, similarity: float):
        self.id = id
        self.similarity = similarity


class BaseVectorStore(ABC):
    @abstractmethod
    async def store_vector(self, id: str, sector: str, vector: List[float], dim: int, user_id: Optional[str] = None):
        """
        存储向量
        :param id: 唯一标识
        :param sector: 扇区名称
        :param vector: 向量列表
        :param dim: 向量维度
        :param user_id: 用户标识
        :return:
        """
        pass

    @abstractmethod
    async def get_vectors_by_id(self, id: str) -> List[VectorRow]:
        """
        根据 ID 获取所有相关向量
        :param id: 唯一标识
        :return:
        """
        pass

    @abstractmethod
    async def get_vector(self, id: str, sector: str) -> Optional[VectorRow]:
        """
        根据 ID 和 sector 获取单个向量
        :param id: 唯一标识
        :param sector: 扇区名称
        :return:
        """
        pass

    @abstractmethod
    async def delete_vectors(self, id: str):
        """
        删除指定 ID 的所有向量
        :param id: 唯一标识
        :return:
        """
        pass

    @abstractmethod
    async def search(self, vector: List[float], sector: str, k: int, filters: IMemoryFilters = None) -> List[VectorSearch]:
        """
        相似度搜索
        :param vector: 向量列表
        :param sector: 扇区名称
        :param k: 返回结果数量
        :param filters: 过滤条件
        :return:
        """
        pass


def get_vector_store() -> BaseVectorStore:
    """
    根据配置获取向量存储后端实例
    :return:
    """
    backend = env.IM_VECTOR_STORE or "postgres"
    if backend == "postgres":
        from src.core.vector.postgres_vector_store import PostgresVectorStore
        dsn = env.POSTGRES_DB_URL
        logger.info(f"Using PostgresVectorStore at {dsn}")
        return PostgresVectorStore(dsn)
    elif backend == "valkey" or backend == "redis":
        from src.core.vector.redis_vector_store import RedisVectorStore
        url = env.REDIS_URL
        logger.info(f"Using RedisVectorStore at {url}")
        return RedisVectorStore(url)

    raise ValueError(f"Unsupported vector store backend: {backend}")


# 全局向量存储实例
vector_store: BaseVectorStore = get_vector_store()
