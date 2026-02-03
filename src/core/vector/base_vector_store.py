from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from src.core.config import env
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


class VectorRow:

    def __init__(self, id: str, sector: str, vector: List[float], dim: int):
        self.id = id
        self.sector = sector
        self.vector = vector
        self.dim = dim


class BaseVectorStore(ABC):
    @abstractmethod
    async def store_vector(self, id: str, sector: str, vector: List[float], dim: int, user_id: Optional[str] = None):
        pass

    @abstractmethod
    async def get_vectors_by_id(self, id: str) -> List[VectorRow]:
        pass

    @abstractmethod
    async def get_vector(self, id: str, sector: str) -> Optional[VectorRow]:
        pass

    @abstractmethod
    async def delete_vectors(self, id: str):
        pass

    @abstractmethod
    async def search(self, vector: List[float], sector: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
