from agile.db.vector.milvus.milvus_manager import MilvusManager
from pymilvus import FieldSchema, DataType

from src.ai.model_provider import get_embed_model
from src.core.config import env
from src.core.constants import COMPONENTS_CACHE


def get_milvus_manager() -> MilvusManager:
    """
    获取 MilvusManager 实例
    :return: MilvusManager 实例
    """
    milvus_manager = COMPONENTS_CACHE.get(MilvusManager.__name__)
    if not milvus_manager:
        milvus_manager = MilvusManager(
            uri=env.MILVUS_URL,
            token=env.MILVUS_TOKEN,
            default_collection_name=env.MILVUS_COLLECTION_NAME,
            default_field_schemas=[
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=env.VECTOR_DIM),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ],
            embedding_model=get_embed_model(),
            vector_dim=env.VECTOR_DIM
        )
        COMPONENTS_CACHE.set(MilvusManager.__name__, milvus_manager)
    return milvus_manager
