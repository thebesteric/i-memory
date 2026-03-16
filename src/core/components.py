import os

import pyrootutils
from agile.db.vector.milvus.milvus_manager import MilvusManager
from pymilvus import FieldSchema, DataType

from src.ai.model_provider import get_embed_model
from src.core.config import env
from src.core.constants import COMPONENTS_CACHE
from src.core.sector_classify import SectorClassifier


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


def get_sector_classifier() -> SectorClassifier:
    """
    获取语义分类器实例
    :return: SectorClassifier 实例
    """
    sector_classifier = COMPONENTS_CACHE.get(SectorClassifier.__name__)
    if not sector_classifier:
        if env.USE_BERT_CLASSIFIER:
            checkpoint_path = os.path.join(
                pyrootutils.find_root(),
                "assets",
                "bert",
                "checkpoint",
                "checkpoint.pth",
            )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Sector classifier checkpoint not found: {checkpoint_path}")
            sector_classifier = SectorClassifier(checkpoint_path=checkpoint_path)
        else:
            sector_classifier = SectorClassifier()
        COMPONENTS_CACHE.set(SectorClassifier.__name__, sector_classifier)
    return sector_classifier
