import os

import pyrootutils
from agile.cache import MemoryCache
from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.db.vector.milvus.milvus_manager import MilvusManager
from agile.utils import LogHelper, TimeUnit
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pymilvus import FieldSchema, DataType

from shared.config.settings import env
from shared.config.constants import VectorStoreProvider, ModelProvider, EmbedModelProvider

from infra.vector_store.base import BaseVectorStore

logger = LogHelper.get_logger()

# 记忆查询缓存
MEMORIES_CACHE = MemoryCache(
    maxsize=2048,
    default_ttl=60,
    time_unit=TimeUnit.SECONDS
)

# 记忆分类缓存（用于存放记忆分类结果，减少重复计算）
QUERY_CLASSIFY_CACHE = MemoryCache()

# 组件查询缓存（用于存放一些单例的组件）
COMPONENTS_CACHE = MemoryCache()

# 模型缓存
MODEL_CACHE = MemoryCache()

# 嵌入模型缓存
EMBED_MODEL_CACHE = MemoryCache()

# 用户画像缓存
USER_PROFILE_CACHE = MemoryCache(default_ttl=12, time_unit=TimeUnit.HOURS)

# 用户身份缓存
USER_IDENTITY_CACHE = MemoryCache(default_ttl=60, time_unit=TimeUnit.MINUTES)

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


def get_sector_classifier():
    """
    获取语义分类器实例
    :return: SectorClassifier 实例
    """
    from services.memory.sector_classify import SectorClassifier
    sector_classifier: SectorClassifier = COMPONENTS_CACHE.get(SectorClassifier.__name__)
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
            # 如果 Bert 模型不存在会自动下载相应的模型
            sector_classifier = SectorClassifier(checkpoint_path=checkpoint_path)
        else:
            sector_classifier = SectorClassifier()
        COMPONENTS_CACHE.set(SectorClassifier.__name__, sector_classifier)
    return sector_classifier


def get_vector_store() -> BaseVectorStore:
    """
    根据配置获取向量存储后端实例
    :return:
    """
    backend = env.VECTOR_STORE or VectorStoreProvider.POSTGRES.value
    if backend == VectorStoreProvider.POSTGRES.value:
        from infra.vector_store.postgres_impl import PostgresVectorStore
        dsn = env.POSTGRES_DB_URL
        _vector_store: PostgresVectorStore = COMPONENTS_CACHE.get_or_set(
            backend,
            lambda: PostgresVectorStore(dsn),
            on_set=lambda k, v: logger.info(f"Using PostgresVectorStore at {dsn}")
        )
        return _vector_store
    elif backend == VectorStoreProvider.VALKEY.value or backend == VectorStoreProvider.REDIS.value:
        from infra.vector_store.redis_impl import RedisVectorStore
        url = env.REDIS_URL
        password = env.REDIS_PASSWORD
        _vector_store: RedisVectorStore = COMPONENTS_CACHE.get_or_set(
            backend,
            lambda: RedisVectorStore(url),
            on_set=lambda k, v: logger.info(f"Using RedisVectorStore at {url}")
        )
        return _vector_store

    raise ValueError(f"Unsupported vector store backend: {backend}")


def get_embed_model() -> BaseEmbedModel:
    """
    根据配置获取嵌入模型实例
    :return:
    """
    embed_model_provider = env.EMBED_MODEL_PROVIDER or EmbedModelProvider.OPENAI.value
    # OpenAI 嵌入模型
    if embed_model_provider == EmbedModelProvider.OPENAI.value:
        from infra.ai.embedding.providers.openai_embed import OpenAIEmbed
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            EmbedModelProvider.OPENAI.value,
            lambda: OpenAIEmbed(dim=env.VECTOR_DIM),
            on_set=lambda k, v: logger.info(f"Using OpenAI embedding model: {v.model}")
        )
        return _embed_model
    # Gemini 嵌入模型
    if embed_model_provider == EmbedModelProvider.GEMINI.value:
        from infra.ai.embedding.providers.gemini_embed import GeminiEmbed
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            EmbedModelProvider.GEMINI.value,
            lambda: GeminiEmbed(dim=env.VECTOR_DIM),
            on_set=lambda k, v: logger.info(f"Using Gemini embedding model: {v.model}")
        )
        return _embed_model
    # DashScope 嵌入模型
    if embed_model_provider == EmbedModelProvider.DASHSCOPE.value:
        from infra.ai.embedding.providers.dashscope_embed import DashScopeEmbed
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            EmbedModelProvider.DASHSCOPE.value,
            lambda: DashScopeEmbed(dim=env.VECTOR_DIM),
            on_set=lambda k, v: logger.info(f"Using DashScope embedding model: {v.model}")
        )
        return _embed_model
    # 本地嵌入模型
    if embed_model_provider == EmbedModelProvider.LOCAL.value:
        from infra.ai.embedding.providers.local_embed import LocalEmbed
        from infra.ai.local_models.embed_manager import EmbedManager
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            EmbedModelProvider.LOCAL.value,
            lambda: LocalEmbed(model=EmbedManager.DEFAULT_MODEL_NAME_OR_PATH, dim=env.VECTOR_DIM),
            on_set=lambda k, v: logger.info(f"Using Local Embed model: {v.model}")
        )
        return _embed_model
    # 合成嵌入模型
    if embed_model_provider == EmbedModelProvider.SYNTHETIC.value:
        from infra.ai.embedding.providers.synthetic_embed import SyntheticEmbed
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            EmbedModelProvider.SYNTHETIC.value,
            lambda: SyntheticEmbed(dim=env.VECTOR_DIM),
            on_set=lambda k, v: logger.info(f"Using Synthetic Embed model: {v.model}")
        )
        return _embed_model

    raise ValueError(f"Unsupported embed model: {embed_model_provider}")


def get_chat_model() -> BaseChatModel:
    """
    根据配置获取大语言模型实例
    :return:
    """
    model_provider = env.MODEL_PROVIDER or ModelProvider.OPENAI.value
    if model_provider == ModelProvider.OPENAI.value:
        return MODEL_CACHE.get_or_set(
            ModelProvider.OPENAI.value,
            lambda: ChatOpenAI(
                model=env.OPENAI_MODEL,
                temperature=0.0,
                api_key=env.OPENAI_API_KEY,
                base_url=env.OPENAI_BASE_URL,
                extra_body={"enable_thinking": False}
            ),
            on_set=lambda k, v: logger.info(f"Using OpenAI model: {env.OPENAI_MODEL}")
        )
    if model_provider == ModelProvider.GEMINI.value:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return MODEL_CACHE.get_or_set(
            ModelProvider.GEMINI.value,
            lambda: ChatGoogleGenerativeAI(
                model=env.GEMINI_MODEL,
                temperature=0.0,
                api_key=env.GEMINI_API_KEY,
                base_url=env.GEMINI_BASE_URL,
                include_thoughts=False,
                thinking_level="minimal",  # Gemini 3+，"minimal", "low", "medium", "high" (default for Pro)
                thinking_budget=0  # Gemini 2.5，0 (off), -1 (dynamic), or a positive integer (token limit)
            ),
            on_set=lambda k, v: logger.info(f"Using Gemini chat model: {env.GEMINI_MODEL}")
        )
    if model_provider == ModelProvider.DASHSCOPE.value:
        return MODEL_CACHE.get_or_set(
            ModelProvider.DASHSCOPE.value,
            lambda: ChatOpenAI(
                model=env.DASHSCOPE_MODEL,
                temperature=0.0,
                api_key=env.DASHSCOPE_API_KEY,
                base_url=env.DASHSCOPE_BASE_URL,
                extra_body={"enable_thinking": False}
            ),
            on_set=lambda k, v: logger.info(f"Using DashScope model: {env.DASHSCOPE_MODEL}")
        )

    raise ValueError(f"Unsupported chat model: {model_provider}")
