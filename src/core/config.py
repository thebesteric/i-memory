from dataclasses import dataclass

import pyrootutils
from utils.env_helper import EnvHelper
from utils.singleton import singleton

from src.core.constants import ModelProvider, VectorStoreProvider


@singleton
@dataclass
class EnvConfig:
    def __init__(self, env_file_path: str = f"{pyrootutils.find_root()}/.env"):
        # 加载环境变量文件
        env_helper = EnvHelper(env_file_path=env_file_path, override=False)

        # Web 服务配置
        self.WEB_HOST = env_helper.get("WEB_HOST", "127.0.0.1")
        self.WEB_PORT = env_helper.get("WEB_PORT", 8000)
        self.WEB_DEBUG = env_helper.get("WEB_DEBUG", "false")

        # 数据库配置
        self.POSTGRES_DB_HOST = env_helper.get("DB_HOST", "localhost")
        self.POSTGRES_DB_PORT = env_helper.get("DB_PORT", 5432)
        self.POSTGRES_DB_NAME = env_helper.get("DB_NAME", "i-memory")
        self.POSTGRES_DB_USER = env_helper.get("DB_USER", "admin")
        self.POSTGRES_DB_PASSWORD = env_helper.get("DB_PASSWORD", "123456")
        self.POSTGRES_DB_AUTOCOMMIT = env_helper.get("DB_AUTOCOMMIT", True)
        self.POSTGRES_DB_URL = f"postgresql://{self.POSTGRES_DB_USER}:{self.POSTGRES_DB_PASSWORD}@{self.POSTGRES_DB_HOST}:{self.POSTGRES_DB_PORT}/{self.POSTGRES_DB_NAME}"

        # Redis 配置
        self.REDIS_URL = env_helper.get("REDIS_URL", "redis://localhost:6379/0")

        # OpenAI 配置
        self.OPENAI_API_KEY = env_helper.get("OPENAI_API_KEY")
        self.OPENAI_BASE_URL = env_helper.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.OPENAI_MODEL = env_helper.get("OPENAI_MODEL", "gpt-4o-mini")
        self.OPENAI_EMBEDDING_MODEL = env_helper.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        # Gemini 配置
        self.GEMINI_API_KEY = env_helper.get("GEMINI_API_KEY")
        self.GEMINI_BASE_URL = env_helper.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        self.GEMINI_MODEL = env_helper.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
        self.GEMINI_EMBEDDING_MODEL = env_helper.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

        # DashScope 配置
        self.DASHSCOPE_API_KEY = env_helper.get("DASHSCOPE_API_KEY")
        self.DASHSCOPE_BASE_URL = env_helper.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.DASHSCOPE_MODEL = env_helper.get("DASHSCOPE_MODEL", "qwen-plus")
        self.DASHSCOPE_EMBEDDING_MODEL = env_helper.get("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")

        # 模型提供商（包含向量和记忆相关识别模型）
        self.MODEL_PROVIDER = env_helper.get("IM_MODEL_PROVIDER", ModelProvider.DASHSCOPE.value)
        # 向量存储提供商
        self.IM_VECTOR_STORE = env_helper.get("IM_VECTOR_STORE", VectorStoreProvider.POSTGRES.value)

        # 向量维度
        self.VEC_DIM = env_helper.get("IM_VEC_DIM", 1536)
        # 最小向量维度
        self.MIN_VEC_DIM = env_helper.get("IM_MIN_VEC_DIM", 64)
        # 最大向量维度
        self.MAX_VEC_DIM = env_helper.get("IM_MAX_VEC_DIM", 1536)

        # 摘要最大长度
        self.SUMMARY_MAX_LENGTH = env_helper.get("IM_SUMMARY_MAX_LENGTH", 1000)
        # 摘要层数
        self.SUMMARY_LAYERS = env_helper.get("IM_SUMMARY_LAYERS", 3)
        # 是否只使用摘要进行信息检索
        self.USE_SUMMARY_ONLY = env_helper.get("IM_USE_SUMMARY_ONLY", True)

        # 扇区存储大小
        self.SECTOR_SIZE = env_helper.get("IM_SECTOR_SIZE", 10000)

        # 文本相似度阈值（用于记忆激活）
        self.SIMILARITY_THRESHOLD = env_helper.get("IM_SIMILARITY_THRESHOLD", 0.95)

        # ================ 衰减相关配置 ================
        # 衰减线程数
        self.DECAY_THREADS = env_helper.get("IM_DECAY_THREADS", 3)
        # 衰减冷阈值
        self.DECAY_COLD_THRESHOLD = env_helper.get("IM_DECAY_COLD_THRESHOLD", 0.25)


env = EnvConfig()
