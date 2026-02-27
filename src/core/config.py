import os
from dataclasses import dataclass

import dotenv
import pyrootutils
from utils.singleton import singleton

from src.core.constants import ModelProvider, VectorStoreProvider


@singleton
@dataclass
class EnvConfig:
    def __init__(self, env_file_path: str = f"{pyrootutils.find_root()}/.env"):
        # 加载环境变量文件
        dotenv.load_dotenv(dotenv_path=env_file_path, override=True)

        # Web 服务配置
        self.WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
        self.WEB_PORT = int(os.getenv("WEB_PORT", 8000))
        self.WEB_DEBUG = os.getenv("WEB_DEBUG", "false").lower() in ("true", "1")

        # 数据库配置
        self.POSTGRES_DB_HOST = os.getenv("DB_HOST", "localhost")
        self.POSTGRES_DB_PORT = int(os.getenv("DB_PORT", 5432))
        self.POSTGRES_DB_NAME = os.getenv("DB_NAME", "i-memory")
        self.POSTGRES_DB_USER = os.getenv("DB_USER", "admin")
        self.POSTGRES_DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
        self.POSTGRES_DB_AUTOCOMMIT = os.getenv("DB_AUTOCOMMIT", "true").lower() in ("true", "1")
        self.POSTGRES_DB_URL = f"postgresql://{self.POSTGRES_DB_USER}:{self.POSTGRES_DB_PASSWORD}@{self.POSTGRES_DB_HOST}:{self.POSTGRES_DB_PORT}/{self.POSTGRES_DB_NAME}"

        # Redis 配置
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # OpenAI 配置
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        # Gemini 配置
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
        self.GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

        # DashScope 配置
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        self.DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.DASHSCOPE_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")

        # 模型提供商（包含向量和记忆相关识别模型）
        self.MODEL_PROVIDER = os.getenv("IM_MODEL_PROVIDER", ModelProvider.DASHSCOPE.value)
        # 向量存储提供商
        self.IM_VECTOR_STORE = os.getenv("IM_VECTOR_STORE", VectorStoreProvider.POSTGRES.value)

        # 向量维度
        self.VEC_DIM = int(os.getenv("IM_VEC_DIM", 1536))
        # 最小向量维度
        self.MIN_VEC_DIM = int(os.getenv("IM_MIN_VEC_DIM", 64))
        # 最大向量维度
        self.MAX_VEC_DIM = int(os.getenv("IM_MAX_VEC_DIM", 1536))

        # 摘要最大长度
        self.SUMMARY_MAX_LENGTH = int(os.getenv("IM_SUMMARY_MAX_LENGTH", 1000))
        # 摘要层数
        self.SUMMARY_LAYERS = int(os.getenv("IM_SUMMARY_LAYERS", 3))
        # 是否只使用摘要进行信息检索
        self.USE_SUMMARY_ONLY = os.getenv("IM_USE_SUMMARY_ONLY", "true").lower() in ("true", "1")

        # 扇区存储大小
        self.SECTOR_SIZE = int(os.getenv("IM_SECTOR_SIZE", 10000))

        # 文本相似度阈值（用于记忆激活）
        self.SIMILARITY_THRESHOLD = float(os.getenv("IM_SIMILARITY_THRESHOLD", 0.95))

        # ================ 衰减相关配置 ================
        # 衰减线程数
        self.DECAY_THREADS = int(os.getenv("IM_DECAY_THREADS", 3))
        # 衰减冷阈值
        self.DECAY_COLD_THRESHOLD = float(os.getenv("IM_DECAY_COLD_THRESHOLD", 0.25))


env = EnvConfig()
