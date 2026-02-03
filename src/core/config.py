import os
from dataclasses import dataclass

import dotenv
import pyrootutils
from injector import inject

from src.utils.singleton import singleton


@singleton
@dataclass
class EnvConfig:
    def __init__(self, env_file_path: str = f"{pyrootutils.find_root()}/.env"):
        # 加载环境变量文件
        dotenv.load_dotenv(dotenv_path=env_file_path, override=True)

        # 数据库配置
        self.POSTGRES_DB_HOST = os.getenv("DB_HOST", "localhost")
        self.POSTGRES_DB_PORT = os.getenv("DB_PORT", 5432)
        self.POSTGRES_DB_NAME = os.getenv("DB_NAME", "i-memory")
        self.POSTGRES_DB_USER = os.getenv("DB_USER", "admin")
        self.POSTGRES_DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
        self.POSTGRES_DB_AUTOCOMMIT = os.getenv("DB_AUTOCOMMIT", True)
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
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gpt-4o-mini")
        self.GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")


        # MiniLM 模型（相似度判断、意图解析）
        self.MINI_LM_MODEL = os.getenv("IM_MINI_LM_MODEL", "all-MiniLM-L12-v2")

        # 向量嵌入提供商
        self.EMB_KIND = os.getenv("IM_EMBED_KIND", "openai")

        # 向量维度（minilm 为 384）
        self.VEC_DIM = os.getenv("IM_VEC_DIM", 1536)

        # 摘要最大长度
        self.SUMMARY_MAX_LENGTH = os.getenv("IM_SUMMARY_MAX_LENGTH", 1000)

        # 扇区大小
        self.SECTOR_SIZE = os.getenv("IM_SECTOR_SIZE", 10000)

        # 是否只使用摘要进行信息检索
        self.USE_SUMMARY_ONLY = os.getenv("IM_USE_SUMMARY_ONLY", False)

        # 向量存储配置
        self.IM_VECTOR_STORE = os.getenv("IM_VECTOR_STORE", "postgres")

        # 文本相似度阈值（用于记忆激活）
        self.SIMILARITY_THRESHOLD = os.getenv("IM_SIMILARITY_THRESHOLD", 0.95)


env = EnvConfig()