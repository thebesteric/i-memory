from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.ai.embed.base_embed_model import BaseEmbedModel
from src.core.config import env
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


def get_embed_model() -> BaseEmbedModel:
    """
    根据配置获取嵌入模型实例
    :return:
    """
    model_provider = env.MODEL_PROVIDER or "openai"
    if model_provider == "openai":
        from src.ai.embed.openai_embed import OpenAIEmbed
        _embed_model = OpenAIEmbed()
        logger.info(f"Using OpenAI embedding model: {_embed_model.model}")
        return _embed_model
    if model_provider == "gemini":
        from src.ai.embed.gemini_embed import GeminiEmbed
        _embed_model = GeminiEmbed()
        logger.info(f"Using Gemini embedding model: {_embed_model.model}")
        return _embed_model
    if model_provider == "dashscope":
        from src.ai.embed.dashscope_embed import DashScopeEmbed
        _embed_model = DashScopeEmbed()
        logger.info(f"Using DashScope embedding model: {_embed_model.model}")
        return _embed_model

    raise ValueError(f"Unsupported embed model: {model_provider}")


def get_chat_model() -> BaseChatModel:
    """
    根据配置获取大语言模型实例
    :return:
    """
    model_provider = env.MODEL_PROVIDER or "openai"
    if model_provider == "openai":
        return ChatOpenAI(model=env.OPENAI_MODEL, temperature=0.0, api_key=env.OPENAI_API_KEY, base_url=env.OPENAI_BASE_URL)
    if model_provider == "dashscope":
        return ChatOpenAI(model=env.DASHSCOPE_MODEL, temperature=0.0, api_key=env.DASHSCOPE_API_KEY, base_url=env.DASHSCOPE_BASE_URL)
    if model_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=env.GEMINI_CHAT_MODEL, api_key=env.GEMINI_API_KEY, base_url=env.GEMINI_BASE_URL)

    raise ValueError(f"Unsupported chat model: {model_provider}")


# 全局向量存储实例
embed_model: BaseEmbedModel = get_embed_model()
# 全局大语言模型实例
chat_model: BaseChatModel = get_chat_model()
