from agile.utils import LogHelper
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.ai.embed.base_embed_model import BaseEmbedModel
from src.ai.embed.local_embed import LocalEmbed
from src.ai.model.embed.embed_manager import EmbedManager
from src.core.config import env
from src.core.constants import ModelProvider, EMBED_MODEL_CACHE, MODEL_CACHE

logger = LogHelper.get_logger()


def get_embed_model() -> BaseEmbedModel:
    """
    根据配置获取嵌入模型实例
    :return:
    """
    embed_model_provider = env.EMBED_MODEL_PROVIDER or ModelProvider.OPENAI.value
    # OpenAI 嵌入模型
    if embed_model_provider == ModelProvider.OPENAI.value:
        from src.ai.embed.openai_embed import OpenAIEmbed
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            ModelProvider.OPENAI.value,
            lambda: OpenAIEmbed(),
            on_set=lambda k, v: logger.info(f"Using OpenAI embedding model: {v.model}")
        )
        return _embed_model
    # Gemini 嵌入模型
    if embed_model_provider == ModelProvider.GEMINI.value:
        from src.ai.embed.gemini_embed import GeminiEmbed
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            ModelProvider.GEMINI.value,
            lambda: GeminiEmbed(),
            on_set=lambda k, v: logger.info(f"Using Gemini embedding model: {v.model}")
        )
        return _embed_model
    # DashScope 嵌入模型
    if embed_model_provider == ModelProvider.DASHSCOPE.value:
        from src.ai.embed.dashscope_embed import DashScopeEmbed
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            ModelProvider.DASHSCOPE.value,
            lambda: DashScopeEmbed(),
            on_set=lambda k, v: logger.info(f"Using DashScope embedding model: {v.model}")
        )
        return _embed_model
    # Local 嵌入模型
    if embed_model_provider == ModelProvider.LOCAL.value:
        _embed_model = EMBED_MODEL_CACHE.get_or_set(
            ModelProvider.LOCAL.value,
            lambda: LocalEmbed(model=EmbedManager.DEFAULT_MODEL_NAME_OR_PATH),
            on_set=lambda k, v: logger.info(f"Using Local Embed model: {v.model}")
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
            lambda: ChatOpenAI(model=env.OPENAI_MODEL, temperature=0.0, api_key=env.OPENAI_API_KEY, base_url=env.OPENAI_BASE_URL),
            on_set=lambda k, v: logger.info(f"Using OpenAI model: {env.OPENAI_MODEL}")
        )
    if model_provider == ModelProvider.GEMINI.value:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return MODEL_CACHE.get_or_set(
            ModelProvider.GEMINI.value,
            lambda: ChatGoogleGenerativeAI(model=env.GEMINI_MODEL, api_key=env.GEMINI_API_KEY, base_url=env.GEMINI_BASE_URL),
            on_set=lambda k, v: logger.info(f"Using Gemini chat model: {env.GEMINI_MODEL}")
        )
    if model_provider == ModelProvider.DASHSCOPE.value:
        return MODEL_CACHE.get_or_set(
            ModelProvider.DASHSCOPE.value,
            lambda: ChatOpenAI(model=env.DASHSCOPE_MODEL, temperature=0.0, api_key=env.DASHSCOPE_API_KEY, base_url=env.DASHSCOPE_BASE_URL),
            on_set=lambda k, v: logger.info(f"Using DashScope model: {env.DASHSCOPE_MODEL}")
        )

    raise ValueError(f"Unsupported chat model: {model_provider}")


# 全局向量存储实例
embed_model: BaseEmbedModel = get_embed_model()
# 全局大语言模型实例
chat_model: BaseChatModel = get_chat_model()
