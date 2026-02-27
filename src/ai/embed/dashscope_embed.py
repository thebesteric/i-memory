import asyncio
from typing import List

from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from utils.singleton import singleton

from src.ai.embed.base_embed_model import BaseEmbedModel
from src.ai.embed.openai_embed import OpenAIEmbed
from src.core.config import env


@singleton
class DashScopeEmbed(OpenAIEmbed):

    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or env.DASHSCOPE_API_KEY,
            base_url=env.DASHSCOPE_BASE_URL,
            model=env.DASHSCOPE_EMBEDDING_MODEL
        )
