from typing import List

from openai import AsyncOpenAI

from src.ai.embed.base_embed_model import BaseEmbedModel
from src.core.config import env
from src.utils.singleton import singleton


@singleton
class OpenAIEmbed(BaseEmbedModel):

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        super().__init__()
        self.api_key = api_key or env.OPENAI_API_KEY
        self.base_url = base_url or env.OPENAI_BASE_URL
        self.model = model or env.OPENAI_EMBEDDING_MODEL
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)


    async def embed(self, text: str, model: str = None, dim: int = None) -> List[float]:
        return (await self.embed_batch([text], model, dim))[0]

    async def embed_batch(self, texts: List[str], model: str = None, dim: int = None) -> List[List[float]]:
        self.model = model or self.model
        res = await self.client.embeddings.create(input=texts, model=self.model, dimensions=dim or self.dim)
        return [d.embedding for d in res.data]
