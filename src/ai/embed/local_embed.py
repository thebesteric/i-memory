from typing import List

from agile.db.vector.base.base_embed_model import BaseEmbedModel

from src.ai.model.embed.embed_manager import EmbedManager


class LocalEmbed(BaseEmbedModel):

    def __init__(self, model: str = None):
        super().__init__()
        self.model = model
        self.client = EmbedManager(model_name_or_path=model)

    async def embed(self, text: str, model: str = None, dim: int = None) -> List[float]:
        return (await self.embed_batch([text], model, dim))[0]

    async def embed_batch(self, texts: List[str], model: str = None, dim: int = None) -> List[List[float]]:
        return await self.client.embed_batch(texts, dim=dim or self.dim)
