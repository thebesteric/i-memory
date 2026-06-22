from typing import List

from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import timing

from infra.ai.local_models.embed_manager import EmbedManager


class LocalEmbed(BaseEmbedModel):

    def __init__(self, model: str = None, dim: int = 1536):
        super().__init__(dim=dim)
        self.model = model
        self.client = EmbedManager(model_name_or_path=model)

    @timing
    async def embed(self, text: str, model: str = None, dim: int = None) -> List[float]:
        return (await self.embed_batch([text], model, dim))[0]

    @timing
    async def embed_batch(self, texts: List[str], model: str = None, dim: int = None) -> List[List[float]]:
        return await self.client.embed_batch(texts, dim=dim or self.dim)
