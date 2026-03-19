import httpx
from typing import List

from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import singleton, timing

from src.core.config import env


@singleton
class GeminiEmbed(BaseEmbedModel):

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or env.GEMINI_API_KEY
        self.base_url = env.GEMINI_BASE_URL
        self.model = env.GEMINI_EMBEDDING_MODEL

    @timing
    async def embed(self, text: str, model: str = None, dim: int = None) -> List[float]:
        return (await self.embed_batch([text], model))[0]

    @timing
    async def embed_batch(self, texts: List[str], model: str = None, dim: int = None) -> List[List[float]]:
        if not self.api_key: raise ValueError("Gemini key missing")
        self.model = model or self.model
        if "models/" not in self.model:
            m = f"models/{self.model}"
        url = f"{self.base_url}/{self.model}:batchEmbedContents?key={self.api_key}"
        reqs = []
        for t in texts:
            reqs.append({
                "model": self.model,
                "content": {"parts": [{"text": t}]},
                "taskType": "SEMANTIC_SIMILARITY",
                "outputDimensionality": dim or self.dim
            })

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json={"requests": reqs})
            if res.status_code != 200:
                raise Exception(f"Gemini: {res.text}")
            data = res.json()
            if "embeddings" not in data:
                return []
            return [e["values"] for e in data["embeddings"]]
