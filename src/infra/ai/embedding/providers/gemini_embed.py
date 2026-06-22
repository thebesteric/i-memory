import httpx
from typing import List

from agile.db.vector.base.base_embed_model import BaseEmbedModel
from agile.utils import singleton, timing

from shared.config.settings import env


@singleton
class GeminiEmbed(BaseEmbedModel):

    def __init__(self, api_key: str = None, dim: int = 1536):
        super().__init__(dim=dim)
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
        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        url = f"{self.base_url}/{model_name}:batchEmbedContents?key={self.api_key}"
        reqs = []
        for t in texts:
            reqs.append({
                "model": model_name,
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
