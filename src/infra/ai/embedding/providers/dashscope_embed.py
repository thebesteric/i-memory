from agile.utils import singleton

from infra.ai.embedding.providers.openai_embed import OpenAIEmbed
from shared.config.settings import env


@singleton
class DashScopeEmbed(OpenAIEmbed):

    def __init__(self, api_key: str = None, dim: int = 1536):
        super().__init__(
            api_key=api_key or env.DASHSCOPE_API_KEY,
            base_url=env.DASHSCOPE_BASE_URL,
            model=env.DASHSCOPE_EMBEDDING_MODEL,
            dim=dim
        )
