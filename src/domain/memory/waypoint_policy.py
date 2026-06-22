from typing import List

from pydantic import BaseModel, Field


class Expansion(BaseModel):
    id: str = Field(..., description="记忆 ID")
    weight: float = Field(..., description="扩展权重")
    path: List[str] = Field(default_factory=list, description="扩展路径")