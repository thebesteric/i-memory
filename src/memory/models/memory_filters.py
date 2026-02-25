from typing import TypedDict

from pydantic import BaseModel, Field, ConfigDict


class IMemoryFilters(BaseModel):
    """
    记忆查询过滤器
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    user_id: str = Field(default="anonymous", description="用户 ID")
    sectors: list[str] = Field(default_factory=list, description="检索扇区范围")
    min_salience: float = Field(default=0.0, description="最小显著性过滤值")
    debug: bool = Field(default=False, description="是否启用调试模式")
