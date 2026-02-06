import math
import sys
from typing import List, Any

from pydantic import BaseModel, Field, computed_field


class PagingRequest(BaseModel):
    """
    分页请求模型
    """

    current: int = Field(default=1, description="当前页码")
    size: int = Field(default=10, description="每页记录数")

    def get_offset(self) -> int:
        """计算偏移量"""
        return (self.current - 1) * self.size

    def unlimited(self):
        """设置为不分页模式"""
        self.current = 1
        self.size = sys.maxsize


class PagingResponse(BaseModel):
    """
    分页响应模型
    """

    records: List[Any] = Field(default_factory=list, description="记录列表")
    total: int = Field(default=0, description="总记录数")
    current: int = Field(default=0, description="当前页码")
    size: int = Field(default=10, description="每页记录数")
    extension: dict[str, Any] = Field(default_factory=dict, description="扩展字段")

    @computed_field
    @property
    def pages(self) -> int:
        """总页数"""
        if self.size == 0:
            return 0
        return math.ceil(self.total / self.size)

    def has_previous(self) -> bool:
        """是否有上一页"""
        return self.current > 1

    def has_next(self) -> bool:
        """是否有下一页"""
        return self.current < self.pages
