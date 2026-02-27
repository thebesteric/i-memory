from typing import Optional, Dict, List, Any

from pydantic import BaseModel, Field

from src.memory.models.memory_models import IMemoryFilters, IMemoryUserIdentity


class AddMemoryRequest(BaseModel):
    """
    添加记忆请求模型
    """
    content: str = Field(..., description="记忆内容文本")
    user_identity: Optional[IMemoryUserIdentity] = Field(default=None, description="用户身份")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签列表")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="其他元数据")


class SearchMemoryRequest(BaseModel):
    """
    搜索记忆请求模型
    """
    query: str = Field(..., description="搜索查询文本")
    limit: Optional[int] = Field(default=10, ge=1, le=100, description="至少要返回的结果数量")
    filters: Optional[IMemoryFilters] = Field(default=None, description="搜索过滤条件")


class HistoryMemoryRequest(BaseModel):
    """
    历史记忆请求模型
    """
    user_identity: IMemoryUserIdentity = Field(..., description="用户身份")
    current: Optional[int] = Field(default=1, ge=1, description="当前页码")
    size: Optional[int] = Field(default=10, ge=1, le=100, description="每页记录数")
