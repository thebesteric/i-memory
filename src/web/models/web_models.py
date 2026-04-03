from typing import Optional, Dict, List, Any

from pydantic import BaseModel, Field

from src.memory.memory_models import IMemoryFilters, IMemoryUserIdentity, QARole


class AddMemoryRequest(BaseModel):
    """
    添加记忆请求模型
    """
    content: str = Field(..., description="记忆内容文本")
    user_identity: Optional[IMemoryUserIdentity] = Field(default=None, description="用户身份")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签列表")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="其他元数据")
    qa_role: Optional[QARole] = Field(default=None, description="QA 角色，仅允许 human/assistant")


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


class CanonicalRelationsRequest(BaseModel):
    """
    查询 canonical 实体关系请求模型
    """
    user_identity: IMemoryUserIdentity = Field(..., description="用户身份")
    canonical_id: str = Field(..., min_length=1, description="规范化实体 ID")
    limit: Optional[int] = Field(default=100, ge=1, le=999, description="最多返回关系数量")

