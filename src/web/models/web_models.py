from typing import Optional, Dict, List, Any, Literal

from pydantic import BaseModel, Field

from src.memory.memory_models import IMemoryFilters, IMemoryUserIdentity, QARole


class AuthRegisterRequest(BaseModel):
    """
    用户注册请求模型
    """
    tenant_key: str = Field(default=..., description="租户标识")
    project_key: str = Field(default=..., description="项目标识")
    user_key: str = Field(default=..., description="用户标识")

    def to_identity_model(self) -> IMemoryUserIdentity:
        return IMemoryUserIdentity(
            tenant_key=self.tenant_key,
            project_key=self.project_key,
            user_key=self.user_key,
        )


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
    sort_order: Optional[Literal["asc", "desc"]] = Field(default="desc", description="排序顺序：asc-正序，desc-倒序")


class CanonicalRelationsRequest(BaseModel):
    """
    查询 canonical 实体关系请求模型
    """
    user_identity: IMemoryUserIdentity = Field(..., description="用户身份")
    canonical_id: str = Field(..., min_length=1, description="规范化实体 ID")
    limit: Optional[int] = Field(default=100, ge=1, le=999, description="最多返回关系数量")


class GraphPagingBaseRequest(BaseModel):
    user_identity: IMemoryUserIdentity = Field(..., description="用户身份")
    current: Optional[int] = Field(default=1, ge=1, description="当前页码")
    size: Optional[int] = Field(default=20, ge=1, le=100, description="每页记录数")


class GraphFactsFilters(BaseModel):
    topic_id: Optional[str] = Field(default=None, description="按主题 ID 过滤")
    fact_kind: Optional[str] = Field(default=None, description="按事实类型过滤（conversation/event）")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最小置信度")
    max_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最大置信度")
    keyword: Optional[str] = Field(default=None, description="在 what/who/where/why 字段中模糊匹配")


class GraphFactsRequest(GraphPagingBaseRequest):
    filters: Optional[GraphFactsFilters] = Field(default=None, description="事实筛选条件")


class GraphFactEntitiesRequest(GraphPagingBaseRequest):
    fact_id: str = Field(..., min_length=1, description="事实 ID")


class GraphEntityRelationFilters(BaseModel):
    fact_id: Optional[str] = Field(default=None, description="按证据 fact_id 过滤")
    edge_relations: Optional[List[str]] = Field(default=None, description="按边关系过滤")
    infer_sources: Optional[List[str]] = Field(default=None, description="按推断来源过滤")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最小置信度")
    max_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最大置信度")
    related_canonical_id: Optional[str] = Field(default=None, description="按关联实体 ID 过滤")


class GraphEntityRelationsRequest(GraphPagingBaseRequest):
    canonical_id: str = Field(..., min_length=1, description="规范化实体 ID")
    filters: Optional[GraphEntityRelationFilters] = Field(default=None, description="实体关系筛选条件")


class GraphEntityTopicsRequest(GraphPagingBaseRequest):
    canonical_id: str = Field(..., min_length=1, description="规范化实体 ID")


class GraphTopicMemoriesRequest(GraphPagingBaseRequest):
    topic_id: str = Field(..., min_length=1, description="主题 ID")


class GraphExploreRequest(BaseModel):
    user_identity: IMemoryUserIdentity = Field(..., description="用户身份")
    seed_type: Literal["canonical", "fact", "topic"] = Field(..., description="探索起点类型")
    seed_id: str = Field(..., min_length=1, description="探索起点 ID")
    current: Optional[int] = Field(default=1, ge=1, description="当前页码")
    size: Optional[int] = Field(default=20, ge=1, le=100, description="起点分页大小")
    relation_size: Optional[int] = Field(default=20, ge=1, le=100, description="每个实体关系扩展上限")
    entity_size: Optional[int] = Field(default=20, ge=1, le=100, description="每个事实实体扩展上限")
    include_relations_on_topic_entities: Optional[bool] = Field(
        default=False,
        description="topic 起点下是否继续扩展实体关系"
    )
    max_nodes: Optional[int] = Field(default=None, ge=1, le=5000, description="返回节点数上限")
    max_edges: Optional[int] = Field(default=None, ge=1, le=10000, description="返回边数上限")
