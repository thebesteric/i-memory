import datetime
from typing import Any, Literal

from agile.utils.pydantic_extension import BaseModelEnhance
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

QARole = Literal["human", "assistant"]
QueryMode = Literal["vector", "qa", "prefer"]


class IMemoryUserIdentity(BaseModel):
    """
    用户身份信息
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    tenant_key: str | None = Field(default=None, description="租户标识")
    project_key: str | None = Field(default=None, description="项目标识")
    user_key: str = Field(default="anonymous", description="用户标识")

    _id: str | None = PrivateAttr(default=None)

    @staticmethod
    def from_dict(user: "IMemoryUser") -> "IMemoryUserIdentity":
        return IMemoryUserIdentity(
            _id=user.id,
            tenant_key=user.tenant_key,
            project_key=user.project_key,
            user_key=user.user_key,
        )

    @property
    def id(self):
        return self.model_extra.get("_id", None)

    def check_legality(self):
        """
        检查用户身份是否合法
        :return: True 如果合法，否则 False
        """
        if not self.tenant_key or not self.project_key or not self.user_key:
            raise ValueError("User tenant_key and project_key and user_key cannot be empty")


class IMemoryConfig(BaseModel):
    """
    记忆配置类
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    force_root: bool = Field(default=False, description="是否强制使用 root-child 结构存储")
    large_token_thresh: int = Field(default=200, description="长语句 token 的阈值，超过该值将使用 root-child 结构存储")
    section_length: int = Field(default=150, description="文本分段大小阈值，用于长文本分割")
    summary_length: int = Field(default=200, description="摘要长度，用于长文本摘要生成")

    @staticmethod
    def create_default():
        return IMemoryConfig()


class IMemoryFiltersConfig(BaseModel):
    """
    记忆查询过滤器相关配置
    """
    bm25_enable: bool = Field(default=True, description="是否启用 BM25 关键词检索")
    graph_enable: bool = Field(default=True, description="是否启用图检索")
    debug: bool = Field(default=False, description="是否启用调试模式")


class IMemoryFilters(BaseModel):
    """
    记忆查询过滤器
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    user_identity: IMemoryUserIdentity = Field(..., description="用户身份信息")
    sectors: list[str] = Field(default_factory=list, description="检索扇区范围")
    min_salience: float = Field(default=0.0, description="最小显著性过滤值")
    query_mode: QueryMode = Field(default="prefer", description="查询模式：vector/qa/prefer")
    config: IMemoryFiltersConfig = Field(default_factory=IMemoryFiltersConfig, description="相关查询配置")


class IMemoryItemDebugInfo(BaseModel):
    """
    调试信息模型
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    sim_adjust: float = Field(default=0.0, description="相似度调整值")
    token_overlap: float = Field(default=0.0, description="关键词重叠")
    recency_score: float = Field(default=0.0, description="时效性得分")
    waypoint_weight: float = Field(default=0.0, description="路标点权重")
    tag_match_score: float = Field(default=0.0, description="标签匹配得分")
    penalty_score: float = Field(default=0.0, description="惩罚得分")


class IMemoryItemInfo(BaseModelEnhance):
    """
    记忆项模型
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: Any = Field(..., description="记忆 ID")
    content: str | Any = Field(..., description="记忆内容")
    score: float = Field(..., description="综合得分")
    primary_sector: str = Field(..., description="主扇区")
    path: list[str] = Field(default_factory=list, description="记忆路径")
    salience: float = Field(default=0.0, description="显著性分数")
    last_seen_at: datetime.datetime | None = Field(default=None, description="最近一次访问时间")
    created_at: datetime.datetime | None = Field(default=None, description="创建时间")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    qa_role: QARole | None = Field(default=None, description="QA 角色")
    qa_pair_id: str | None = Field(default=None, exclude=True, description="内部问答对 ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    debug: IMemoryItemDebugInfo | None = Field(default=None, description="调试信息")


class IMemoryUser(BaseModel):
    """
    记忆用户模型
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(..., description="主键")
    tenant_key: str | None = Field(default=None, description="租户标识")
    project_key: str | None = Field(default=None, description="项目标识")
    user_key: str | None = Field(default=None, description="用户标识")
    summary: str | None = Field(default=None, description="用户摘要")
    reflection_count: int = Field(default=0, description="反思次数")
    created_at: datetime.datetime | None = Field(default=None, description="创建时间")
    updated_at: datetime.datetime | None = Field(default=None, description="更新时间")

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return IMemoryUser(
            id=data["id"],
            tenant_key=data["tenant_key"],
            project_key=data["project_key"],
            user_key=data["user_key"],
            summary=data["summary"],
            reflection_count=data["reflection_count"],
            created_at=data["created_at"],
            updated_at=data["updated_at"]
        ) if data else None
