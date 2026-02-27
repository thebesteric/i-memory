import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from utils.pydantic_extension import BaseModelEnhance


class IMemoryUserIdentity(BaseModel):
    """
    用户身份信息
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    tenant_id: str | None = Field(default=None, description="租户 ID")
    project_id: str | None = Field(default=None, description="项目 ID")
    user_id: str = Field(default="anonymous", description="用户 ID")


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


class IMemoryFilters(BaseModel):
    """
    记忆查询过滤器
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    user_identity: IMemoryUserIdentity = Field(..., description="用户身份信息")
    sectors: list[str] = Field(default_factory=list, description="检索扇区范围")
    min_salience: float = Field(default=0.0, description="最小显著性过滤值")
    debug: bool = Field(default=False, description="是否启用调试模式")


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
    tags: list[str] = Field(default_factory=list, description="标签列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    debug: IMemoryItemDebugInfo | None = Field(default=None, description="调试信息")


class IMemoryUser(BaseModel):
    """
    记忆用户模型
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(..., description="主键")
    tenant_id: str | None = Field(default=None, description="租户 ID")
    project_id: str | None = Field(default=None, description="项目 ID")
    user_id: str | None = Field(default=None, description="用户 ID")
    summary: str | None = Field(default=None, description="用户摘要")
    reflection_count: int = Field(default=0, description="反思次数")
    created_at: datetime.datetime | None = Field(default=None, description="创建时间")
    updated_at: datetime.datetime | None = Field(default=None, description="更新时间")
