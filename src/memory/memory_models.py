import datetime
from typing import Any, Literal

from agile.utils.pydantic_extension import BaseModelEnhance
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.memory.profile.user_profile_models import UserProfile

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
            user_key=user.user_key or "anonymous",
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


class IMemoryGraphConfig(BaseModel):
    """图检索参数配置。"""

    enable: bool = Field(default=True, description="是否启用图检索")
    type: Literal["recall", "precision", "custom"] = Field(default="precision", description="图检索类型")
    max_hops: int = Field(default=1, ge=1, le=4, description="图检索实体游走最大跳数")
    hop_decay: float = Field(default=0.8, ge=0.1, le=1.0, description="图检索每跳分数衰减系数")
    per_hop_limit: int = Field(default=200, ge=10, le=2000, description="图检索每跳保留候选上限")
    min_walk_score: float = Field(default=0.05, ge=0.0, le=1.0, description="图检索游走最小分数阈值")
    min_relation_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="图关系最小置信度阈值")

    @staticmethod
    def recall_first() -> "IMemoryGraphConfig":
        """
        召回优先预设：放宽阈值并提高扩展规模
        """
        return IMemoryGraphConfig(
            enable=True,
            type="recall",
            max_hops=2,
            hop_decay=0.9,
            per_hop_limit=400,
            min_walk_score=0.02,
            min_relation_confidence=0.4,
        )

    @staticmethod
    def precision_first() -> "IMemoryGraphConfig":
        """
        精度优先预设：收紧阈值并减少扩展规模
        """
        return IMemoryGraphConfig(
            enable=True,
            type="precision",
            max_hops=1,
            hop_decay=0.7,
            per_hop_limit=120,
            min_walk_score=0.1,
            min_relation_confidence=0.65,
        )


class IMemoryFiltersConfig(BaseModel):
    """
    记忆查询过滤器相关配置
    """
    bm25_enable: bool = Field(default=True, description="是否启用 BM25 关键词检索")
    user_profile_enable: bool = Field(default=False, description="是否返回用户画像")
    session_summary_enable: bool = Field(default=True, description="是否启用会话摘要")
    session_dedup_enable: bool = Field(default=False, description="是否启用会话摘要")
    graph: IMemoryGraphConfig = Field(default_factory=IMemoryGraphConfig.precision_first, description="图检索配置")
    debug: bool = Field(default=False, description="是否启用调试模式")

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_graph_fields(cls, data: Any):
        if not isinstance(data, dict):
            return data

        graph_data = dict(data.get("graph") or {})

        if "graph_enable" in data and "enable" not in graph_data:
            graph_data["enable"] = data["graph_enable"]

        legacy_to_new = {
            "graph_max_hops": "max_hops",
            "graph_hop_decay": "hop_decay",
            "graph_per_hop_limit": "per_hop_limit",
            "graph_min_walk_score": "min_walk_score",
            "graph_min_relation_confidence": "min_relation_confidence",
        }
        for legacy_key, graph_key in legacy_to_new.items():
            if legacy_key in data and graph_key not in graph_data:
                graph_data[graph_key] = data[legacy_key]

        if graph_data:
            data = dict(data)
            data["graph"] = graph_data

        return data


class IMemoryFilters(BaseModel):
    """
    记忆查询过滤器
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    user_identity: IMemoryUserIdentity = Field(..., description="用户身份信息")
    sectors: list[str] = Field(default_factory=list, description="检索扇区范围")
    min_salience: float = Field(default=0.0, ge=0.0, le=1.0, description="最小显著性过滤值")
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


class IMemorySearchResult(BaseModelEnhance):
    """
    记忆查询返回结果
    """

    user_profile: UserProfile | None = Field(default=None, description="用户画像信息")
    memories: list[IMemoryItemInfo] = Field(default_factory=list, description="记忆项列表")


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
