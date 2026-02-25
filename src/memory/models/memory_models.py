from pydantic import BaseModel, ConfigDict, Field


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

    user_id: str = Field(default="anonymous", description="用户 ID")
    sectors: list[str] = Field(default_factory=list, description="检索扇区范围")
    min_salience: float = Field(default=0.0, description="最小显著性过滤值")
    debug: bool = Field(default=False, description="是否启用调试模式")
