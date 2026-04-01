from pydantic import BaseModel, Field, ConfigDict


class Session(BaseModel):
    """
    围绕着某一件事请进行讨论
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    summary: str = Field(..., description="会话摘要，尽量准确、详实，包含关键信息")
    dialogue_ids: list[str] = Field(..., description="原始对话 ID 列表，表示这个会话单元包含了哪些原始对话 ID")
    key_facts: list[str] = Field(
        default_factory=list,
        description="从对话中提取出的关键信息点，应该是对话中明确提及的事实，尽量具体、清晰。包含提取用户透露的个人信息、偏好\行为习惯、状态等"
    )


class Sessions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    sessions: list[Session] = Field(default_factory=list, description="会话列表")
