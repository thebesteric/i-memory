from pydantic import BaseModel, Field, ConfigDict, PrivateAttr


class Session(BaseModel):
    """
    围绕着某一件事请进行讨论
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    summary: str = Field(..., description="会话摘要，尽量准确、详实，包含关键信息，能够完整的表达出整个会话的核心内容和事件经过")
    dialogue_ids: list[str] = Field(..., description="原始对话 ID 列表，表示这个会话单元包含了哪些原始对话 ID")
    key_facts: list[str] = Field(
        default_factory=list,
        description="从对话中提取出的关键信息点，应该是对话中明确提及的事实，尽量具体、清晰。包含提取用户透露的个人信息、偏好、行为习惯、状态等"
    )

    _id: str | None = PrivateAttr(None)
    _user_id: str | None = PrivateAttr(None)
    _vector: list[float] = PrivateAttr(None)
    _similarity: float = PrivateAttr(None)

    @property
    def id(self):
        return self.model_extra.get("_id", None)

    @property
    def user_id(self):
        return self.model_extra.get("_user_id", None)

    @property
    def vector(self):
        return self.model_extra.get("_vector", None)

    @property
    def similarity(self):
        return self.model_extra.get("_similarity", None)

    def set_id(self, _id: str):
        self._id = _id
        self.model_extra["_id"] = _id

    def set_user_id(self, user_id: str):
        self._user_id = user_id
        self.model_extra["_user_id"] = user_id

    def set_vector(self, vector: list[float]):
        self._vector = vector
        self.model_extra["_vector"] = vector

    def set_similarity(self, similarity: float):
        self._similarity = similarity
        self.model_extra["_similarity"] = similarity


class Sessions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    sessions: list[Session] = Field(default_factory=list, description="会话列表")
