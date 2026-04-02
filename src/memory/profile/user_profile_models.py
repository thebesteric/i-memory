from datetime import datetime
from typing import Literal, Any, cast
import json

from agile.commons.enum import LabeledStrEnum
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr


class Personality(LabeledStrEnum):
    EXTROVERT = ("extrovert", "外向")
    INTROVERT = ("introvert", "内向")
    AMBITIOUS = ("ambitious", "有野心")
    CALM = ("calm", "冷静")
    CHEERFUL = ("cheerful", "开朗")
    CONFIDENT = ("confident", "自信")
    CREATIVE = ("creative", "有创造力")
    EMPATHETIC = ("empathetic", "有同理心")
    PATIENT = ("patient", "耐心")
    RESPONSIBLE = ("responsible", "有责任心")
    SHY = ("shy", "腼腆/害羞")
    SOCIABLE = ("sociable", "善于社交")
    LOGICAL = ("logical", "理性")
    EMOTIONAL = ("emotional", "感性")
    THOUGHTFUL = ("thoughtful", "温柔/体贴")
    OTHER = ("other", "其他")


class Location(BaseModel):
    """
    位置信息
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    country: str | None = Field(default=None, description="国家")
    hometown: str | None = Field(default=None, description="家乡")
    city: str | None = Field(default=None, description="城市")
    address: str | None = Field(default=None, description="详细地址")


class Demographic(BaseModel):
    """
    人口统计/基础属性
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    age_range: str | None = Field(default=None, description="年龄范围：如：20-25")
    gender: Literal["male", "female", "unknown"] | None = Field(default="unknown", description="性别")
    occupation: str | None = Field(default=None, description="职业")
    education: str | None = Field(default=None, description="教育水平")
    location: Location = Field(default_factory=Location, description="位置信息")
    personality: list[Personality] = Field(default_factory=list, description="性格特征")
    extra: dict[str, Any] = Field(default_factory=dict, description="其他人口统计信息，可自由补充")


class Habit(BaseModel):
    """
    行为习惯
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(..., description="习惯名称")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="习惯置信度")
    description: str | None = Field(default=None, description="习惯描述")
    evidences: dict[str, str] = Field(default_factory=list,
                                      description="习惯形成的证据列表，key 是对话 ID，value 是提取理由")
    created_at: datetime = Field(default_factory=datetime.now, description="首次创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最近一次更新时间")


class Setting(BaseModel):
    """
    相关设置（用户主动设置）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(..., description="偏好名称")
    key: str = Field(..., description="偏好键")
    value: Any = Field(..., description="偏好值")
    description: str | None = Field(default=None, description="偏好描述")


class Preferences(BaseModel):
    """
    用户偏好
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    habits: list[Habit] = Field(default_factory=list, description="习惯列表（从行为推断）")
    extra: dict[str, Any] = Field(default_factory=dict, description="其他偏好设置，可自由补充")

    _settings: list[Setting] = PrivateAttr(list)

    @property
    def settings(self) -> list[Setting]:
        return self.model_extra.get("_settings", [])

    def add_setting(self, setting: Setting):
        self._settings.append(setting)
        self.model_extra["_settings"].append(setting)


class Attributes(BaseModel):
    """
    其他属性
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    extra: dict[str, Any] = Field(default_factory=dict, description="其他补充设置，可自由补充")


class Tag(BaseModel):
    """
    标签
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(...,
                      description="标签名称（标签名称一定有代表性，能够准确的描述用户某些特征，如：科技控、时尚达人、万事通）")
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="标签权重")
    sub_tags: list[str] = Field(default_factory=list, description="子标签列表")
    created_at: datetime = Field(default_factory=datetime.now, description="首次提及时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最近一次提及时间")
    source: Literal["explicit", "implicit"] = Field(default="explicit",
                                                    description="标签来源：explicit（显式）或 implicit（隐式）")
    evidences: dict[str, str] = Field(default_factory=list,
                                      description="标签形成的证据列表，key 是对话 ID，value 是提取理由")


class UserProfile(BaseModel):
    """
    用户画像
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    demographic: Demographic = Field(default_factory=Demographic, description="个性信息")
    preferences: Preferences = Field(default_factory=Preferences, description="兴趣偏好")
    attributes: Attributes = Field(default_factory=Attributes, description="其他属性")
    tags: list[Tag] = Field(default_factory=list, description="标签信息")

    _id: str | None = PrivateAttr(None)
    _user_id: str | None = PrivateAttr(None)
    _is_active: bool = PrivateAttr(True)

    @property
    def id(self):
        return self.model_extra.get("_id", None)

    @property
    def user_id(self):
        return self.model_extra.get("_user_id", None)

    @property
    def is_active(self):
        return self.model_extra.get("_is_active", True)

    def set_id(self, _id: str):
        self._id = _id
        self.model_extra["_id"] = _id

    def set_user_id(self, user_id: str):
        self._user_id = user_id
        self.model_extra["_user_id"] = user_id

    def set_is_active(self, is_active: bool):
        self._is_active = is_active
        self.model_extra["_is_active"] = is_active

    @staticmethod
    def from_dict(row: dict[str, Any]):
        if not row:
            return None

        def parse_json_field(val, default):
            if val is None:
                return default
            if isinstance(val, (dict, list)):
                return val
            try:
                return json.loads(val)
            except Exception:
                return default

        # 解析 JSON 字段
        demographic_data = parse_json_field(row.get("demographic"), {})
        attributes_data = parse_json_field(row.get("attributes"), {})
        preferences_data = parse_json_field(row.get("preferences"), {})
        tags_data = parse_json_field(row.get("tags"), [])

        # 从个性列表中过滤掉无效/空值
        if "personality" in demographic_data and isinstance(demographic_data["personality"], list):
            valid_personality_names = {e.name for e in Personality}
            demographic_data["personality"] = [
                p for p in demographic_data["personality"]
                if isinstance(p, str) and p and p in valid_personality_names
            ]
        demographic = Demographic(**cast(dict[str, Any], demographic_data)) if demographic_data else Demographic()
        attributes = Attributes(**cast(dict[str, Any], attributes_data)) if attributes_data else Attributes()
        preferences = Preferences(**cast(dict[str, Any], preferences_data)) if preferences_data else Preferences()
        tags = [Tag(**tag) for tag in tags_data]

        # 创建用户画像
        user_profile = UserProfile(
            demographic=demographic,
            attributes=attributes,
            preferences=preferences,
            tags=tags
        )

        # 设置私有字段
        user_id = row.get("user_id")
        if user_id:
            user_profile.set_user_id(str(user_id))
        is_active = row.get("is_active")
        if is_active is not None:
            user_profile.set_is_active(bool(is_active))

        return user_profile
