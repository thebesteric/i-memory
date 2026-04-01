import json
from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Any

from agile.utils import LogHelper
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

logger = LogHelper.get_logger()


class RelationType(str, Enum):
    """
    人与人之间关系类型枚举
    """
    # 家人亲属
    SELF = ("self", "自己")
    FAMILY = ("family", "家人")
    PARENT = ("parent", "父母")
    CHILD = ("child", "子女")
    SPOUSE = ("spouse", "配偶")
    SIBLING = ("sibling", "兄弟姐妹")
    RELATIVE = ("relative", "亲戚")

    # 朋友同学
    FRIEND = ("friend", "朋友")
    BEST_FRIEND = ("best_friend", "挚友")
    CLASSMATE = ("classmate", "同学")

    # 工作职场
    COLLEAGUE = ("colleague", "同事")
    LEADER = ("leader", "上级/领导")
    SUBORDINATE = ("subordinate", "下属")
    PARTNER = ("partner", "合作伙伴")

    # 情感亲密
    LOVER = ("lover", "恋人")
    FIANCE = ("fiance", "未婚夫/妻")

    # 社交其他
    NEIGHBOR = ("neighbor", "邻居")
    ACQUAINTANCE = ("acquaintance", "熟人")
    STRANGER = ("stranger", "陌生人")
    OTHER = ("other", "其他")

    def __new__(cls, value: str, label: str):
        """支持元组形式的枚举值"""
        obj = str.__new__(cls)
        obj._value_ = value
        obj._label = label
        return obj

    @property
    def label(self) -> str:
        """
        获取中文标签
        :return:
        """
        return self._label

    def display(self) -> str:
        """获取显示文本"""
        return f"{self.value} ({self.label})"

    @classmethod
    def get_all_labels(cls) -> list[tuple[str, str]]:
        """
        获取所有类型的 value 和 label 列表
        :return:
        """
        return [(e.value, e.label) for e in cls]

    @classmethod
    def get_prompt_description(cls) -> str:
        """生成用于提示词的描述"""
        lines = []
        for e in cls:
            lines.append(f"  - {e.value}：{e.label}")
        return "\n".join(lines)

    @classmethod
    def from_value(cls, value: str) -> Optional["RelationType"]:
        """
        根据 value 获取枚举成员
        :param value:
        :return:
        """
        for e in cls:
            if e.value == value:
                return e
        return None

    @classmethod
    def from_name(cls, name: str) -> Optional["RelationType"]:
        """
        根据 name 获取枚举成员
        :param name:
        :return:
        """
        for e in cls:
            if e.name == name:
                return e
        return None


class EntityType(str, Enum):
    """
    实体类型枚举
    """
    # 核心实体
    PERSON = ("person", "人物/个体")
    ORGANIZATION = ("organization", "组织/机构/公司/团体")
    LOCATION = ("location", "地理位置/区域")

    # 事物实体
    PRODUCT = ("product", "产品/商品/物品")
    SYSTEM = ("system", "系统/平台/软件/APP")
    DEVICE = ("device", "设备/硬件/工具")
    DOCUMENT = ("document", "文档/文件/资料")

    # 概念/抽象实体
    CONCEPT = ("concept", "抽象概念/术语/主题")
    ROLE = ("role", "职位/角色/身份")
    EVENT = ("event", "事件实体/历史事件")
    CAPABILITY = ("capability", "能力/功能")
    ATTRIBUTE = ("attribute", "属性/特征")

    # 规则/标准实体
    RULE = ("rule", "规则/标准/条款/法律")

    # 语言/技术实体
    LANGUAGE = ("language", "语言/协议/技术框架")

    # 其他无法归类
    OTHER = ("other", "其他无法归类的实体类型")

    def __new__(cls, value: str, label: str):
        """支持元组形式的枚举值"""
        obj = str.__new__(cls)
        obj._value_ = value
        obj._label = label
        return obj

    @property
    def label(self) -> str:
        """
        获取中文标签
        :return:
        """
        return self._label

    def display(self) -> str:
        """获取显示文本"""
        return f"{self.value} ({self.label})"

    @classmethod
    def get_all_labels(cls) -> list[tuple[str, str]]:
        """
        获取所有类型的 value 和 label 列表
        :return:
        """
        return [(e.value, e.label) for e in cls]

    @classmethod
    def get_prompt_description(cls) -> str:
        """生成用于提示词的描述"""
        lines = []
        for e in cls:
            lines.append(f"  - {e.value}：{e.label}")
        return "\n".join(lines)

    @classmethod
    def from_value(cls, value: str) -> Optional["EntityType"]:
        """
        根据 value 获取枚举成员
        :param value:
        :return:
        """
        for e in cls:
            if e.value == value:
                return e
        return None

    @classmethod
    def from_name(cls, name: str) -> Optional["EntityType"]:
        """
        根据 name 获取枚举成员
        :param name:
        :return:
        """
        for e in cls:
            if e.name == name:
                return e
        return None


class Entity(BaseModel):
    """
    从文本中提取的实体
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    text: str = Field(..., description="事实中出现的具体命名实体（命名尽可能规范）")
    entity_type: EntityType = Field(..., description="实体类型")
    relation_to_user: RelationType | None = Field(default=None, description="与用户的关系")

    _id: str | None = PrivateAttr(None)
    _user_id: str | None = PrivateAttr(None)
    _canonical_id: str | None = PrivateAttr(None)

    @classmethod
    @field_validator('entity_type', mode='before')
    def validate_entity_type(cls, v):
        try:
            e_type = EntityType.from_value(str(v))
            return e_type if e_type is not None else EntityType.OTHER
        except Exception as e:
            logger.warning(e)
            return EntityType.OTHER

    @classmethod
    @field_validator('relation_to_user', mode='before')
    def validate_relation_to_user(cls, v):
        try:
            r_type = RelationType.from_value(str(v))
            return r_type if r_type is not None else RelationType.OTHER
        except Exception as e:
            logger.warning(e)
            return RelationType.OTHER

    @property
    def id(self):
        return self.model_extra.get("_id", None)

    @property
    def user_id(self):
        return self.model_extra.get("_user_id", None)

    @property
    def canonical_id(self):
        return self.model_extra.get("_canonical_id", None)

    def set_id(self, id_value: str):
        self._id = id_value
        self.model_extra["_id"] = id_value

    def set_user_id(self, user_id: str):
        self._user_id = user_id
        self.model_extra["_user_id"] = user_id

    def set_canonical_id(self, canonical_id_value: str):
        self._canonical_id = canonical_id_value
        self.model_extra["_canonical_id"] = canonical_id_value


class CanonicalEntity(BaseModel):
    """
    规范化实体
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(..., description="主键")
    name: str = Field(..., description="实体名称")
    entity_type: EntityType = Field(..., description="实体类型")
    vector: list[float] = Field(..., description="实体的向量表示")
    occurrence_count: int = Field(default=1, description="出现次数")
    first_seen_at: datetime = Field(..., description="首次出现时间")
    last_seen_at: datetime = Field(..., description="最近出现时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "CanonicalEntity":
        return CanonicalEntity(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType.from_name(row["entity_type"]),
            vector=row["vector"] if isinstance(row["vector"], list) else json.loads(row["vector"]),
            occurrence_count=row["occurrence_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class Topic(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(..., description="主题")
    summary: str = Field(..., description="内容简介（精简，紧凑，能够表达核心含义）")
    keywords: list[str] = Field(default_factory=list, description="关键词/信息，尽可能细分")
    dialogue_ids: list[str] = Field(..., description="原始对话 ID 列表，表示这个知识单元包含了哪些原始对话 ID")

    _id: str | None = PrivateAttr(None)

    # 私有字段，用于后面将 summary 进行向量嵌入
    _vector: list[float] | None = PrivateAttr(default=None)

    @property
    def id(self):
        return self.model_extra.get("_id", None)

    @property
    def vector(self):
        return self.model_extra.get("_vector", None)

    def set_id(self, id_value: str):
        self._id = id_value
        self.model_extra["_id"] = id_value

    def set_vector(self, vector_value: list[float]):
        self._vector = vector_value
        self.model_extra["_vector"] = vector_value


class Fact(BaseModel):
    """
    事实抽取模型
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    what: str = Field(
        ...,
        description="发生了什么 - 完整、详尽的描述，包含所有具体细节（原始事实、基于对话内容）。"
                    "切勿概括或省略细节，需涵盖：具体行为、对象、数量、详细信息。"
                    "务必详尽：记录提及的每一个细节。"
                    "示例：“艾米莉与萨拉在屋顶花园举行婚礼，有 50 位宾客出席，并有一支爵士乐队现场演奏”，而非：“举办了一场婚礼” 或 “艾米莉结婚了”。"
    )

    when: str | None = Field(
        default=None,
        description="事件发生时间 - 若文本中提及，务必包含时间相关信息。需包含：具体日期、具体时刻、持续时长、相对时间表述。"
                    "示例：“2024 年 6 月 15 日下午 3 点”，“上周末”，“过去三年间”，“每天早上 6 点”。"
                    "仅在完全无任何时间相关语境时，设置为空，尽可能转换为绝对日期表述。"
    )

    where: str | None = Field(
        default=None,
        description="事件发生或相关的地点 - 若适用，需填写具体地点、场所、区域、地区。"
                    "包含：提及的城市、社区、举办场地、建筑、国家、具体地址。"
                    "示例：“合肥市中心的一处屋顶花园场地”，“用户位于北京的家中”，“通过 Zoom 线上进行”，“安徽合肥”。"
                    "仅在完全不存在任何地点相关语境，或该事实与地点完全无关时，填写空。"
    )

    who: str | None = Field(
        default=None,
        description="涉及人员/实体 - 所有掌握完整背景信息及关联关系的人员/实体。"
                    "需包含：姓名、身份角色、与用户的关系、背景详情。"
                    "明确指代关系（若前文提及 “我的室友”，后文明确其名为 “艾米莉”，则应表述为 “艾米莉，用户的大学室友”）。"
                    "详细说明人物关系与身份角色。"
                    "示例：“艾米莉（用户在斯坦福大学的大学室友，现就职于谷歌），萨拉（艾米莉交往 5 年的伴侣，软件工程师）”"
                    "禁止表述为：“我的朋友” 或 “艾米莉和萨拉”"
    )

    why: str | None = Field(
        default=None,
        description="重要性/动机/语境 - 所有情感、语境及动机层面的细节。涵盖全部内容：情绪感受、个人偏好、内在动机、观察所得、背景语境、深层意义。"
                    "表述需详尽细致，捕捉所有细微差别与内涵。针对助手相关事实：必须包含用户此次交互中提出的问题或需求！"
                    "场景示例：客观事实 - “用户感到兴奋且备受鼓舞，一直梦想举办一场户外仪式，提及想要一处风格相似的花园场地，尤其被温馨私密的氛围和个性化誓词所打动”"
                    "场景示例：助手交互 - “用户询问如何解决千级以上并发用户下的 API 性能缓慢问题，期望将数据库负载降低 70% 至 80%”"
                    "禁止表述为：“用户很喜欢” 或 “为了帮助用户”"
    )

    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="置信度：0.0~1.0 之间的浮点数，表示事实的准确程度或重要性，0 表示完全不确定，1 表示完全确定"
    )

    fact_kind: Literal["conversation", "event"] = Field(
        default="conversation",
        description="事实类型 - event：可确定具体日期的事件（已设定发生日期），conversation：常规信息（无发生日期）",
    )

    occurred_start: str | None = Field(
        default=None,
        description="事件发生的时间（ISO 时间戳）- 仅当 fact_kind 为 'event' 时使用；对话类型数据请置空。",
    )

    occurred_end: str | None = Field(
        default=None,
        description="事件结束时间（ISO 时间戳）- 仅当 fact_kind 为 'event' 时，且有持续时长的事件，对话类型数据请置空。",
    )

    entities: list[Entity] = Field(
        default_factory=list,
        description="从事实中提取的命名实体、具体事物以及抽象概念。"
                    "包括：人物姓名、组织机构、地点、重要物品（如 “咖啡机”，“汽车”），"
                    "以及抽象概念/主题（如 “友谊”，“职业发展”，“庆典”）。"
                    "提取所有有助于关联相关事实的内容。注意：提取的实体名称要规范化",
    )

    # 私有字段：主键
    _id: str | None = PrivateAttr(None)

    # 私有字段：用于设置语义，列表的第一个为主语义，其他为辅助语义
    _sectors: list[Literal["episodic", "semantic", "procedural", "emotional", "reflective"]] = PrivateAttr(...)

    # 私有字段：事实的语义向量（由 5W 组合生成）
    _vector: list[float] | None = PrivateAttr(default=None)

    @property
    def id(self):
        return self.model_extra.get("_id", None)

    @property
    def vector(self):
        return self.model_extra.get("_vector", None)

    @property
    def sectors(self) -> list[str]:
        return self.model_extra.get("_sectors", [])

    def set_id(self, id_value: str):
        self._id = id_value
        self.model_extra["_id"] = id_value

    def set_vector(self, vector_value: list[float]):
        self._vector = vector_value
        self.model_extra["_vector"] = vector_value

    def set_sectors(self, sectors: list[Literal["episodic", "semantic", "procedural", "emotional", "reflective"]]):
        self._sectors = sectors
        self.model_extra["_sectors"] = sectors


class NodeType(str, Enum):
    """
    节点类型枚举
    """
    FACT = "fact"  # 事实节点（对应一条 Fact 记忆）
    EVENT = "event"  # 事件节点（对应 what 的核心事件）
    PERSON = "person"  # 人物节点（对应 who 中的人物）
    LOCATION = "location"  # 地点节点（对应 where 字段）
    TIME = "time"  # 时间节点（对应 when/occurred_start/occurred_end 字段）
    CONCEPT = "concept"  # 概念节点（对应 entities 中的抽象概念）
    ENTITY = "entity"  # 实体节点（对应 entities 中的具体实体）
    SECTOR = "sector"  # 记忆类型节点（episodic/semantic 等）
    EMOTION = "emotion"  # 情感/情绪节点（从 why 中提取）
    MOTIVATION = "motivation"  # 动机节点（从 why 中提取）


class BaseNode(BaseModel):
    """
    基础节点模型
    """
    node_id: str = Field(..., description="唯一标识符")
    node_type: NodeType = Field(..., description="节点类型")
    name: str = Field(..., description="节点名称")
    properties: dict = Field(default_factory=dict, description="额外属性")


class FactNode(BaseNode):
    """
    事实节点 - 对应一条完整的 Fact
    """
    node_type: NodeType = NodeType.FACT

    what: str = Field(..., description="事件的发生经过和细节描述")
    when: str | None = Field(default=None, description="事件发生时间")
    where: str | None = Field(default=None, description="事件发生地点")
    who: str | None = Field(default=None, description="涉及人员/实体")
    why: str | None = Field(default=None, description="重要性所在")
    fact_kind: Literal["conversation", "event"] = Field(..., description="事件类型")
    occurred_start: datetime | None = Field(default=None, description="事件发生的开始时间")
    occurred_end: datetime | None = Field(default=None, description="事件发生的结束时间")
    sectors: list[Literal["episodic", "semantic", "procedural", "emotional", "reflective"]] = Field(default_factory=list, description="记忆类型标签")


class EventNode(BaseNode):
    """
    事件节点 - 从 what 中提取的核心事件
    可以通过边关联：发生时间、地点、涉及人物等
    """
    node_type: NodeType = NodeType.EVENT
    description: str = Field(..., description="核心事件的简要描述")


class PersonNode(BaseNode):
    """
    人物节点
    """
    node_type: NodeType = NodeType.PERSON
    name: str = Field(..., description="姓名")
    aliases: list[str] = Field(default_factory=list, description="别名或昵称")
    background: str | None = Field(default=None, description="背景信息（从 who 字段积累）")
    relationships: dict = Field(default_factory=dict, description="与其他人物关系")


class LocationNode(BaseNode):
    """
    地点节点
    """
    node_type: NodeType = NodeType.LOCATION
    name: str = Field(..., description="完整地点名称")
    hierarchy: str | None = Field(default=None, description="层级关系，如 '中国/安徽/合肥'")


class TimeNode(BaseNode):
    """
    时间节点
    """
    node_type: NodeType = NodeType.TIME
    expression: str = Field(..., description="原始时间表述")
    timestamp: datetime | None = Field(..., description="标准化后的事实发生时间")
    is_range: bool = Field(default=False, description="是否为时间段")
    is_recurring: bool = Field(default=False, description="是否为重复事件")
    start: datetime | None = Field(default=None, description="开始时间")
    end: datetime | None = Field(default=None, description="结束时间")


class ConceptNode(BaseNode):
    """
    概念/主题节点 - 从 entities 或 why 中提取的抽象概念
    可以通过边关联相关事件、人物等
    """
    node_type: NodeType = NodeType.CONCEPT
    definition: str | None = Field(default=None, description="概念定义或描述")


class EntityNode(BaseNode):
    """
    具体实体节点 - 从 entities 中提取的具体事物
    """
    node_type: NodeType = NodeType.ENTITY
    entity_type: str = Field(..., description="实体类型")
    description: str | None = Field(default=None, description="实体描述")


class SectorNode(BaseNode):
    """
    记忆类型节点 - 用于分类记忆
    """
    node_type: NodeType = NodeType.SECTOR
    sector_type: Literal["episodic", "semantic", "procedural", "emotional", "reflective"] = Field(..., description="语义类型")
    primary: bool = Field(default=False, description="是否为主扇区")
    score: float = Field(default=0.0, description="相关性得分")


class EmotionNode(BaseNode):
    """
    情感节点 - 从 why 中提取的情绪
    """
    node_type: NodeType = NodeType.EMOTION
    emotion_type: str = Field(..., description="情感类型")
    intensity: float = Field(default=0.0, description="情感强度 0-1 之间的数值，表示情感的强烈程度")
    context: str | None = Field(default=None, description="上下文描述")


class MotivationNode(BaseNode):
    """
    动机节点 - 从why中提取的动机、目标
    """
    node_type: NodeType = NodeType.MOTIVATION
    motivation_type: str = Field(..., description="动机类型")
    description: str = Field(default=None, description="描述")


class EdgeType(str, Enum):
    """
    边类型枚举 - Fact 与各字段的连接
    """
    HAS_WHAT = "has_what"  # Fact -> Event
    HAS_WHEN = "has_when"  # Fact -> Time
    HAS_WHERE = "has_where"  # Fact -> Location
    HAS_WHO = "has_who"  # Fact -> Person
    HAS_WHY = "has_why"  # Fact -> Concept/Emotion/Motivation
    CONTAINS_ENTITY = "contains_entity"  # Fact -> Entity

    BELONGS_TO_SECTOR = "belongs_to_sector"  # Fact -> Sector

    # 实体之间的关系
    RELATED_TO = "related_to"  # 通用关系
    CO_OCCURS_WITH = "co_occurs_with"  # 共现关系
    PRECEDES = "precedes"  # 时间先后
    CAUSES = "causes"  # 因果关系
    LOCATED_IN = "located_in"  # 地点包含关系
    KNOWS = "knows"  # 人物认识关系
    WORKED_WITH = "worked_with"  # 人物合作关系

    # 时序关系
    SAME_AS = "same_as"  # 实体等同关系（用于消歧）
    PART_OF = "part_of"  # 整体-部分关系


class Edge(BaseModel):
    """
    边模型
    """
    fact_id: str | None = Field(default=None, description="来源的 Fact ID")
    source_id: str = Field(..., description="源节点 ID")
    target_id: str = Field(..., description="目标节点 ID")
    edge_type: EdgeType = Field(..., description="边类型")
    properties: dict = Field(default_factory=dict, description="额外属性，如权重、时间戳等")


if __name__ == '__main__':
    print(EntityType.CAPABILITY.name)
    print(EntityType.CAPABILITY.value)
    print(EntityType.CAPABILITY.label)
