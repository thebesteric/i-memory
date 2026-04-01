from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """
    节点类型枚举
    """
    FACT = ("fact", "事实节点")
    EVENT = ("event", "事件节点")
    PERSON = ("person", "人物节点")
    LOCATION = ("location", "地点节点")
    TIME = ("time", "时间节点")
    CONCEPT = ("concept", "概念节点")
    ENTITY = ("entity", "实体节点")
    SECTOR = ("sector", "记忆类型节点")
    EMOTION = ("emotion", "情感/情绪节点")
    MOTIVATION = ("motivation", "动机节点")


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
    sectors: list[Literal["episodic", "semantic", "procedural", "emotional", "reflective"]] = Field(
        default_factory=list, description="记忆类型标签")


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
    sector_type: Literal["episodic", "semantic", "procedural", "emotional", "reflective"] = Field(...,
                                                                                                  description="语义类型")
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
    HAS_WHAT = ("has_what", "Fact -> Event")
    HAS_WHEN = ("has_when", "Fact -> Time")
    HAS_WHERE = ("has_where", "Fact -> Location")
    HAS_WHO = ("has_who", "Fact -> Person")
    HAS_WHY = ("has_why", "Fact -> Concept/Emotion/Motivation")
    CONTAINS_ENTITY = ("contains_entity", "Fact -> Entity")

    BELONGS_TO_SECTOR = ("belongs_to_sector", "Fact -> Sector")

    # 实体之间的关系
    RELATED_TO = ("related_to", "通用关系")
    CO_OCCURS_WITH = ("co_occurs_with", "共现关系")
    PRECEDES = ("precedes", "时间先后")
    CAUSES = ("causes", "因果关系")
    LOCATED_IN = ("located_in", "地点包含关系")
    KNOWS = ("knows", "人物认识关系")
    WORKED_WITH = ("worked_with", "人物合作关系")

    # 时序关系
    SAME_AS = ("same_as", "实体等同关系（用于消歧）")
    PART_OF = ("part_of", "整体-部分关系")


class Edge(BaseModel):
    """
    边模型
    """
    fact_id: str | None = Field(default=None, description="来源的 Fact ID")
    source_id: str = Field(..., description="源节点 ID")
    target_id: str = Field(..., description="目标节点 ID")
    edge_type: EdgeType = Field(..., description="边类型")
    properties: dict = Field(default_factory=dict, description="额外属性，如权重、时间戳等")
