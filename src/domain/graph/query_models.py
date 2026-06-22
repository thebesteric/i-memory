from typing import Optional, List

from pydantic import BaseModel, Field


class GraphFactsFilters(BaseModel):
    topic_id: Optional[str] = Field(default=None, description="按主题 ID 过滤")
    fact_kind: Optional[str] = Field(default=None, description="按事实类型过滤（conversation/event）")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最小置信度")
    max_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最大置信度")
    keyword: Optional[str] = Field(default=None, description="在 what/who/where/why 字段中模糊匹配")


class GraphEntityRelationFilters(BaseModel):
    fact_id: Optional[str] = Field(default=None, description="按证据 fact_id 过滤")
    edge_relations: Optional[List[str]] = Field(default=None, description="按边关系过滤")
    infer_sources: Optional[List[str]] = Field(default=None, description="按推断来源过滤")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最小置信度")
    max_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最大置信度")
    related_canonical_id: Optional[str] = Field(default=None, description="按关联实体 ID 过滤")