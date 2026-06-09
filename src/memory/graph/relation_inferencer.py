from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.core.components import get_chat_model
from src.memory.graph.graph_models import Fact
from src.memory.graph.graph_node_models import EdgeRelation


class RelationInferenceOutput(BaseModel):
    edge_relation: str = Field(..., description="边关系，必须是枚举中的 name")
    relation_evidence: str | None = Field(default=None, description="边关系证据")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")


class RelationInference:
    PROMPT = """
你是一个严格的图关系抽取器。请根据给定 fact 和实体对，选择最合适的 edge_relation。

## 允许的边关系类型（必须输出 name）
{edge_relations}

## 强约束
1. 仅基于输入文本，不得引入外部知识或常识补全。
2. 输出 edge_relation 必须是上面枚举中的 name（全大写下划线风格）。
3. 形成关系的证据 relation_evidence 字段，必须引用输入中的关键信号（关键词、语义线索），禁止空泛描述。
4. 关系方向按输入顺序理解：source -> target。

## 兜底规则
- 若证据不足以支持明确关系，或仅共同出现而无语义关系：必须输出 **CO_OCCURS_WITH**

## 置信度标尺（confidence）
- 0.90~1.00: 文本中有直接、明确、无歧义证据
- 0.70~0.89: 证据较强，但仍有轻微歧义
- 0.40~0.69: 仅弱线索或间接线索
- 0.00~0.39: 基本无法判断，应倾向 CO_OCCURS_WITH

## 输出格式
{format_instructions}

## 输入
- fact.what: {what}
- fact.when: {when}
- fact.where: {where}
- fact.who: {who}
- fact.why: {why}
- source_entity_texts: {source_texts}
- source_entity_type: {source_type}
- target_entity_texts: {target_texts}
- target_entity_type: {target_type}
"""

    @classmethod
    def get_chain(cls):
        output_parser = PydanticOutputParser(pydantic_object=RelationInferenceOutput)
        prompt_template = PromptTemplate(
            template=cls.PROMPT,
            input_variables=[
                "what", "when", "where", "who", "why",
                "source_texts", "source_type", "target_texts", "target_type"
            ],
            partial_variables={
                "edge_relations": EdgeRelation.get_format_instructions(),
                "format_instructions": output_parser.get_format_instructions(),
            },
        )
        llm = get_chat_model()
        return prompt_template | llm | output_parser

    async def invoke(self, fact: Fact, source_meta: dict, target_meta: dict) -> RelationInferenceOutput:
        chain = self.get_chain()
        return await chain.ainvoke(
            {
                "what": fact.what,
                "when": fact.when,
                "where": fact.where,
                "who": fact.who,
                "why": fact.why,
                "source_texts": source_meta.get("texts", []),
                "source_type": source_meta.get("entity_type", "OTHER"),
                "target_texts": target_meta.get("texts", []),
                "target_type": target_meta.get("entity_type", "OTHER"),
            }
        )
