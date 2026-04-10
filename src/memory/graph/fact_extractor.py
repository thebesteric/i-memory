from agile.utils import timing
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.core.components import get_chat_model
from src.core.mem_ops import mem_ops
from src.memory.graph.semantic_spliter import Topic, Dialogue
from src.memory.graph.graph_models import Fact, EntityType, RelationType


class FactExtractor:
    PROMPT = """
你是一个对话事实内容提取专家。请将以下对话主题单元转换为结构化的事实（Fact）。

## 核心原则
1. **忠实原文**：what 字段应基于原始对话内容，不要添加对话中没有的信息
2. **保留原始引述**：将关键的直接引述放入 direct_quotes

## 实体标签类型
从主题单元中解析出的实体，必须严格按照如下类型赋值，如果以下类型不满足，请使用 **other** 类型：
{entity_types}

## 关系标签类型
从主题单元中解析出的实体与用户的关系，必须严格按照如下类型赋值，如果以下类型不满足，请使用 **other** 类型：
{relation_types}

## 重要提醒
1. 只提取对话中**明确提及**的信息
2. 不要添加外部知识或过度推断
3. 如果信息不完整，相关字段留空
4. 关于实体名称的提取，请务必确保准确无误，避免歧义，名称需要规范化

## 输出格式
{format_instructions}

begin!!
## 对话主题单元
- 主题: {topic}
- 摘要: {summary}
- 关键词: {keywords}
- 对话上下文: {dialogues}
"""

    def __init__(self):
        self.mem_ops = mem_ops

    @classmethod
    def get_chain(cls):
        """
        获取模型执行链
        :return:
        """
        # 构建输出解析器
        output_parser = PydanticOutputParser(pydantic_object=Fact)
        # 构建提示词模板
        prompt_template = PromptTemplate(
            template=cls.PROMPT,
            input_variables=["topic", "summary", "keywords", "dialogues"],
            partial_variables={
                "entity_types": EntityType.get_format_instructions(),
                "relation_types": RelationType.get_format_instructions(),
                "format_instructions": output_parser.get_format_instructions()
            }
        )
        # 构建语言模型
        llm = get_chat_model()
        # 构建执行链
        return prompt_template | llm | output_parser

    @timing
    async def invoke(self, topic: Topic) -> Fact:
        memories = self.mem_ops.find_mem_by_ids(topic.dialogue_ids)
        input_variables = {
            "topic": topic.name,
            "summary": topic.summary,
            "keywords": topic.keywords,
            "dialogues": [Dialogue.mem_to_dialogue(m) for m in memories]
        }
        chain = self.get_chain()
        output: Fact = await chain.ainvoke(input=input_variables)
        return output
