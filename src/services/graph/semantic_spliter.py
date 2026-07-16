from datetime import datetime
from typing import Any

from agile.utils import timing
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from services.memory.components import get_chat_model
from domain.graph.models import Topic


class Dialogue(BaseModel):
    id: str
    content: str
    role: str
    created_at: datetime

    @staticmethod
    def mem_to_dialogue(mem: dict[str, Any]) -> "Dialogue":
        return Dialogue(
            id=mem["id"],
            content=mem["content"],
            role=mem.get("role", "human") if mem.get("role") else "human",
            created_at=mem.get("created_at", "")
        )


class SemanticsOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    topics: list[Topic] = Field(default_factory=list, description="相关主题对象")


class SemanticSpliter:
    PROMPT = """
你是一个对话信息提取专家。请将以下对话按语义主题切分成多个独立的知识单元。

## 切分规则
1. 每个知识单元应该是一个完整的知识点，能够独立理解；
2. 当对话切换主题时（如从"定义"切换到"安装方法"），则应该切分；
3. 当多个问答围绕同一个主题时，应该合并为一个单元；
4. 对话是自然对话，回答不一定紧跟在问题的后面，也许会在后面的对话中才出现；
5. 切分时需要考虑上下文，确保每个单元的内容完整且有逻辑连贯性；
6. 切分后每个单元应该能够独立成章，具备完整的知识点，不依赖于其他单元的内容；

## 注意事项
1. 严格遵循对话的事实（实际）内容，不要添加任何没有在对话中出现过的信息；
2. 如果给定的内容没有按照时间排序，你需要先排序，在进行分析提取；
3. 对话中可能会出现无关的对话，或者没有意义的对话，允许适当丢弃，不要强加入某个主题对象中；

## 输出格式
{format_instructions}

begin!!
## 对话内容
{dialogues}
"""

    def __init__(self):
        pass

    @classmethod
    def get_chain(cls):
        """
        获取模型执行链
        :return:
        """
        # 构建输出解析器
        output_parser = PydanticOutputParser(pydantic_object=SemanticsOutput)
        # 构建提示词模板
        prompt_template = PromptTemplate(
            template=cls.PROMPT,
            input_variables=["dialogues"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            }
        )
        # 构建语言模型
        llm = get_chat_model()
        # 构建执行链
        return prompt_template | llm | output_parser

    @timing
    async def invoke(self, dialogues: list[Dialogue]) -> SemanticsOutput:
        chain = self.get_chain()
        output: SemanticsOutput = await chain.ainvoke({"dialogues": dialogues})
        return output
