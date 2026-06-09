from typing import Any

from agile.utils import timing
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.core.components import get_chat_model
from src.memory.graph.semantic_spliter import Dialogue
from src.memory.memory_models import IMemoryUser
from src.memory.session.session_models import Sessions


class SessionExtractor:
    PROMPT = """
# 会话提取任务

## 角色定义
你是一个对话内容提取专家。将对话提取为会话，请根据以下对话内容，分析一系列对话的核心信息，并生成为会话。

## 语言设定
根据对话内容，确认使用的语言设置，对话主要是中文，那么就使用中文输出内容，如果对话主要以英文为主，则使用英文来输出内容

## 会话的定义
1. **核心定位**：会话是用户围绕「同一个核心事件」，在一段连续或相近时间内产生的完整交互单元。
2. **主题关联**：一个会话可包含多个相关细分主题，所有对话消息、细分主题均围绕核心事件展开，服务于事件的推进、信息补充、决策制定或执行落地，不包含与核心事件无关的冗余内容。

### 会话结构逻辑
```text
一个会话 = 一个事件
   ↳ 包含多个主题（均围绕核心事件）
   ↳ 包含多条对话消息（构成完整交互）
```

### 会话示例说明
- 核心事件：讨论准备买什么样的汽车
- 会话包含的细分主题：讨论预算、讨论油车还是电车、讨论使用场景、讨论车型的选择、讨论车辆保养费用等。

## 输出格式要求
{format_instructions}

begin!! 请开始分析以下对话内容，严格按照上述规则提取会话信息：
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
        output_parser = PydanticOutputParser(pydantic_object=Sessions)
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
    async def invoke(self, memories: list[dict[str, Any]]) -> Sessions:
        input_variables = {
            "dialogues": [Dialogue.mem_to_dialogue(m) for m in memories]
        }
        chain = self.get_chain()
        output: Sessions = await chain.ainvoke(input=input_variables)
        return output
