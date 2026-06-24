from typing import Any

from agile.utils import timing
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from services.memory.components import get_chat_model
from services.graph.semantic_spliter import Dialogue
from domain.session.models import SessionCollection


class SessionExtractor:
    PROMPT = """
# 会话提取任务

## 角色定义
你是对话结构化提取专家。请将给定对话按“事件”聚合为一个或多个会话（sessions）。

## 核心目标
- 一个会话 = 一个核心事件
- 如果对话中存在多个独立事件，必须输出多个 session
- 如果仅有一个事件，输出一个 session
- 如果没有有效事件，返回空 sessions

## 语言设定
根据对话内容确认主要语言：中文为主则中文输出，英文为主则英文输出。

## 会话的定义
1. **核心定位**：会话是用户围绕「同一个核心事件」，在一段连续或相近时间内产生的完整交互单元。
2. **主题关联**：一个会话可包含多个相关细分主题，所有对话消息、细分主题均围绕核心事件展开，服务于事件的推进、信息补充、决策制定或执行落地，不包含与核心事件无关的冗余内容。

## 会话划分规则（必须遵守）
1. 同一事件的不同子主题应合并到同一个 session。
2. 不同事件必须拆分为不同 session，不可强行合并。
3. 事件切换信号包括但不限于：目标变化、时间/场景明显切换、任务对象变化。
4. 与任何核心事件无关的寒暄、噪声、无意义片段可忽略。

## dialogue_ids 规则（必须遵守）
1. `dialogue_ids` 只能使用输入对话中真实存在的 id，不可编造。
2. 同一条对话 id 只能属于一个 session（不重复、不重叠）。
3. 每个 session 的 `dialogue_ids` 需按时间升序排列。
4. 在不引入噪声的前提下，尽量覆盖所有有效对话。

## summary / key_facts 规则（必须遵守）
1. `summary` 要准确描述该事件的过程与结论，不要泛化为空话。
2. `key_facts` 只保留对话中明确提及的事实，不得臆造。
3. `key_facts` 尽量具体，优先提取时间、偏好、约束、决策、行动项。

### 会话结构逻辑
```text
一个会话 = 一个事件
   ↳ 包含多个主题（均围绕核心事件）
   ↳ 包含多条对话消息（构成完整交互）
```

### 会话示例说明
- 核心事件：讨论准备买什么样的汽车
- 会话包含的细分主题：讨论预算、讨论油车还是电车、讨论使用场景、讨论车型的选择、讨论车辆保养费用等。

## 输出格式要求（严格）
{format_instructions}

只输出符合格式的结果，不要输出任何额外解释文本。

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
        output_parser = PydanticOutputParser(pydantic_object=SessionCollection)
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
    async def invoke(self, memories: list[dict[str, Any]]) -> SessionCollection:
        # 先按时间排序，降低模型在乱序输入时的分段抖动。
        sorted_memories = sorted(memories, key=lambda m: str(m.get("created_at") or ""))
        input_variables = {
            "dialogues": [Dialogue.mem_to_dialogue(m) for m in sorted_memories]
        }
        chain = self.get_chain()
        output: SessionCollection = await chain.ainvoke(input=input_variables)
        return output
