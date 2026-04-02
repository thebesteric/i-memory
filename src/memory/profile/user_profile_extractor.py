import datetime
from typing import Any

from agile.utils import timing
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.core.components import get_chat_model
from src.core.mem_ops import mem_ops
from src.memory.graph.semantic_spliter import Dialogue
from src.memory.memory_models import IMemoryUser
from src.memory.profile import user_profile_ops
from src.memory.profile.user_profile_models import UserProfile, Personality


class UserProfileExtractor:
    PROMPT = """
# 用户画像分析任务

## 角色定义
你是一个专业的用户画像分析专家，擅长从对话中提取用户特征、行为习惯、偏好倾向和关键标签。

## 语言设定
根据对话内容，确认使用的语言设置，对话主要是中文，那么就使用中文输出内容，如果对话主要以英文为主，则使用英文来输出内容

## 任务目标
分析用户的历史对话内容，完善或更新用户画像信息。你需要根据对话内容，判断哪些画像字段需要新增、更新或保持不变。

## 画像字段说明
- **personality**: 个性信息（年龄范围、性别、职业、教育、位置等）
- **preferences.habits**: 习惯列表，从行为推断
- **preferences.settings**: 偏好设置列表
- **tags**: 标签列表

## [重要]长对话处理策略
对话内容可能很长，请按照以下策略高效处理：
### 1. 信息提取优先级（由高到低）
- **直接陈述**: 用户明确说"我是..."、"我喜欢..."、"我习惯..."等
- **重复提及**: 多次出现的主题或关键词
- **详细描述**: 包含具体细节的对话段落
- **情感表达**: 带有强烈情感倾向的内容

### 2. 提取方法
- **先扫描后聚焦**: 快速浏览全文，识别关键信息点，再深入分析
- **主题聚类**: 将相关对话按主题分组（如职业、爱好、生活习惯）
- **去重合并**: 相同主题的信息合并处理，避免重复

### 3. 信息筛选标准
**应该提取的信息**:
- 个人信息：年龄、职业、教育、地理位置等
- 明确偏好：喜欢/不喜欢什么
- 行为模式：经常做什么、习惯性行为
- 知识领域：专业背景、擅长领域
- 价值观：对事物的看法和态度

**应该忽略的信息**:
- 临时性的、一次性的对话（如"今天天气真好"）
- 上下文无关的闲聊
- 明显的玩笑或反话（除非能明确判断）
- 敏感隐私信息（身份证号、密码等）

### 4. 证据引用规范
- 每个推断至少提供 1 条以上的对话证据
- 证据要简洁，提取关键句子，不要整段复制
- 格式：`[时间戳或对话序号] 用户说："..."`
- 如果对话中有明确时间标记，请保留证据的时间信息

## 分析规则
### 置信度判断标准
- **高置信度(0.8-1.0)**: 
  - 用户明确陈述，且多次提及
  - 有具体细节和行为描述
  - 结合当前时间，信息具有持续性
  
- **中置信度(0.5-0.8)**: 
  - 有相关提及但不够明确
  - 仅有1-2次提及
  - 间接但合理的推断
  
- **低置信度(0.0-0.5)**: 
  - 单次提及且模糊
  - 间接暗示，缺乏直接证据
  - 需要谨慎推断的信息

### 关于标签的提取
- 标签设置规则
    - 标签名称一定有代表性，能够准确的描述用户某些特征，如：科技控、时尚达人、万事通
- 标签来源判断
    - **explicit**: 用户明确说 "我喜欢XX"、"我是XX"、"我经常XX" 等
    - **implicit**: 从对话内容推断，如用户多次讨论某个话题，或从行为描述中推断

### 信息更新策略
1. **新增**: 
    - 现有画像中没有的信息
    - 检查是否有 `created_at` 字段，若有则更新为当前时间
2. **更新**: 
   - 当新信息与旧信息冲突时，以最新信息为准
   - 对于习惯和标签，若已存在同名项，则更新其权重和证据，提高置信度
   - 检查是否有 `updated_at` 字段，若有则更新为当前时间
   - 检查是否有 `weight` 或 `confidence` 字段，适当调整数值以反映新信息的可靠程度和重要性
3. **保留**: 
    - 无相关新信息时，保持原有内容不变

### 字段填写规范
- `confidence`: 0-1 之间的浮点数，表示推断可靠程度
- `weight`: 0-1 之间的浮点数，表示标签重要程度
- `source`: "explicit" 或 "implicit"
- `evidences`: 引用具体的对话内容作为证据，按上述规范格式化

### 时间信息处理
- 用户对话中的相对时间（如 "昨天"、"上周"、"最近"）需要结合当前时间 {current_time} 转换为绝对时间
> 示例：如果当前时间是 2026-03-31，用户说"我上周开始学Python"，则记录首次提及时间为 2026-03-24 左右

### 输出数量控制
优先保留置信度高、证据充分的信息，可使用 LRU 淘汰策略
- `habits`: 最多输出 10 个最相关、置信度最高的习惯
- `tags`: 最多输出 15 个最相关、权重最高的标签

## 关于 extra 字段
- extra 字段为扩展字段，类型为字典类型，当某些特殊信息无法匹配到上述字段时，可以放在 extra 字段中，key 是字段名称，value 是字段值
> 示例：如 Demographic 中的 extra 字段，当解析出用户的身高、体重，但是 Demographic 中并没有对应的字段，可将该信息储存到 Demographic 到 extra 字段中

## 当前时间
{current_time}

## 当前用户画像
{current_user_profile}

## 性格枚举（Personality）
用户性格必须严格按照下述进行赋值，不允许私自创建性格
{personalities}

## 输出格式
{format_instructions}

## 注意事项
1. 只更新在对话中有明确依据的信息，不要过度推断
2. 对于不确定的信息，适当降低置信度
3. 如果某个字段没有需要更新的内容，保持原有值不变
4. 对话证据要引用原文不要改写，保持真实性
5. 控制输出数量，避免输出过多低质量信息
6. 如果对话内容过长，优先提取关键信息，忽略无关闲聊
7. 所有时间字段必须使用 yyyy-MM-dd HH:mm:ss 格式

begin!! 请开始分析并生成更新后的用户画像
{dialogues}
"""

    def __init__(self):
        self.user_profile_ops = user_profile_ops
        self.mem_ops = mem_ops

    @classmethod
    def get_chain(cls):
        """
        获取模型执行链
        :return:
        """
        # 构建输出解析器
        output_parser = PydanticOutputParser(pydantic_object=UserProfile)
        # 构建提示词模板
        prompt_template = PromptTemplate(
            template=cls.PROMPT,
            input_variables=["current_user_profile", "dialogues"],
            partial_variables={
                "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "personalities": Personality.get_format_instructions(),
                "format_instructions": output_parser.get_format_instructions()
            }
        )
        # 构建语言模型
        llm = get_chat_model()
        # 构建执行链
        return prompt_template | llm | output_parser

    @timing
    async def invoke(self, user: IMemoryUser, memories: list[dict[str, Any]]) -> UserProfile:
        current_user_profile: UserProfile = await self.user_profile_ops.get_user_profile(user)
        input_variables = {
            "current_user_profile": current_user_profile.model_dump() if current_user_profile else None,
            "dialogues": [Dialogue.mem_to_dialogue(m) for m in memories]
        }
        chain = self.get_chain()
        output: UserProfile = await chain.ainvoke(input=input_variables)
        return output