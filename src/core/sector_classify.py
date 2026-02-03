import asyncio
from typing import TypedDict, List, Any, Dict

import regex as re
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict

from src.core.config import env


class SectorCfg(BaseModel):
    """
    记忆领域配置
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field("", description="领域名称")
    model: str = Field(..., description="领域记忆专属的模型标识")
    decay_lambda: float = Field(..., description="记忆衰减系数")
    weight: float = Field(..., description="记忆权重")
    description: str = Field("", description="领域描述")


class ClassifyResult(BaseModel):
    """
    记忆领域分类结果
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    primary: str = Field(..., description="主扇区")
    additional: List[str] = Field(default_factory=list, description="辅扇区")
    confidence: float = Field(..., description="置信度")
    scores: Dict[str, float] = Field(..., description="各扇区分数")


class ClassifyOutput(BaseModel):
    """
    记忆领域分类模型输出格式
    """
    episodic: float = Field(0.0, description="情景记忆分数")
    semantic: float = Field(0.0, description="语义记忆分数")
    procedural: float = Field(0.0, description="程序记忆分数")
    emotional: float = Field(0.0, description="情绪记忆分数")
    reflective: float = Field(0.0, description="反思记忆分数")
    reasoning: str = Field(..., description="简短说明分类理由")

    def sorted_scores(self) -> List[tuple[str, int]]:
        """
        返回按分数从高到低排序的领域列表
        :return:
        """
        model_dump = self.model_dump()
        scores = {k: v for k, v in model_dump.items() if k != "reasoning"}
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def get_primary_sector(self) -> tuple[str, int]:
        """
        获取主扇区
        :return:
        """
        sorted_scores = self.sorted_scores()
        primary, p_score = sorted_scores[0]
        return primary, p_score

    def get_additional_sectors(self, *, threshold: float = 30, primary_percent: float = 0.4) -> List[tuple[str, int]]:
        """
        获取辅扇区，分数高于阈值的领域
        :param threshold: 分数阈值
        :param primary_percent: 主扇区分数的百分比阈值
        :return:
        """
        sorted_scores = self.sorted_scores()
        primary, p_score = sorted_scores[0]
        # 设定阈值：分数 >= 30 且 >= 主分数的 40% 的其他扇区作为辅扇区
        thresh = max(threshold, p_score * primary_percent)
        additional = [(sec, score) for sec, score in sorted_scores[1:] if score >= thresh]
        return additional or []

    def get_confidence(self) -> float:
        """
        计算置信度：基于最高分与次高分的差距
        :return:
        """
        sorted_scores = self.sorted_scores()
        primary, p_score = sorted_scores[0]
        # 获取次高分
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        if p_score > 0:
            # 归一化到 0-1 之间，分数差距越大置信度越高
            confidence = min(1.0, (p_score - second_score) / 100.0 + 0.5)
        else:
            confidence = 0.2
        return round(confidence, 2)


# 领域配置
SECTOR_CONFIGS: Dict[str, SectorCfg] = {
    "episodic": SectorCfg(
        name="情景记忆",
        model="episodic-optimized",
        decay_lambda=0.015,
        weight=1.2,
        description="具体的时间、地点、事件、经历",
    ),
    "semantic": SectorCfg(
        name="语义记忆",
        model="semantic-optimized",
        decay_lambda=0.005,
        weight=1.0,
        description="客观的知识、概念、事实、学科信息",
    ),
    "procedural": SectorCfg(
        name="程序记忆",
        model="procedural-optimized",
        decay_lambda=0.008,
        weight=1.1,
        description="步骤、方法、操作流程、技能",
    ),
    "emotional": SectorCfg(
        name="情绪记忆",
        model="emotional-optimized",
        decay_lambda=0.02,
        weight=1.3,
        description="情绪、感受、主观体验",
    ),
    "reflective": SectorCfg(
        name="反思记忆",
        model="reflective-optimized",
        decay_lambda=0.001,
        weight=0.8,
        description="思考、洞察、总结、成长、反馈",
    ),
}

# 领域权重
SEC_WTS = {k: v.weight for k, v in SECTOR_CONFIGS.items()}

# 领域描述
SEC_DESCRIPTIONS = {k: v.description for k, v in SECTOR_CONFIGS.items()}


class SectorClassifier:

    # 分类提示词模板
    CLASSIFY_PROMPT = """
你是一个记忆分类专家。请分析以下内容，判断它属于哪个记忆类型，并为每个类型打分（0-100分）。

## 记忆类型说明：
{sec_descriptions}


## 输出格式
{format_instructions}

## 注意事项
1. 分数总和不必为 100，每个类型独立打分，区间为 0-100 分；
2. 主要类型分数应明显高于其他类型；
3. 一段内容可能同时涉及多个类型；
4. 分数 0 表示完全不相关，100 表示非常典型；

begin!!
分析内容：{content}
"""

    def __init__(self, *, content: str, metadata: Any = None):
        self.content = content
        self.metadata = metadata or {}

    @classmethod
    def get_chain(cls):
        """
        获取模型执行链
        :return:
        """
        # 构建输出解析器
        output_parser = PydanticOutputParser(pydantic_object=ClassifyOutput)
        # 构建提示词模板
        prompt_template = PromptTemplate(
            template=cls.CLASSIFY_PROMPT,
            input_variables=["content"],
            partial_variables={
                "sec_descriptions": chr(10).join([f"- {k}: {v.description}" for k, v in SECTOR_CONFIGS.items()]),
                "format_instructions": output_parser.get_format_instructions()
            }
        )
        # 构建语言模型
        llm = ChatOpenAI(
            model=env.OPENAI_MODEL,
            temperature=0.0,
            api_key=env.OPENAI_API_KEY,
            base_url=env.OPENAI_BASE_URL
        )
        # 构建执行链
        return prompt_template | llm | output_parser

    async def classify(self) -> ClassifyResult:
        """
        根据文本内容分类记忆领域
        :param content: 文本内容
        :param metadata: 元数据，可包含预设的 sector
        :return: 包含 primary（主扇区）、additional（辅扇区）、confidence（置信度）、scores（各扇区分数）的字典
        """

        # 如果 metadata 参数中已经指定了 sector（且在配置中合法），直接返回该 sector 作为主扇区，置信度为 1.0，不再做内容分析
        meta_sec = self.metadata.get("sector") if isinstance(self.metadata, dict) else None
        if meta_sec and meta_sec in SECTOR_CONFIGS:
            return ClassifyResult(
                primary=meta_sec,
                additional=[],
                confidence=1.0,
                scores={meta_sec: 100.0}
            )

        chain = self.get_chain()
        output: ClassifyOutput = await chain.ainvoke({"content": self.content})

        # 获取主扇区
        primary, p_score = output.get_primary_sector()
        # 获取辅扇区，分数阈值大于 30.0 且大于主扇区分数的 40%
        additional = output.get_additional_sectors(threshold=30.0, primary_percent=0.4)
        # 计算置信度：基于最高分与次高分的差距
        confidence = output.get_confidence()

        # 如果主扇区分数过低（< 20），使用 semantic 作为兜底
        if p_score < 20:
            primary = "semantic"
            confidence = 0.3

        return ClassifyResult(
            primary=primary,
            additional=[sec for sec, score in additional],
            confidence=confidence,
            scores={sec: round(secore, 2) for sec, secore in output.sorted_scores()}
        )


if __name__ == '__main__':
    asyncio.run(SectorClassifier(content="我昨天去了公园，感觉非常开心！").classify())
