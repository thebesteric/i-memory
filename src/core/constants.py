import re
from typing import TypedDict, List, Dict


class MemoryConstants:
    """
    记忆相关的常量配置
    """
    # 长语句阈值
    LARGE_TOKEN_THRESH = 8000
    # 文本分段大小阈值
    SECTION_SIZE = 3000


class SectorCfg(TypedDict):
    """
    记忆领域配置
    """
    # 领域记忆专属的模型标识
    model: str
    # 记忆衰减系数
    decay_lambda: float
    # 记忆权重
    weight: float
    # 正则表达式列表
    patterns: List[re.Pattern]


# 领域配置
SECTOR_CONFIGS: Dict[str, SectorCfg] = {
    # 情景记忆：具体的时间、地点、事件、经历
    "episodic": SectorCfg(
        model="episodic-optimized",
        decay_lambda=0.015,
        weight=1.2,
        patterns=[
            re.compile(r"\b(today|yesterday|tomorrow|last\s+(week|month|year)|next\s+(week|month|year))\b", re.I),
            re.compile(r"\b(remember\s+when|recall|that\s+time|when\s+I|I\s+was|we\s+were)\b", re.I),
            re.compile(r"\b(went|saw|met|felt|heard|visited|attended|participated)\b", re.I),
            re.compile(r"\b(at\s+\d{1,2}:\d{2}|on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.I),
            re.compile(r"\b(event|moment|experience|incident|occurrence|happened)\b", re.I),
            re.compile(r"\bI\s+'?m\s+going\s+to\b", re.I),
        ]
    ),
    # 语义记忆：客观知识、概念、事实、学科信息
    "semantic": SectorCfg(
        model="semantic-optimized",
        decay_lambda=0.005,
        weight=1.0,
        patterns=[
            re.compile(r"\b(is\s+a|represents|means|stands\s+for|defined\s+as)\b", re.I),
            re.compile(r"\b(concept|theory|principle|law|hypothesis|theorem|axiom)\b", re.I),
            re.compile(r"\b(fact|statistic|data|evidence|proof|research|study|report)\b", re.I),
            re.compile(r"\b(capital|population|distance|weight|height|width|depth)\b", re.I),
            re.compile(r"\b(history|science|geography|math|physics|biology|chemistry)\b", re.I),
            re.compile(r"\b(know|understand|learn|read|write|speak)\b", re.I),
        ]
    ),
    # 程序记忆：步骤、方法、操作流程、技能
    "procedural": SectorCfg(
        model="procedural-optimized",
        decay_lambda=0.008,
        weight=1.1,
        patterns=[
            re.compile(r"\b(how\s+to|step\s+by\s+step|guide|tutorial|manual|instructions)\b", re.I),
            re.compile(r"\b(first|second|then|next|finally|afterwards|lastly)\b", re.I),
            re.compile(r"\b(install|run|execute|compile|build|deploy|configure|setup)\b", re.I),
            re.compile(r"\b(click|press|type|enter|select|drag|drop|scroll)\b", re.I),
            re.compile(r"\b(method|function|class|algorithm|routine|recipie)\b", re.I),
            re.compile(r"\b(to\s+do|to\s+make|to\s+build|to\s+create)\b", re.I),
        ]
    ),
    # 情绪记忆：情绪、感受、主观体验
    "emotional": SectorCfg(
        model="emotional-optimized",
        decay_lambda=0.02,
        weight=1.3,
        patterns=[
            re.compile(r"\b(feel|feeling|felt|emotions?|mood|vibe)\b", re.I),
            re.compile(r"\b(happy|sad|angry|mad|excited|scared|anxious|nervous|depressed)\b", re.I),
            re.compile(r"\b(love|hate|like|dislike|adore|detest|enjoy|loathe)\b", re.I),
            re.compile(r"\b(amazing|terrible|awesome|awful|wonderful|horrible|great|bad)\b", re.I),
            re.compile(r"\b(frustrated|confused|overwhelmed|stressed|relaxed|calm)\b", re.I),
            re.compile(r"\b(wow|omg|yay|nooo|ugh|sigh)\b", re.I),
            re.compile(r"[!]{2,}", re.I),
        ]
    ),
    # 反思记忆：思考、洞察、总结、成长、反馈
    "reflective": SectorCfg(
        model="reflective-optimized",
        decay_lambda=0.001,
        weight=0.8,
        patterns=[
            re.compile(r"\b(realize|realized|realization|insight|epiphany)\b", re.I),
            re.compile(r"\b(think|thought|thinking|ponder|contemplate|reflect)\b", re.I),
            re.compile(r"\b(understand|understood|understanding|grasp|comprehend)\b", re.I),
            re.compile(r"\b(pattern|trend|connection|link|relationship|correlation)\b", re.I),
            re.compile(r"\b(lesson|moral|takeaway|conclusion|summary|implication)\b", re.I),
            re.compile(r"\b(feedback|review|analysis|evaluation|assessment)\b", re.I),
            re.compile(r"\b(improve|grow|change|adapt|evolve)\b", re.I),
        ]
    ),
}

# 领域权重
SEC_WTS = {k: v["weight"] for k, v in SECTOR_CONFIGS.items()}
