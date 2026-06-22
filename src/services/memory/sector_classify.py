import os
from typing import List, Any, Dict, Literal

from agile.utils import timing, LogHelper
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ConfigDict

from infra.ai.classifier.bert_manager import BertManager, BertIncrModel, LabelConfig, LabelBranchConfig
from shared.config.settings import env

logger = LogHelper.get_logger(title="[CLASSIFY]")


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

    @classmethod
    def from_predictions(cls, predictions: dict[str, dict[str, Any]]) -> "ClassifyOutput":
        # 获取主扇区信息
        primary_dict = predictions["primary"]
        primary_pred = primary_dict['pred']

        # 获取所有扇区信息
        labels_dict = predictions["labels"]
        labels_pred = labels_dict["pred"]
        probabilities = labels_dict["probabilities"]

        # 兜底：如果预测出的所有扇区不包含主扇区的话，则强制增加主扇区
        p_val = labels_pred[primary_pred]
        if p_val == 0:
            labels_pred[primary_pred] = 1
            probabilities[primary_pred] = primary_dict["confidence"]

        # 计算各扇区得分
        scores = {SECTOR_INDEX_KEY_MAPPING.get(idx, "semantic"): round(prob * 100, 1) for idx, prob in enumerate(probabilities)}

        return cls(
            **scores,
            reasoning="Using BERT model predictions to classify."
        )

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
        # 如：主扇区 80 分，相对阈值 80 * 0.4 = 32，最终阈值 thresh = 32 那么其他扇区里分数 >= 32 的都会被选为辅扇区
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
        description="思考、想法、洞察、总结、成长、反馈",
    ),
}

# 提取键值并按顺序映射 => {'episodic': 0, 'semantic': 1, 'procedural': 2, 'emotional': 3, 'reflective': 4}
SECTOR_KEY_INDEX_MAPPING: dict[str, int] = {
    sector_key: idx for idx, sector_key in enumerate(SECTOR_CONFIGS.keys())
}

# 提取索引并按顺序映射 => {0: 'episodic', 1: 'semantic', 2: 'procedural', 3: 'emotional', 4: 'reflective'}
SECTOR_INDEX_KEY_MAPPING: dict[int, str] = {
    idx: sector_key for idx, sector_key in enumerate(SECTOR_CONFIGS.keys())
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
1. 每个类型独立打分，区间为 0-100 分；
2. 一段内容可能同时涉及多个类型，但是主类型分数应明显高于其他类型；
3. 分数 0 表示完全不相关，100 表示非常典型；
4. 评分理由 reasoning 字段需要简短说明分类理由，突出主类型的核心特征和内容中的相关信息；

begin!!
分析内容：{content}
"""

    def __init__(self, *, checkpoint_path: str = None):
        # 权重文件
        self.checkpoint_path = checkpoint_path
        # 初始化 Bert 模型
        self.bert_manager, self.bert_incr_model = self.init_bert_model()

    def init_bert_model(self) -> tuple[BertManager | None, BertIncrModel | None]:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return None, None

        # Bert 管理器
        bert_manager = BertManager(
            model_name_or_path="google-bert/bert-base-multilingual-cased",
        )
        # Bert 增量模型
        bert_incr_model = BertIncrModel(
            bert_manager=bert_manager,
            in_features=768,
            out_features_config=LabelConfig(
                branches={
                    "primary": LabelBranchConfig(type="single", num_classes=5),
                    "labels": LabelBranchConfig(type="multi", num_classes=5),
                }
            ),
        )
        return bert_manager, bert_incr_model

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
        from services.memory.components import get_chat_model
        llm = get_chat_model()
        # 构建执行链
        return prompt_template | llm | output_parser

    @timing
    async def classify(self, *, content: str, metadata: Any = None) -> ClassifyResult:
        """
        根据文本内容分类记忆领域
        :return: 包含 primary（主扇区）、additional（辅扇区）、confidence（置信度）、scores（各扇区分数）的字典
        """

        # 如果 metadata 参数中已经指定了 sector（且在配置中合法），直接返回该 sector 作为主扇区，置信度为 1.0，不再做内容分析
        meta_sec = metadata.get("sector") if isinstance(metadata, dict) else None
        if meta_sec and meta_sec in SECTOR_CONFIGS:
            return ClassifyResult(
                primary=meta_sec,
                additional=[],
                confidence=1.0,
                scores={meta_sec: 100.0}
            )

        if env.USE_BERT_CLASSIFIER:
            # 检查
            if not self.bert_manager or not self.bert_incr_model:
                raise ValueError("BERT model is not initialized. Please provide a valid checkpoint path for initialization.")
            # 调用 Bert 模型进行预测
            result = self.bert_manager.predict(
                text=content,
                bert_incr_model=self.bert_incr_model,
                checkpoint_path=self.checkpoint_path,
                strict=False,
                return_probabilities=True,
            )
            # 获取预测结果
            predictions = result["predictions"]
            # 计算各扇区得分
            output: ClassifyOutput = ClassifyOutput.from_predictions(predictions)
        else:
            # 调用模型执行链获取分类结果
            chain = self.get_chain()
            # 计算各扇区得分
            output: ClassifyOutput = await chain.ainvoke({"content": content})

        # 返回 ClassifyResult 对象
        return self.to_classify_result(content, output)

    @staticmethod
    def to_classify_result(content: str, output: ClassifyOutput) -> ClassifyResult:
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
            logger.info(f"Primary '{primary}' Sector score is too low: {p_score}, fallback to 'semantic' sector.")

        # 各扇区得分情况
        scores = {sec: round(secore, 2) for sec, secore in output.sorted_scores()}

        logger.info(
            f"Classify content: {content[:30]}... => primary: {primary}, additional: {[sec for sec, score in additional]}, "
            f"confidence: {confidence}, scores: {scores}"
        )

        return ClassifyResult(
            primary=primary,
            additional=[sec for sec, score in additional],
            confidence=confidence,
            scores=scores
        )


class SectorSentenceOutput(BaseModel):
    sector: str = Field(..., description="所属领域")
    sentences: list[str] = Field(default_factory=list, description="该领域的相关句子列表")


class SectorSentenceCreator:
    CREATE_PROMPT = """
你精通语义领域分析，并擅长根据语义领域创建符合该领域特征的句子，请根据以下内容要求创建“主要含义”符合该语义领域的句子：

## 领域类型说明：
{sec_descriptions}

## 输出格式
{format_instructions}

## 生成要求
1. 句子内容要相对完整，质量高，最好不要超过 100 个字符；
2. 句子主要含义需要符合所给定的主领域，并且能够体现该领域的核心特征；
3. 生成句子的过程中，你需要自习揣摩并思考，句子除主领域含义外，尽量包含一个或多个其他领域的含义，但是不要让其他领域含义反客为主；
4. 辅助扇区的权重要大于主领域的权重的 40%，如：整条句子中主领域得分 80，那么辅助领域至少要有 80 * 0.4 = 32 的分数才能被包含在句子中；
5. 每次生成的句子不要和上一批次生成的句子重复；
6. 生成的句子需满足上述条件；

## 其他注意事项
1. 如果需要你生成 emotional 类型的句子的话，多加一些情绪、感受、主观体验的描述信息。

## 例子
### Example 1: 去年春天去婺源看油菜花，漫山遍野的金黄色让我忍不住在田埂上跑了起来。
主领域：episodic
包含其他领域：emotional

### Example 2: 知道了声音在空气中的传播速度约340米/秒，由此反思出为什么先看到闪电后听到雷声。
主领域：semantic
包含其他领域：reflective

### Example 3: 掌握了叠衣服的标准流程：先铺平再对折，知道正确收纳能节省30%的衣柜空间，这让我意识到整理的重要性。
主领域：procedural
包含其他领域：semantic, reflective

### Example 4: 深夜改完第十二稿方案，合上电脑那刻涌上的不是轻松，是种疲惫而踏实的微光。
主领域：emotional
包含其他领域：episodic, reflective

### Example 5: 给绿萝换盆时先轻拍旧盆脱土、剪除腐根、填入新营养土再定植压实，三次实践后，我发现自己开始用系统思维观察植物生长节律。
主领域：reflective
包含其他领域：episodic, procedural

begin!!
所属主领域：{sector}
生成相关领域句子的数量：{num}
"""

    def __init__(self, *, sector: Literal["episodic", "semantic", "procedural", "emotional", "reflective"], num: int):
        self.sector = sector
        self.num = max(num, 1)

    @classmethod
    def get_chain(cls):
        """
        获取模型执行链
        :return:
        """
        # 构建输出解析器
        output_parser = PydanticOutputParser(pydantic_object=SectorSentenceOutput)
        # 构建提示词模板
        prompt_template = PromptTemplate(
            template=cls.CREATE_PROMPT,
            input_variables=["sector", "num"],
            partial_variables={
                "sec_descriptions": chr(10).join([f"- {k}: {v.description}" for k, v in SECTOR_CONFIGS.items()]),
                "format_instructions": output_parser.get_format_instructions()
            }
        )
        # 构建语言模型
        from services.memory.components import get_chat_model
        llm = get_chat_model()
        # 构建执行链
        return prompt_template | llm | output_parser

    @timing
    async def create(self) -> SectorSentenceOutput:
        # 调用模型执行链获取分类结果
        chain = self.get_chain()
        output: SectorSentenceOutput = await chain.ainvoke({"sector": self.sector, "num": self.num})
        return output
