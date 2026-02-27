from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils.log_helper import LogHelper

from src.ai.model_provider import get_chat_model
from src.core.config import env

logger = LogHelper.get_logger()


class ExtractEssence:
    """
    从文本中提取摘要内容
    """

    EXTRACT_ESSENCE_PROMPT = """
你是一个内容摘要提取专家。你的任务是从给定的文本中提取最重要和最有价值的信息，生成一个简洁的摘要。

提取时请遵循以下优先级规则：
1. 事件信息：日期、时间、地点、人物、具体发生的事情
2. 关键数据：金额、数字、具体数据、统计信息
3. 结论和观点：重要的结论、决定、学习要点
4. 背景信息：为理解上述信息所需的背景

要求：
- 保留最重要的细节和具体信息
- 删除冗余、重复和不必要的修饰
- 使用原文的表述方式，保持信息的完整性
- 输出的长度不超过 {max_len} 个字符
- 按原文的逻辑顺序组织摘要内容
- 不添加任何额外的分析或解释

原始文本：
{content}

请直接返回摘要内容，不要添加任何前缀或说明。"""

    def __init__(self, *, content: str, max_len: int = None):
        self.content = content
        self.max_len = max_len or env.SUMMARY_MAX_LENGTH

    def get_chain(self):
        """
        获取模型执行链
        :return: langchain chain 对象
        """
        # 构建输出解析器
        output_parser = StrOutputParser()
        # 构建提示词模板
        prompt_template = PromptTemplate(
            template=self.EXTRACT_ESSENCE_PROMPT,
            input_variables=["content"],
            partial_variables={
                "max_len": self.max_len
            }
        )
        # 构建语言模型
        llm = get_chat_model()
        # 构建执行链
        return prompt_template | llm | output_parser

    async def extract(self) -> str:
        """
        使用大模型异步提取文本的精华内容
        :return: 提取后的摘要文本
        """
        # 如果环境变量 env.USE_SUMMARY_ONLY 为假，或原文长度不超过 max_len，直接返回原文，不做摘要
        if not env.USE_SUMMARY_ONLY or len(self.content) <= self.max_len:
            return self.content

        try:
            # 使用大模型提取精华内容
            chain = self.get_chain()
            essence = await chain.ainvoke({"content": self.content})
            return essence.strip()
        except Exception as e:
            logger.error(f"Error during extraction: {e}. Falling back to original content slicing.", exc_info=True)
            return self.content[:self.max_len]
