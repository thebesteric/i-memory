import asyncio
import unittest

from src.memory.graph.fact_extractor import FactExtractor
from src.memory.graph.semantic_spliter import Topic
from src.memory.graph.graph_models import Fact


class TestSemanticSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fact_extract = FactExtractor()

    def test_extract(self):
        topic = Topic(
            name="OpenClaw定义与核心定位",
            summary="OpenClaw是一个本地部署的AI智能体执行框架，核心能力是真实任务执行而非仅限对话；其定位为本地开源、模型无关、多渠道接入、强执行能力的AI智能体。",
            keywords=["OpenClaw", "AI智能体", "执行框架", "本地部署", "任务执行", "模型无关", "多渠道接入", "强执行"],
            dialogue_ids=["1", "2", "34"]
        )
        fact: Fact = asyncio.run(self.fact_extract.invoke(topic=topic))
        print(fact)