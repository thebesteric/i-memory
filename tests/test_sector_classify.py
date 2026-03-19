import asyncio
import unittest

from src.core.components import get_sector_classifier
from src.core.sector_classify import ClassifyResult


class TestSector_classify(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sector_classifier = get_sector_classifier()


    def test_sector_classify(self):
        content = "在海边看日落，心中感受到一种无言的美好与宁静。"
        result: ClassifyResult = asyncio.run(self.sector_classifier.classify(content=content))
        print(result.model_dump())