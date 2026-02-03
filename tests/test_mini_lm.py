import asyncio
import unittest

from src.ai.embed.mini_lm_model import MiniLM


class TestMiniLM(unittest.TestCase):

    def setUp(self):
        print("开始执行测试方法...")
        self.mini_lm = MiniLM(mini_lm_model="all-MiniLM-L12-v2")

    def tearDown(self):
        print("测试方法执行结束。\n")

    def test_minilm_embedding(self):
        print("测试 MiniLM 嵌入向量生成...")
        embedding = asyncio.run(self.mini_lm.embedding("今天天气真好"))
        self.assertEqual(len(embedding), 384, "MiniLM 嵌入向量维度应为 384")

    def test_minilm_hash(self):
        print("测试 MiniLM MinHash 签名生成...")
        embedding = asyncio.run(self.mini_lm.embedding("今天天气真好"))
        result = asyncio.run(self.mini_lm.hash(embedding))
        print(result)

    def test_hash_equal(self):
        print("测试 MiniLM MinHash 签名一致性...")
        embedding1 = asyncio.run(self.mini_lm.embedding("今天天气真好"))
        hash1 = asyncio.run(self.mini_lm.hash(embedding1))
        embedding2 = asyncio.run(self.mini_lm.embedding("今天天气真好"))
        hash2 = asyncio.run(self.mini_lm.hash(embedding2))
        self.assertEqual(hash1, hash2, "相同文本的 MinHash 签名应相等")

    def test_similarity(self):
        print("测试 MiniLM 相似度计算...")
        sentences = [
            ("今天天气真好", "今天天气真好"),
            ("今天天气真好", "今天天气不错"),
            ("今天天气真好", "我喜欢吃苹果"),
        ]
        for sent1, sent2 in sentences:
            sim_score = asyncio.run(self.mini_lm.similarity(sent1, sent2))
            print(f"相似度（'{sent1}' vs '{sent2}'）: {sim_score:.4f}")