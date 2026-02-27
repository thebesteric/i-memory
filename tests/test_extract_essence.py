import asyncio
import sys
import unittest

from src.core.extract_essence import ExtractEssence

class TestIMemoryExtractEssence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_content = """
今天我在上海遇见了张三，我们在2024-01-15讨论了一个重要的项目。
这个项目涉及1000万元的投资，预计需要3个月完成。
张三提出了三个建议：首先，需要加强团队合作；其次，要确保质量管理；最后，必须控制成本。
我们最终同意了这个方案，并计划下周开始执行。
这是一个很有前景的项目，相信能够取得成功。
"""


    def test_extract(self):
        """测试异步提取"""
        extractor = ExtractEssence(
            content=self.test_content,
            max_len=100
        )
        try:
            result = asyncio.run(extractor.extract())
            print(f"原始文本长度: {len(self.test_content)} 字符")
            print(f"提取摘要长度: {len(result)} 字符")
            print(f"提取结果: {result}")
            return True
        except Exception as e:
            print(f"✗ 异步提取失败: {e}")
            return False

if __name__ == "__main__":
    unittest.main(verbosity=2)
