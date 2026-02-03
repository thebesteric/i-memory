import unittest

from src.memory.hsg import compute_simhash, compute_hamming_distance


class TestSimHash(unittest.TestCase):

    def setUp(self):
        print("开始执行测试方法...")


    def tearDown(self):
        print("测试方法执行结束。\n")

    # # 新增：汉明距离计算方法（SimHash相似度核心，测试类内复用）
    # def calc_hamming_distance(self, hash1, hash2):
    #     """
    #     计算两个SimHash值的汉明距离（二进制位不同的个数）
    #     :param hash1: 第一个SimHash值（int/str类型，需保证类型一致）
    #     :param hash2: 第二个SimHash值
    #     :return: 汉明距离（int）
    #     """
    #     # 若SimHash返回字符串，先转为整数（二进制哈希值）
    #     if isinstance(hash1, str):
    #         hash1 = int(hash1, 2)
    #     if isinstance(hash2, str):
    #         hash2 = int(hash2, 2)
    #     # 异或运算：相同位为0，不同位为1 → 统计1的个数即为汉明距离
    #     xor_result = hash1 ^ hash2
    #     return bin(xor_result).count('1')

    def test_simhash(self):
        # 场景1：完全相同的文本，预期SimHash值完全一致
        text_same1 = "今天天气真好"
        text_same2 = "今天天气真好"
        simhash_same1 = compute_simhash(text_same1)
        simhash_same2 = compute_simhash(text_same2)
        hamming_sim = compute_hamming_distance(simhash_same1, simhash_same2)
        self.assertEqual(hamming_sim, 0, "相同文本的SimHash值应完全相等")

        # 场景2：高度相似的文本（如你的示例），预期 SimHash 值高度相似（需结合汉明距离校验）
        text_sim1 = "今天天气真好"
        text_sim2 = "今天天气不错"
        simhash_sim1 = compute_simhash(text_sim1)
        simhash_sim2 = compute_simhash(text_sim2)
        hamming_sim = compute_hamming_distance(simhash_sim1, simhash_sim2)
        self.assertLess(hamming_sim, 25, "高度相似文本的汉明距离应小于阈值（如10）")

        # 场景3：完全无关的文本，预期 SimHash 值差异大（汉明距离大）
        text_diff1 = "今天天气真好"
        text_diff2 = "宇宙探索发现新的星系"
        simhash_diff1 = compute_simhash(text_diff1)
        simhash_diff2 = compute_simhash(text_diff2)
        hamming_diff = compute_hamming_distance(simhash_diff1, simhash_diff2)
        self.assertGreater(hamming_diff, 25, "无关文本的汉明距离应大于阈值（如30）")



if __name__ == '__main__':
    unittest.main()