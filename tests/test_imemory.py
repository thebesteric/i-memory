import asyncio
import unittest

from src.imemory import IMemory
from src.memory.models.memory_cfg import IMemoryConfig


# @unittest.skip("Skipping TestIMemory")
class TestIMemory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mem = IMemory(user="test_user")

    @classmethod
    def tearDownClass(cls):
        pass

    # @unittest.skip("Skipping test_add_memory")
    def test_add_memory(self):
        contents = [
            "今天我去了公园，看到很多美丽的花朵和快乐的人们。",
            "我家的猫是一只非常可爱的波斯猫，它有着长长的毛发和温柔的性格。",
            "最近我在学习编程，发现Python是一门非常有趣且强大的语言。",
            "昨天晚上我看了一部电影，剧情非常精彩，让我印象深刻。",
            "我喜欢旅行，探索不同的文化和风景，这让我感到非常充实。",
        ]
        for content in contents:
            res = asyncio.run(self.mem.add(content,
                                           cfg=IMemoryConfig(force_root=False),
                                           meta={"source": "unit_test"},
                                           tags=["test", "memory"]))
            print("Memory added:", res)

    def test_add_long_memory(self):
        content = """
带着项目推进的期许，奔赴北京与项目组汇合，一场深耕细节、凝聚共识的线下对接，就此展开。不同于线上的远程沟通，面对面的交流更能碰撞出思维的火花，也更能精准捕捉每一个需求的核心，高效破解前期推进中的疑点与难点。

抵达后便迅速投入工作，与项目组的伙伴们围坐一堂，从项目整体规划、阶段性目标，到具体执行细节、潜在风险预案，逐一深入探讨。大家各抒己见，坦诚交流，既有对现有方案的优化建议，也有对关键节点的反复推敲，每一个观点的碰撞，都只为让项目落地更顺畅、成果更优质。
我们一同梳理项目推进中的堵点，明确各岗位职责分工，细化后续执行时间表，针对前期线上沟通中模糊的细节，逐一核对、确认，确保每一项要求都清晰可落地、每一个环节都衔接无疏漏。从需求拆解到流程优化，从进度把控到质量保障，每一个话题都围绕项目核心，每一份努力都朝着共同的目标。
线下对接的时光虽紧凑，却收获满满。不仅高效解决了前期积累的问题，更深化了彼此的协作默契，让跨区域配合更具凝聚力。看着大家为了同一个目标并肩探讨、全力以赴的模样，更坚定了我们把项目做好的信心。

此次北京之行，既是一次工作对接，更是一次思维的同频与能力的提升。未来，我们将带着此次沟通达成的共识，各司其职、紧密配合，稳步推进项目每一个环节，不负信任、不负期许，全力以赴交出满意的项目答卷。
        """
        res = asyncio.run(self.mem.add(content,
                                       cfg=IMemoryConfig(force_root=False),
                                       meta={"source": "unit_test"},
                                       tags=["test", "memory"]))
        print("Memory added:", res)

    # @unittest.skip("Skipping test_search_memory")
    def test_search_memory(self):
        query = "我家的猫是什么品种？"
        results = asyncio.run(self.mem.search(query, limit=5))
        print("Search results:", results)

    def test_get_memory(self):
        memory_id = "507ecc2a-8f07-458f-a178-3e3e886a27db"
        result = asyncio.run(self.mem.get(memory_id))
        print("Get memory result:", result)

    def test_delete_memory(self):
        memory_ids = [
            "cead287a-26cb-4f41-a4b1-fd7df776cf2c",
            "12b63067-28c8-4d34-98b3-67776bf9edec",
            "004c599d-aef8-4d57-83ba-8e839d4b0eac",
            "51f6778e-e32d-42d0-bd5e-0115a1cd1fa4",
            "ac972727-e95d-4696-9ca4-eac03a1e552b",
        ]
        for memory_id in memory_ids:
            asyncio.run(self.mem.delete(memory_id))
            print(f"Memory {memory_id} deleted.")

    def test_clear_memory(self):
        asyncio.run(self.mem.clear(user_id="test_user"))
        print("All memories for the user cleared.")

    def test_history_memory(self):
        history = self.mem.history(user_id="test_user", current=1, size=2)
        print("Memory history:", history.model_dump())


if __name__ == '__main__':
    unittest.main(verbosity=2)
