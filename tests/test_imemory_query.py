import asyncio
import unittest

from src.imemory import IMemory
from src.memory.models.memory_cfg import IMemoryConfig


# @unittest.skip("Skipping TestIMemory")
class TestIMemoryQuery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.user_id: str = "test_user"
        cls.mem = IMemory(user=cls.user_id)

    @classmethod
    def tearDownClass(cls):
        pass

    # @unittest.skip
    def test_search_memory(self):
        query = "我去哪里开会了？"
        results = asyncio.run(self.mem.search(query, user_id=self.user_id, limit=5))
        print("Search results:", results)

    # @unittest.skip
    def test_get_memory(self):
        memory_id = "507ecc2a-8f07-458f-a178-3e3e886a27db"
        result = asyncio.run(self.mem.get(memory_id))
        print("Get memory result:", result)

    # @unittest.skip
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

    # @unittest.skip
    def test_history_memory(self):
        response = self.mem.history(user_id="test_user", current=1, size=2)
        print("Memory history:", response.model_dump())


if __name__ == '__main__':
    unittest.main(verbosity=2)
