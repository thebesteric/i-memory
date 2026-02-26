import asyncio
import unittest

from src.imemory import IMemory

from src.memory.models.memory_models import IMemoryUserIdentity, IMemoryFilters


# @unittest.skip("Skipping TestIMemory")
class TestIMemoryQuery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.user_identity: IMemoryUserIdentity = IMemoryUserIdentity(
            user_id="test_user",
            tenant_id="test_tenant",
            project_id="test_project"
        )
        cls.mem = IMemory(user_identity=cls.user_identity)

    @classmethod
    def tearDownClass(cls):
        pass

    # @unittest.skip
    def test_search_memory(self):
        query = "我家的是什么猫？"
        results = asyncio.run(self.mem.search(query, limit=5, filters=IMemoryFilters(user_identity=self.user_identity)))
        print("Search results:", results)

    # @unittest.skip
    def test_get_memory(self):
        memory_id = "66965ce7-4195-404a-8e84-0fc5659ff777"
        result = asyncio.run(self.mem.get(memory_id))
        print("Get memory result:", result)

    # @unittest.skip
    def test_delete_memory(self):
        memory_ids = [
            "66965ce7-4195-404a-8e84-0fc5659ff777",
        ]
        for memory_id in memory_ids:
            asyncio.run(self.mem.delete(memory_id))
            print(f"Memory {memory_id} deleted.")

    # @unittest.skip
    def test_clear_memory(self):
        asyncio.run(self.mem.clear(user_identity=self.user_identity))

    # @unittest.skip
    def test_history_memory(self):
        response = asyncio.run(self.mem.history(user_identity=self.user_identity, current=1, size=2))
        print("Memory history:", response.model_dump())


if __name__ == '__main__':
    unittest.main(verbosity=2)
