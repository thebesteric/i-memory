import asyncio
import unittest

from src.imemory import IMemory

from src.memory.models.memory_models import IMemoryUserIdentity, IMemoryFilters, IMemoryItemInfo


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

    @unittest.skip
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
