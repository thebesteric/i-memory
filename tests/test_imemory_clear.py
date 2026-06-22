import asyncio
import unittest

from services.i_memory import IMemory

from domain.memory.models import IMemoryUserIdentity


# @unittest.skip("Skipping TestIMemory")
class TestIMemoryQuery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.user_identity: IMemoryUserIdentity = IMemoryUserIdentity(
            user_key="test_user",
            tenant_key="test_tenant",
            project_key="test_project"
        )
        cls.mem = IMemory(user_identity=cls.user_identity)

    @classmethod
    def tearDownClass(cls):
        pass

    # @unittest.skip
    def test_delete_memory(self):
        memory_ids = [
            "09adf7da-a854-4197-a579-52fab861ac23",
        ]
        for memory_id in memory_ids:
            asyncio.run(self.mem.delete(memory_id))
            print(f"Memory {memory_id} deleted.")

    # @unittest.skip
    def test_clear_memory(self):
        asyncio.run(self.mem.clear(user_identity=self.user_identity))


if __name__ == '__main__':
    unittest.main(verbosity=2)
