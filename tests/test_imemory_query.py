import asyncio
import unittest

from src.imemory import IMemory

from src.memory.models.memory_models import IMemoryUserIdentity, IMemoryFilters, IMemoryItemInfo


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
    def test_search_memory(self):
        query = "聊聊北京之行"
        results: list[IMemoryItemInfo] = asyncio.run(self.mem.search(
            query,
            limit=10,
            filters=IMemoryFilters(
                user_identity=self.user_identity,
                query_mode="prefer",
            )
        ))
        for result in results:
            print(f"Result: content={result.content}, score={result.score}")

    # @unittest.skip
    def test_get_memory(self):
        memory_id = "66965ce7-4195-404a-8e84-0fc5659ff777"
        result = asyncio.run(self.mem.get(memory_id))
        print("Get memory result:", result)

    # @unittest.skip
    def test_history_memory(self):
        response = asyncio.run(self.mem.history(user_identity=self.user_identity, current=1, size=2))
        print("Memory history:", response.model_dump())


if __name__ == '__main__':
    unittest.main(verbosity=2)
