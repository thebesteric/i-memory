from typing import Dict, Any, List

from src.ai.client.openai_registrar import OpenAIRegistrar
from src.core.db import get_db
from src.memory.hsg import hsg_query
from src.memory.memory_filters import MemoryFilters
from src.ops.ingest import ingest_document
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()
db = get_db()


class IMemory:

    def __init__(self, user: str = None):
        self.default_user = user
        db.connect()
        self._openai = OpenAIRegistrar(self)

    @property
    def openai(self):
        return self._openai

    async def add(self, content: str, user_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        添加记忆内容
        :param content: 记忆内容文本
        :param user_id: 用户标识
        :param kwargs: 其他参数
        :return: 添加结果
        """
        uid = user_id or self.default_user
        # 处理文档，标记文档类型、文档内容，元数据、用户标识、标签
        res = await ingest_document(content_type="text",
                                    user_id=uid,
                                    data=content,
                                    cfg=kwargs.get("cfg"),
                                    meta=kwargs.get("meta"),
                                    tags=kwargs.get("tags"))
        if "root_memory_id" in res:
            res["id"] = res["root_memory_id"]
        return res

    async def search(self, query: str, user_id: str = None, limit: int = 10, filters: MemoryFilters = None) -> List[Dict[str, Any]]:
        """
        搜索记忆内容
        :param query: 查询文本
        :param user_id: 用户标识
        :param limit: 返回结果数量限制
        :param filters: 过滤条件
        :return: 搜索结果列表
        """
        uid = user_id or self.default_user
        # 创建 MemoryFilters 对象
        if not filters:
            filters = MemoryFilters(user_id=uid)

        return await hsg_query(query, limit, filters)


if __name__ == '__main__':
    import asyncio


    async def test_add_memory():
        mem = IMemory(user="test_user")
        contents = [
            "昨天我带着我的猫去宠物医院做了例行检查，医生说它的健康状况很好，没有任何问题。",
            "我喜欢给我的猫买各种玩具，尤其是那些可以动的玩具，它总是玩得不亦乐乎。",
        ]
        for content in contents:
            res = await mem.add(content, cfg={"force_root": False}, meta={"source": "unit_test"}, tags=["test", "memory"])
            print("Memory added:", res)


    asyncio.run(test_add_memory())

    async def test_search_memory():
        mem = IMemory(user="test_user")
        query = "我家的猫是什么品种？"
        results = await mem.search(query, limit=5)
        print("Search results:", results)


    # asyncio.run(test_search_memory())
