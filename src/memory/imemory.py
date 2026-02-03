from typing import Dict, Any

from src.ai.client.openai_registrar import OpenAIRegistrar
from src.core.db import db
from src.ops.ingest import ingest_document
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


class IMemory:

    def __init__(self, user: str = None):
        self.default_user = user
        db.connect()
        self._openai = OpenAIRegistrar(self)

    @property
    def openai(self):
        return self._openai

    async def add(self, content: str, user_id: str = None, **kwargs) -> Dict[str, Any]:
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


if __name__ == '__main__':
    import asyncio


    async def test_memory():
        mem = IMemory(user="test_user")
        contents = ["你是谁？"]
        for content in contents:
            res = await mem.add(content, cfg={"force_root": False}, meta={"source": "unit_test"}, tags=["test", "memory"])
            print("Memory added:", res)


    asyncio.run(test_memory())
