import math
from typing import Dict, Any, List

from utils.log_helper import LogHelper
from utils.singleton import singleton
from web.paging import PagingResponse

from src.ai.client.openai_registrar import OpenAIRegistrar
from src.core.db import get_db
from src.core.dml_ops import dml_ops
from src.memory.hsg import hsg_query
from src.memory.models.memory_models import IMemoryConfig, IMemoryFilters, IMemoryUserIdentity, IMemoryItemInfo
from src.ops.ingest import ingest_document

logger = LogHelper.get_logger()
db = get_db()


@singleton
class IMemory:

    def __init__(self, user_identity: IMemoryUserIdentity = None):
        self.default_user_identity: IMemoryUserIdentity = user_identity or IMemoryUserIdentity()
        self.dml_ops = dml_ops
        db.connect()
        self._openai = OpenAIRegistrar(self)

    @property
    def openai(self):
        return self._openai

    async def add(self, content: str, user_identity: IMemoryUserIdentity = None, cfg: IMemoryConfig = None, tags: List[str] = None,
                  meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        添加记忆内容
        :param content: 记忆内容文本
        :param user_identity: 用户标识
        :param cfg: 记忆配置
        :param tags: 标签列表
        :param meta: 其他元数据
        :return: 添加结果
        """
        user_identity = user_identity or self.default_user_identity
        # 处理文档，标记文档类型、文档内容，元数据、用户标识、标签
        res = await ingest_document(content_type="text",
                                    user_identity=user_identity,
                                    data=content,
                                    cfg=cfg,
                                    meta=meta,
                                    tags=tags)
        if "root_memory_id" in res:
            res["id"] = res["root_memory_id"]
        return res

    async def search(self, query: str, *, limit: int = 10, filters: IMemoryFilters = None) -> List[IMemoryItemInfo]:
        """
        搜索记忆内容
        :param query: 查询文本
        :param limit: 至少要返回的结果数量
        :param filters: 过滤条件
        :return: 搜索结果列表
        """
        # 创建 MemoryFilters 对象
        if not filters:
            filters = IMemoryFilters(user_identity=self.default_user_identity)

        return await hsg_query(query, limit, filters)

    async def get(self, memory_id: str) -> Dict[str, Any] | None:
        """
        获取记忆内容
        :param memory_id: 记忆标识
        :return: 记忆内容
        """
        return self.dml_ops.get_mem(memory_id)

    async def delete(self, memory_id: str) -> int:
        """
        删除记忆内容
        :param memory_id: 记忆标识
        :return: 删除结果
        """
        return self.dml_ops.del_mem(memory_id)

    async def clear(self, user_identity: IMemoryUserIdentity = None) -> int:
        """
        清除用户所有记忆内容
        :param user_identity: 用户身份
        :return:
        """
        user_identity = user_identity or self.default_user_identity
        return self.dml_ops.del_mem_by_user(user_identity)

    async def history(self, *, user_identity: IMemoryUserIdentity = None, current: int = 1, size: int = 10) -> PagingResponse:
        """
        获取用户记忆历史
        :param user_identity: 用户身份
        :param current: 当前页码
        :param size: 每页大小
        :return: 记忆历史列表
        """
        user_identity = user_identity or self.default_user_identity
        total = self.dml_ops.count_mem_by_user(user_identity)
        offset = (current - 1) * size
        rows = self.dml_ops.all_mem_by_user(user_identity, size, offset)
        return PagingResponse(
            records=[dict(r) for r in rows],
            total=total,
            current=current,
            size=size
        )
