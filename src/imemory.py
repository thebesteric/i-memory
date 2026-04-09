import asyncio
from typing import Dict, Any, List

from agile.db.vector.milvus.milvus_manager import MilvusManager
from agile.utils import LogHelper, singleton
from agile.web import PagingResponse
from pymilvus import Collection

from src.ai.client.openai_registrar import OpenAIRegistrar
from src.core import user_ops
from src.core.components import get_milvus_manager
from src.core.config import env
from src.core.db import get_db
from src.core.mem_ops import mem_ops
from src.exceptions.exceptions import UserNotFoundError
from src.memory.hsg import query_hsg_memories
from src.memory.memory_models import IMemoryConfig, IMemoryFilters, IMemoryUserIdentity, QARole, IMemoryUser, IMemorySearchResult
from src.ops.ingest import ingest_document

logger = LogHelper.get_logger(title="[IMEMORY]")


@singleton
class IMemory:

    def __init__(self, user_identity: IMemoryUserIdentity = None):
        self.default_user_identity: IMemoryUserIdentity = user_identity or IMemoryUserIdentity()
        self.mem_ops = mem_ops
        self._openai = OpenAIRegistrar(self)
        self.db = get_db()
        self.milvus_manager: MilvusManager | None = None
        # 预先准备资源，例如数据库连接、向量数据库集合等
        asyncio.run(self._prepare_resource())

    async def _prepare_resource(self):
        """
        预先准备资源，例如数据库连接、向量数据库集合等
        """
        # 初始化数据库与连接池
        self.db.connect()

        # 初始化向量数据库集合（如果需要）
        logger.info(f"Using Milvus DB for vector storage: {env.VECTOR_MILVUS_SUPPORT}")
        if env.VECTOR_MILVUS_SUPPORT is True:
            self.milvus_manager = get_milvus_manager()
            # 检查集合是否存在，不存在则创建
            await self.milvus_manager.ensure_collection_ready()
            collection: Collection = await self.milvus_manager.get_collection(env.MILVUS_COLLECTION_NAME)
            if collection.num_entities == 0:
                # TODO: 从数据库读取向量数据并插入到 Milvus 中
                pass

    @property
    def openai(self):
        return self._openai

    async def add(self,
                  content: str,
                  user_identity: IMemoryUserIdentity = None,
                  cfg: IMemoryConfig = None,
                  tags: List[str] = None,
                  meta: Dict[str, Any] = None,
                  qa_role: QARole | None = None) -> Dict[str, Any]:
        """
        添加记忆内容
        :param content: 记忆内容文本
        :param user_identity: 用户标识
        :param cfg: 记忆配置
        :param tags: 标签列表
        :param meta: 其他元数据
        :param qa_role: QA 角色（human/assistant）
        :return: 添加结果
        """
        user_identity = user_identity or self.default_user_identity
        # 处理文档，标记文档类型、文档内容，元数据、用户标识、标签
        res = await ingest_document(content_type="text",
                                    user_identity=user_identity,
                                    data=content,
                                    cfg=cfg,
                                    meta=meta,
                                    tags=tags,
                                    qa_role=qa_role)
        if "root_memory_id" in res:
            res["id"] = res["root_memory_id"]
        return res

    async def search(self,
                     query: str,
                     *,
                     limit: int | None = None,
                     filters: IMemoryFilters = None) -> IMemorySearchResult:
        """
        搜索记忆内容
        :param query: 查询文本
        :param limit: 至少要返回的结果数量
        :param filters: 过滤条件
        :return: 搜索结果列表
        """
        search_limit: int = limit or 10

        # 创建 MemoryFilters 对象
        if not filters:
            filters = IMemoryFilters(user_identity=self.default_user_identity)
        # 执行查询
        return await query_hsg_memories(query, search_limit, filters)

    async def get(self, memory_id: str) -> Dict[str, Any] | None:
        """
        获取记忆内容
        :param memory_id: 记忆标识
        :return: 记忆内容
        """
        return self.mem_ops.get_mem(memory_id)

    async def delete(self, memory_id: str) -> int:
        """
        删除记忆内容
        :param memory_id: 记忆标识
        :return: 删除结果
        """
        return self.mem_ops.del_mem(memory_id)

    async def clear(self, user_identity: IMemoryUserIdentity = None) -> int:
        """
        清除用户所有记忆内容
        :param user_identity: 用户身份
        :return:
        """
        user = await self._get_user_by_identity(user_identity or self.default_user_identity)
        return self.mem_ops.del_mem_by_user(user)

    async def history(self,
                      *,
                      user_identity: IMemoryUserIdentity = None,
                      current: int | None = None,
                      size: int | None = None) -> PagingResponse:
        """
        获取用户记忆历史
        :param user_identity: 用户身份
        :param current: 当前页码
        :param size: 每页大小
        :return: 记忆历史列表
        """
        current = current or 1
        size = size or 10

        user = await self._get_user_by_identity(user_identity or self.default_user_identity)
        total = self.mem_ops.count_mem_by_user(user)
        offset = (current - 1) * size
        rows = self.mem_ops.all_mem_by_user(user, size, offset)
        return PagingResponse(
            records=[dict(r) for r in rows],
            total=total,
            current=current,
            size=size
        )

    @staticmethod
    async def _get_user_by_identity(user_identity: IMemoryUserIdentity) -> IMemoryUser:
        user = await user_ops.get_user(user_identity)
        if not user:
            raise UserNotFoundError(user_identity)
        return user
