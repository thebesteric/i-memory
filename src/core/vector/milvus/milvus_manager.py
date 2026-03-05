import asyncio
import uuid
from typing import List, Set

from agile_commons.utils import LogHelper
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema, Collection, connections, utility, SearchResult

from src.ai.embed.base_embed_model import BaseEmbedModel
from src.ai.model_provider import get_embed_model
from src.core.config import env
from src.core.constants import COMPONENTS_CACHE

logger = LogHelper.get_logger()


class MilvusManager:
    """
    Milvus 数据库管理器
    """

    def __init__(
            self,
            *,
            embedding_model: BaseEmbedModel,
            primary_field: str = "id",
            text_field: str = "text",
            vector_field: str = "vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params_nlist: int = 128,
            params_nprobe: int = 10,
            vector_dim: int = 1536,
            search_timeout: float = 30.0
    ):
        # 初始化嵌入模型
        self.embedding_model = embedding_model

        # 属性字段
        self.primary_field = primary_field
        self.text_field = text_field
        self.vector_field = vector_field

        # 参数字段
        self.index_type = index_type
        self.metric_type = metric_type
        self.params_nlist = params_nlist
        self.params_nprobe = params_nprobe
        self.vector_dim = vector_dim
        self.search_timeout = search_timeout

        # 缓存集合：已初始化的 collection（已检查存在性）
        self._initialized_collections: Set[str] = set()
        # 缓存集合：已加载到内存的 collection
        self._loaded_collections: Set[str] = set()

        # 索引参数
        self.index_params = {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": {"nlist": self.params_nlist}
        }

        # 搜索参数
        self.search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": self.params_nprobe}
        }

        # 初始化 Milvus 客户端 pymilvus
        self.milvus_client = MilvusClient(
            uri=env.MILVUS_URL,
            token=env.MILVUS_TOKEN
        )

        # 建立连接
        connections.connect(
            alias="default",
            uri=env.MILVUS_URL,
            token=env.MILVUS_TOKEN
        )

    def _create_collection_schema(self, fields: List[FieldSchema] = None) -> CollectionSchema:
        """
        创建集合模式
        :return: CollectionSchema 对象
        """
        # 定义字段
        fields = [
            FieldSchema(name=self.primary_field, dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
            FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ] if not fields else fields

        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="Milvus collection for document storage and similarity search"
        )
        return schema

    def _ensure_collection_exists(self, collection_name: str):
        """
        确保集合存在，如果不存在则创建
        :param collection_name: 集合名称
        :return:
        """
        # 使用缓存避免重复检查
        if collection_name in self._initialized_collections:
            return

        if not utility.has_collection(collection_name):
            # 创建集合
            schema = self._create_collection_schema()
            collection = Collection(
                name=collection_name,
                schema=schema
            )
            # 创建索引
            collection.create_index(
                field_name=self.vector_field,
                index_params=self.index_params
            )
            logger.info(f"Created new collection: {collection_name}")

        # 标记为已初始化
        self._initialized_collections.add(collection_name)

    async def ensure_collection_ready(self, collection_name: str):
        """
        预初始化 collection：确保存在、已加载、已就绪
        应在应用启动时调用，而不是在每次查询时调用
        :param collection_name: 集合名称
        :return:
        """
        try:
            # 确保集合存在
            self._ensure_collection_exists(collection_name)

            # 获取集合对象并加载到内存
            collection = Collection(name=collection_name)
            if collection_name not in self._loaded_collections:
                collection.load()
                self._loaded_collections.add(collection_name)
                logger.info(f"Collection '{collection_name}' loaded into memory")

            logger.info(f"Collection '{collection_name}' is ready")
        except Exception as e:
            logger.error(f"Failed to ensure collection '{collection_name}' ready: {str(e)}")
            raise e

    def _get_collection(self, collection_name: str) -> Collection:
        """
        获取 collection 对象（优化版本：使用缓存）
        :param collection_name: 集合名称
        :return: Collection 对象
        """
        # 如果已初始化，直接获取（不再检查是否存在）
        if collection_name in self._initialized_collections:
            collection = Collection(name=collection_name)
            # 如果未加载，则加载（通常第一次查询时需要）
            if collection_name not in self._loaded_collections:
                collection.load()
                self._loaded_collections.add(collection_name)
            return collection

        # 如果未初始化，回退到旧逻辑（兼容性）
        self._ensure_collection_exists(collection_name)
        collection = Collection(name=collection_name)
        if collection_name not in self._loaded_collections:
            collection.load()
            self._loaded_collections.add(collection_name)
        return collection

    async def insert_documents(self, collection_name: str, documents: List[Document]):
        """
        插入文档到 Milvus 集合
        :param collection_name: 集合名称
        :param documents: 文档列表
        :return:
        """
        try:
            # 确保集合存在（数据导入时可能需要创建）
            self._ensure_collection_exists(collection_name)
            # 获取集合对象
            collection = self._get_collection(collection_name)

            # 准备数据
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # 生成嵌入向量
            embeddings = await self.embedding_model.aembed_documents(texts)

            # 准备插入数据
            entities = [
                doc_ids,
                embeddings,
                texts,
                metadatas
            ]

            # 插入数据
            collection.insert(entities)
            collection.flush()  # 刷新确保数据持久化

            logger.info(f"Successfully inserted {len(documents)} documents into collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to insert documents into Milvus: {str(e)}")
            raise e

    async def async_search(self, collection_name: str, query: str, top_k: int = 5, timeout: float = None) -> list[Document]:
        """
        在 Milvus 集合中进行异步相似度搜索
        :param collection_name: 集合名称
        :param query: 查询语句
        :param top_k: 返回数量
        :param timeout: 超时时间（秒）
        :return: 搜索结果
        """
        try:
            # 设置超时时间
            timeout = timeout or self.search_timeout
            # 获取集合对象
            collection = self._get_collection(collection_name)
            # 生成查询嵌入向量
            query_embedding = await self.embedding_model.aembed_query(query)

            def _do_search():
                """同步搜索函数，在线程池中执行"""
                return collection.search(
                    data=[query_embedding],
                    anns_field=self.vector_field,
                    param=self.search_params,
                    limit=top_k,
                    output_fields=["*"],
                    timeout=timeout
                )

            # 使用 asyncio.to_thread 运行同步函数
            results = await asyncio.to_thread(_do_search)

            # 处理搜索结果
            documents = self._package_documents(results)

            logger.info(f"Async search completed, found {len(documents)} results")
            return documents

        except asyncio.TimeoutError:
            logger.error(f"Milvus search timeout after {timeout}s for query: {query[:50]}...")
            # 返回空结果而不是抛出异常，避免影响整体流程
            return []
        except Exception as e:
            logger.error(f"Failed to perform async search in Milvus: {str(e)}", exc_info=True)
            return []

    def sync_search(self, collection_name: str, query: str, top_k: int = 5, timeout: float = None) -> list[Document]:
        """
        在 Milvus 集合中进行同步相似度搜索
        :param collection_name: 集合名称
        :param query: 查询语句
        :param top_k: 返回数量
        :param timeout: 超时时间（秒）
        :return: 搜索结果
        """
        try:
            # 设置超时时间
            timeout = timeout or self.search_timeout
            # 获取集合对象
            collection = self._get_collection(collection_name)
            # 生成查询嵌入向量
            query_embedding = self.embedding_model.embed_query(query)

            # 执行搜索
            results = collection.search(
                data=[query_embedding],
                anns_field=self.vector_field,
                param=self.search_params,
                limit=top_k,
                output_fields=["*"],
                timeout=timeout
            )

            # 处理搜索结果
            documents = self._package_documents(results)

            logger.info(f"Sync search completed, found {len(documents)} results")
            return documents

        except Exception as e:
            logger.error(f"Failed to perform sync search in Milvus: {str(e)}")
            return []

    def _package_documents(self, results: SearchResult) -> list[Document]:
        """
        将 Milvus 搜索结果包装为 Document 对象
        :param results: 搜索结果
        :return: 包装后的文档列表
        """
        documents = []
        for hits in results:
            for hit in hits:
                primary_key = hit.entity.get(self.primary_field)
                doc = Document(
                    id=primary_key,
                    page_content=hit.entity.get(self.text_field),
                    metadata={
                        self.primary_field: primary_key,
                        "score": hit.distance,
                        **hit.entity.get("metadata", {})
                    }
                )
                documents.append(doc)
        return documents

    async def delete_collection(self, collection_name: str):
        """
        删除集合
        :param collection_name: 集合名称
        :return:
        """
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                # 从缓存中移除
                self._initialized_collections.discard(collection_name)
                self._loaded_collections.discard(collection_name)
                logger.info(f"Successfully deleted collection: {collection_name}")
            else:
                logger.warning(f"Collection {collection_name} does not exist, cannot delete.")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            raise e

    async def list_collections(self):
        """
        列出所有集合
        :return: 集合名称列表
        """
        try:
            collections = utility.list_collections()
            logger.info(f"Current collections in Milvus: {collections}")
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections in Milvus: {str(e)}")
            raise e


def get_milvus_manager() -> MilvusManager:
    """
    获取 MilvusManager 实例
    :return: MilvusManager 实例
    """
    milvus_manager = COMPONENTS_CACHE.get(MilvusManager.__name__)
    if not milvus_manager:
        milvus_manager = MilvusManager(
            embedding_model=get_embed_model(),
            vector_dim=env.VECTOR_DIM
        )
        COMPONENTS_CACHE.set(MilvusManager.__name__, milvus_manager)
    return milvus_manager
