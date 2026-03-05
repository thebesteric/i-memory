from typing import List, Any

from agile_commons.utils import LogHelper
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, ConfigDict

from src.core.vector.milvus.milvus_manager import MilvusManager

logger = LogHelper.get_logger()


class MilvusRetriever(BaseRetriever):
    """
    基于 Milvus 的 LangChain Retriever
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    milvus_manager: MilvusManager = Field(..., description="Milvus 客户端管理器")
    reranker: Any | None = Field(None, description="可选的 reranker 模型")
    collection_name: str = Field(..., description="Milvus 集合名称")
    top_k: int = Field(5, description="搜索结果数量")

    def __init__(self, *, milvus_manager: MilvusManager, collection_name: str, top_k: int = 5, reranker: Any = None, **kwargs):
        super().__init__(
            milvus_manager=milvus_manager,
            collection_name=collection_name,
            reranker=reranker,
            **kwargs
        )
        self.top_k = top_k

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        """
        同步检索接口
        :param query:
        :param run_manager:
        :return:
        """
        # 调用 Milvus 搜索
        logger.info(f"Starting sync search in collection {self.collection_name} with query: {query}")
        search_results = self.milvus_manager.sync_search(
            collection_name=self.collection_name,
            query=query,
            top_k=self.top_k
        )

        if not search_results or not search_results[0]:
            return []

        if self.reranker:
            filter_result = self.reranker.compress_documents(documents=search_results, query=query)
            logger.info("Reranker filtered results:", filter_result)
            return filter_result

        # 确保返回的文档包含 ID 信息
        for doc in search_results:
            if "doc_id" not in doc.metadata:
                # 如果没有 doc_id，尝试从其他属性获取
                if hasattr(doc, "id") and doc.id:
                    doc.metadata["doc_id"] = doc.id

        return search_results

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun = None) -> list[Document]:
        """
        异步检索接口
        :param query:
        :param run_manager:
        :return:
        """
        # 调用 Milvus 异步搜索
        logger.info(f"Starting async search in collection {self.collection_name} with query: {query}")
        search_results = await self.milvus_manager.async_search(
            collection_name=self.collection_name,
            query=query,
            top_k=self.top_k
        )

        if not search_results or not search_results[0]:
            return []

        if self.reranker:
            filter_result = self.reranker.compress_documents(documents=search_results, query=query)
            print("Reranker filtered results:", filter_result)
            return filter_result

        # 确保返回的文档包含 ID 信息
        for doc in search_results:
            if "doc_id" not in doc.metadata:
                # 如果没有 doc_id，尝试从其他属性获取
                if hasattr(doc, "id") and doc.id:
                    doc.metadata["doc_id"] = doc.id

        return search_results
