from abc import abstractmethod, ABC
from typing import List

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.core.config import env
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


class BaseEmbedModel(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @abstractmethod
    def __init__(self, /, **data):
        super().__init__(**data)
        self.dim = env.VEC_DIM or 1536

    @abstractmethod
    async def embed(self, text: str, model: str = None, dim: int = None) -> List[float]:
        """
        生成单个向量
        :param text: 文本
        :param model: 模型名称
        :param dim: 向量维度
        :return: 嵌入向量
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str], model: str = None, dim: int = None) -> List[List[float]]:
        """
        批量生成向量
        :param texts: 文本列表
        :param model: 模型名称
        :param dim: 向量维度
        :return: 嵌入向量列表
        """
        pass

    @staticmethod
    def similarity(vec1: List[float], vec2: List[float], eps: float = 1e-12) -> float:
        """
        计算两个嵌入向量之间的余弦相似度
        :param vec1: 第一个向量向量
        :param vec2: 第二个向量向量
        :param eps: 防止除零的微小值
        :return: 相似度值
        """
        # 转换为numpy数组
        arr1 = np.array(vec1, dtype=np.float64)
        arr2 = np.array(vec2, dtype=np.float64)
        # 校验维度
        if arr1.shape != arr2.shape:
            raise ValueError("两个向量的维度必须一致")

        # 向量化计算点积和范数
        dot_product = np.dot(arr1, arr2)
        # L2范数
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 <= eps or norm2 <= eps:
            return 0.0

        sim = dot_product / (norm1 * norm2)
        return float(max(min(sim, 1.0), -1.0))
