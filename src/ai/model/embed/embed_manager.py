import os
from typing import Sequence

import numpy as np
import pyrootutils
import torch
from agile.utils import LogHelper, singleton
from sentence_transformers import SentenceTransformer

logger = LogHelper.get_logger(title="[EMBED]")


class EmbedManager:
    DEFAULT_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Embedding-4B"

    def __init__(self,
                 model_name_or_path: str | os.PathLike[str] = DEFAULT_MODEL_NAME_OR_PATH,
                 cache_dir: str | os.PathLike[str] | None = None):
        self.model_name_or_path = os.fspath(model_name_or_path if model_name_or_path else self.DEFAULT_MODEL_NAME_OR_PATH)
        project_root = pyrootutils.find_root()
        self.cache_dir = os.fspath(cache_dir) if cache_dir else os.path.join(project_root, "assets", "embed", "models", self.model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

        # 初始化时检查模型是否已本地存在
        self._is_model_local = self._check_model_files_exist()
        if self._is_model_local:
            self._model = self.load_model()

        logger.info(f"Embed model initialized: {self.model_name_or_path}, "
                    f"cache_dir={self.cache_dir}, device={self.device}, is_model_local={self._is_model_local}")

    def _check_model_files_exist(self) -> bool:
        """
        检查模型文件是否已存在于本地缓存目录
        :return: 存在返回 True，否则返回 False
        """
        return os.path.exists(self.cache_dir)

    def load_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                model_kwargs = {}
                if self.device.type == "cuda":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                tokenizer_kwargs = {
                    "padding_side": "left",
                }

                # 防止 trust_remote_code 场景下触发 Hub 网络请求
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                self._model = SentenceTransformer(
                    self.model_name_or_path,
                    cache_folder=self.cache_dir,
                    trust_remote_code=True,
                    device=str(self.device),
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                    local_files_only=self._is_model_local
                )
                logger.info(f"Model {self.model_name_or_path} loaded successfully")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load model '{self.model_name_or_path}' from path '{self.cache_dir}'. "
                    f"Error: {exc}"
                ) from exc
        return self._model

    @property
    def model(self) -> SentenceTransformer:
        return self._model or self.load_model()

    async def embed(self, content: str, dim: int = 1536) -> list[float]:
        """
        文本嵌入
        :param content: 文本内容
        :param dim: 输出维度
        :return:
        """
        return self.embed_batch([content], dim=dim)[0]

    async def embed_batch(self, contents: list[str], dim: int = 1536) -> list[list[float]]:
        """
        批量文本嵌入
        :param contents: 文本内容列表
        :param dim: 输出维度
        :return:
        """
        embedding = self.model.encode(contents, normalize_embeddings=True, truncate_dim=dim)
        return embedding.tolist()

    def similarity(self, sentence1: str | Sequence[float], sentence2: str | Sequence[float], dim: int = 1536) -> float:
        """
        比较相似度
        :param sentence1: 文本或向量
        :param sentence2: 文本或向量
        :param dim: 维度
        :return:
        """
        if isinstance(sentence1, str) != isinstance(sentence2, str):
            raise TypeError("similarity 入参类型必须一致（都为文本或都为向量）")
        # 文本类型
        if isinstance(sentence1, str) and isinstance(sentence2, str):
            embeddings = self.embed_batch([sentence1, sentence2], dim=dim)
            return self.cosine_similarity(embeddings[0], embeddings[1])
        # 向量类型
        return self.cosine_similarity(sentence1, sentence2)

    @staticmethod
    def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float], eps: float = 1e-12) -> float:
        arr1 = np.asarray(vec1, dtype=np.float64)
        arr2 = np.asarray(vec2, dtype=np.float64)
        if arr1.shape != arr2.shape:
            raise ValueError("两个向量的维度必须一致")
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 <= eps or norm2 <= eps:
            return 0.0
        sim = float(np.dot(arr1, arr2) / (norm1 * norm2))
        return float(max(min(sim, 1.0), -1.0))
