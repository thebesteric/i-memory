from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.memory import IMemory


class BaseModelRegistrar(ABC):
    """
    模型客户端注册器基类
    """
    def __init__(self, memory_instance: "IMemory"):
        self.mem = memory_instance

    @abstractmethod
    async def async_wrapped_create(self, original_create, user_id, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sync_wrapped_create(self, original_create, user_id, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_async(client: Any) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_original_create(client: Any):
        raise NotImplementedError

    def register(self, client: Any, user_id: str = None):
        try:
            original_create = self.get_original_create(client)
        except AttributeError:
            return client
        # 判断 client 是否异步
        wrapped_create = self.async_wrapped_create if self.is_async(client) else self.sync_wrapped_create
        # 使用 wrapped_create 替换原有的 create 方法
        original_create = wrapped_create
        # 返回客户端
        return client
