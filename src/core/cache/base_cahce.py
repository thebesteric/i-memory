from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import time

from utils.time_unit import TimeUnit


class BaseCache(ABC):
    """
    抽象缓存基类
    """

    def __init__(self, default_ttl: Optional[Union[int, float]] = None, time_unit: TimeUnit = TimeUnit.SECONDS) -> None:
        self._default_ttl: Optional[float] = None if default_ttl is None else time_unit.to_seconds(default_ttl)

    def _resolve_ttl(self, ttl: Optional[Union[int, float]], time_unit: TimeUnit = TimeUnit.SECONDS) -> Optional[float]:
        """
        将传入的 ttl 解析为以秒为单位的 float 或 None

        规则：
        - 如果 ttl 为 None -> 返回实例的 default_ttl（可能为 None）。
        - 否则使用 _to_seconds 将 ttl 和 ttl_unit 转为秒；若解析后为 None 或 <= 0 -> 视为永久不过期（返回 None）。
        """
        if ttl is None:
            return self._default_ttl
        return time_unit.to_seconds(ttl)

    def _expiry_timestamp(self, ttl: Optional[Union[int, float]], time_unit: TimeUnit = TimeUnit.SECONDS) -> Optional[float]:
        """
        返回绝对过期时间戳，若不失效则返回 None
        :param ttl: TTL（按照 ttl_unit 指定的单位）
        :param time_unit: TTL 的时间单位
        :return:
        """
        resolved = self._resolve_ttl(ttl, time_unit)
        if resolved is None:
            return None
        return time.time() + resolved

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[Union[int, float]] = None, time_unit: TimeUnit = TimeUnit.SECONDS) -> None:
        """
        使用可选的 ttl（配合 time_unit 指定单位）存储 key 对应的值
        :param key: 键
        :param value: 值
        :param ttl: TTL 的数值
        :param time_unit: TTL 的单位
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取 key 的值；如果不存在或已过期则返回 default
        :param key: 键
        :param default: 默认值
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        删除缓存中的指定 key（若存在）
        :param key: 键
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """
        清空缓存中的所有项
        :return:
        """
        raise NotImplementedError()
