import logging
import threading
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

# 配置日志：输出时间、线程、日志级别、内容
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MonitorData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    created_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="创建时间"
    )
    thread_name: str = Field(
        default_factory=lambda: threading.current_thread().name,
        description="创建线程名称"
    )
    call_count: int = Field(default=0, description="调用次数")

    def incr_call_count(self) -> None:
        """调用次数自增"""
        self.call_count += 1

    @classmethod
    def create(cls, thread_name: str = None) -> "MonitorData":
        """
        类方法：创建 MonitorData 初始对象
        :param thread_name: 可选，指定线程名
        :return: 初始化完成的 MonitorData 实例
        """
        # 若未传线程名，自动获取当前运行线程的名称（核心简化逻辑）
        if thread_name is None:
            thread_name = threading.current_thread().name
        # 调用类的构造方法，返回初始化对象
        return cls(thread_name=thread_name)


def singleton(cls):
    """
    单例装饰器，用于确保一个类只有一个实例，支持继承。
    :return:
    """
    # 类的实例缓存
    _instances = {}
    # 定义全局线程锁，每个装饰的类独享一把锁
    _lock = threading.Lock()

    # 监控数据：键为类，值为{创建时间、创建线程、调用次数}
    _monitor = dict[type, MonitorData]()

    # 创建一个新类来包装原始类
    class SingletonWrapper(cls):
        def __new__(cls, *args, **kwargs):
            # 检查类是否已创建实例，未创建则初始化并缓存
            if cls not in _instances:
                with _lock:
                    # 双重检查，加锁后再次验证，防止多线程同时等待锁后重复创建
                    if cls not in _instances:
                        _instances[cls] = super(SingletonWrapper, cls).__new__(cls)
                        # 手动调用 __init__，因为 __new__ 返回的实例不会自动调用
                        _instances[cls].__init__(*args, **kwargs)
                        logger.debug(f"[{cls.__name__}] singleton instance created.")
                        # 初始化监控数据
                        _monitor[cls] = MonitorData.create()

            # 每次调用都增加调用计数
            if cls in _monitor:
                _monitor[cls].incr_call_count()
            # 返回缓存的唯一实例
            return _instances[cls]

    # 暴露监控数据，支持外部查询
    def monitor_info() -> dict[str, Any]:
        """获取单例的监控信息"""
        if cls in _monitor:
            return {
                "class_name": cls.__name__,
                **_monitor[cls].model_dump()
            }
        # 如果当前类没有监控数据，尝试查找原始类的监控数据
        if cls.__name__ != "SingletonWrapper" and hasattr(cls, "__orig_class__"):
            return monitor_info.__get__(None, cls.__orig_class__)()
        return {}

    def destroy():
        """
        销毁单例实例，释放持有的资源
        为装饰器添加显式销毁方法，挂载在 SingletonWrapper 类上
        """
        if cls in _instances:
            # 若实例有 __del__ 方法，先执行资源释放逻辑
            if hasattr(_instances[cls], "__del__"):
                _instances[cls].__del__()
            # 从缓存中删除实例
            del _instances[cls]
            logger.debug(f"[{cls.__name__}] singleton instance destroyed.")

        # 同时删除监控数据
        if cls in _monitor:
            del _monitor[cls]

    # 绑定方法到 SingletonWrapper 类
    SingletonWrapper.monitor_info = monitor_info
    SingletonWrapper.destroy = destroy

    # 复制原始类的属性
    SingletonWrapper.__name__ = cls.__name__
    SingletonWrapper.__doc__ = cls.__doc__
    SingletonWrapper.__module__ = cls.__module__
    # 保存原始类的引用
    SingletonWrapper.__orig_class__ = cls

    return SingletonWrapper
