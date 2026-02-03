from typing import Iterable, Callable, overload
from pydantic import BaseModel, ConfigDict, Field

from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


class InitTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    func: Callable[[], None] = Field(..., description="任务函数")
    desc: str | None = Field(default=None, description="任务描述")
    lazy: bool = Field(default=False, description="是否延迟执行")
    ran_completed: bool = Field(default=False, description="任务是否已完成")

    def run(self) -> None:
        if self.ran_completed:
            logger.warning("Task has already been completed.")
            return
        self.func()
        self.ran_completed = True

    def __hash__(self):
        return id(self.func)

    def __eq__(self, other):
        if not isinstance(other, InitTask):
            return False
        return self.func is other.func


class TaskInitializer:
    """
    应用初始化器
    """

    _instance = None

    def __init__(self, tasks: Iterable[InitTask] | None = None):
        self.tasks: set[InitTask] = set(tasks) if tasks else set()

    def add_task(self, task: Callable[[], None], *, lazy: bool = False, desc: str = None) -> InitTask:
        """
        添加初始化任务
        :param task: 初始化任务
        :param lazy: 是否延迟执行
        :param desc: 任务描述
        """
        init_task = InitTask(func=task, lazy=lazy, desc=desc)
        self.tasks.add(init_task)
        if not lazy:
            self.run(reload=False)
        return init_task

    @overload
    def task(self, func: Callable[[], None]) -> Callable[[], None]:
        ...

    @overload
    def task(self, *, desc: str = None) -> Callable[[Callable[[], None]], Callable[[], None]]:
        ...

    def task(self, func: Callable[[], None] | None = None, *, desc: str = None):
        """
        装饰器：注册初始化任务
        :param func: 初始化任务函数
        :param desc: 任务描述
        :return: 装饰后的函数或装饰器
        """

        def decorator(f: Callable[[], None]):
            init_task = self.add_task(f, lazy=False, desc=desc)
            return init_task

        if func is not None:
            return decorator(func)
        return decorator

    def run(self, reload: bool = False) -> None:
        """
        运行所有初始化任务
        """
        for init_task in self.tasks:
            if reload or not init_task.ran_completed:
                logger.info("[TASK INITIALIZER] Running init task: %s", init_task.func.__name__)
                init_task.func()
                init_task.ran_completed = True
