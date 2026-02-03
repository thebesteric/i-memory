import functools
import inspect
import time
from typing import Callable

from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


def timing(_func=None, *, func_name: str = None, log_level: str = "INFO", precision: int = 3):
    """
    记录函数执行时间的装饰器，支持同步和异步函数
    :param _func: 被装饰的函数（如果直接调用装饰器）
    :param func_name: 自定义函数名称（用于日志），如果不提供则使用原函数名
    :param log_level: 日志级别，可选值：DEBUG, INFO, WARNING, ERROR
    :param precision: 时间精度（小数位数），默认3位（毫秒级）
    :return: 装饰器函数

    使用示例：
        @log_execution_time()
        def sync_function():
            time.sleep(1)
            return "done"

        @log_execution_time()
        async def async_function():
            await asyncio.sleep(1)
            return "done"

        @log_execution_time(func_name="自定义名称", log_level="DEBUG")
        def my_function():
            pass
    """

    def _format_time(_elapsed_time: float, _precision: int) -> str:
        """格式化时间为人类可读格式"""
        if _elapsed_time < 1:
            time_str = f"{_elapsed_time * 1000:.{_precision}f}ms"
        elif _elapsed_time < 60:
            time_str = f"{_elapsed_time:.{_precision}f}s"
        else:
            minutes = int(_elapsed_time // 60)
            seconds = _elapsed_time % 60
            time_str = f"{minutes}m {seconds:.{_precision}f}s"

        return time_str

    def decorator(func: Callable) -> Callable:
        # 确定函数名称
        display_name = func_name or func.__name__
        # 判断是否为协程函数
        is_coroutine = inspect.iscoroutinefunction(func)

        if is_coroutine:
            # 异步函数装饰器
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed_time = time.perf_counter() - start_time
                    # 格式化时间
                    time_str = _format_time(elapsed_time, precision)
                    # 记录日志
                    log_message = f"函数 '{display_name}' 执行完成，耗时: {time_str}"
                    getattr(logger, log_level.lower())(log_message)

                    return result
                except Exception as e:
                    elapsed_time = time.perf_counter() - start_time
                    time_str = _format_time(elapsed_time, precision)
                    logger.error(f"函数 '{display_name}' 执行失败，耗时: {time_str}，错误: {str(e)}", exc_info=True)
                    raise

            return async_wrapper
        else:
            # 同步函数装饰器
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed_time = time.perf_counter() - start_time
                    # 格式化时间
                    time_str = _format_time(elapsed_time, precision)
                    # 记录日志
                    log_message = f"函数 '{display_name}' 执行完成，耗时: {time_str}"
                    getattr(logger, log_level.lower())(log_message)

                    return result
                except Exception as e:
                    elapsed_time = time.perf_counter() - start_time
                    time_str = _format_time(elapsed_time, precision)
                    logger.error(f"函数 '{display_name}' 执行失败，耗时: {time_str}，错误: {str(e)}", exc_info=True)
                    raise

            return sync_wrapper

    # 如果是 @timing 这种用法，_func 就是被装饰的函数，直接返回包装后的函数
    if _func is not None:
        return decorator(_func)

    # 如果是 @timing(...) 这种用法，返回真正的装饰器
    return decorator


@timing
def test_timing():
    time.sleep(2)


if __name__ == '__main__':
    test_timing()
