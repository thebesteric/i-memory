from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from agile.utils import LogHelper
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import asyncio

from src.core.config import env
from src.memory.hsg import decay
from src.memory.graph import graph_builder
from src.memory.session import session_builder
from src.memory.profile import user_profile_builder

logger = LogHelper.get_logger()


@dataclass(frozen=True)
class JobDefinition:
    """
    统一描述一个 APScheduler 定时任务的注册信息
    """

    # 任务 ID
    id: str
    # 任务名称
    name: str
    # 实际执行的任务函数，支持同步或异步 callable
    func: Callable[..., Any]
    # 触发器类型: 'interval' 或 'cron'
    trigger_type: Literal["interval", "cron"] = "interval"
    # interval 时为 seconds，cron 时为 hour, minute 等
    trigger_args: dict[str, Any] = field(default_factory=dict)
    # 调度执行时传给 func 的关键字参数
    kwargs: dict[str, Any] = field(default_factory=dict)
    # 同一个任务允许的最大并发实例数，避免重复堆积执行
    max_instances: int = 1
    # 是否合并错过的多次触发，只执行一次补偿运行
    coalesce: bool = True
    # 错过触发后的宽限时间，超时后该次执行会被丢弃
    misfire_grace_time: int = 30
    # 是否开启任务
    enable: bool = True


def _build_job_definitions() -> list[JobDefinition]:
    return [
        # 记忆衰减任务（每 60 分钟执行一次）
        JobDefinition(
            id="memory_decay",
            name="Periodic memory decay",
            func=decay.apply_decay,
            trigger_type="interval",
            trigger_args={"seconds": max(1, int(getattr(env, "DECAY_INTERVAL_SECONDS", 60 * 60) or 60 * 60))},
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        ),
        # 图构建任务（每 30 分钟执行一次）
        JobDefinition(
            id="graph_build",
            name="Memory graph build",
            func=graph_builder.graph_build,
            trigger_type="interval",
            trigger_args={"seconds": max(1, int(getattr(env, "GRAPH_BUILD_INTERVAL_SECONDS", 60 * 30) or 60 * 30))},
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
            enable=getattr(env, "GRAPH_BUILD_ENABLE", True)
        ),
        # 每日强制图化任务（每天 2:00 执行）
        JobDefinition(
            id="force_graph_build",
            name="Daily force graph build for cold users",
            func=graph_builder.graph_build_daily_force,
            trigger_type="cron",
            trigger_args={"hour": 2, "minute": 0, "second": 0},
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
            enable=getattr(env, "GRAPH_BUILD_ENABLE", True)
        ),
        # 会话总结任务（每天 3:00 执行）
        JobDefinition(
            id="session_build",
            name="Daily session build",
            func=session_builder.session_build,
            trigger_type="cron",
            trigger_args={"hour": 3, "minute": 0, "second": 0},
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
            enable=getattr(env, "SESSION_BUILD_ENABLE", True)
        ),
        # 用户画像任务（每天 5:00 执行）
        JobDefinition(
            id="user_profile",
            name="Describe user profile",
            func=user_profile_builder.describe_user_profile,
            trigger_type="cron",
            trigger_args={"hour": 5, "minute": 0, "second": 0},
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
            enable=getattr(env, "USER_PROFILE_ENABLE", True),
        )
    ]


def _scheduler_listener(event: JobExecutionEvent):
    if event.exception:
        logger.error(f"[TASKS] Job failed: id={event.job_id}", exc_info=event.exception)
    elif event.code == EVENT_JOB_EXECUTED:
        logger.debug(f"[TASKS] Job executed: id={event.job_id}")


def _register_jobs(scheduler: AsyncIOScheduler):
    for job in _build_job_definitions():
        if not job.enable:
            continue
        if job.trigger_type == "interval":
            trigger = IntervalTrigger(**job.trigger_args)
        elif job.trigger_type == "cron":
            trigger = CronTrigger(**job.trigger_args)
        else:
            raise ValueError(f"Unknown trigger_type: {job.trigger_type}")
        scheduler.add_job(
            job.func,
            trigger=trigger,
            id=job.id,
            name=job.name,
            kwargs=job.kwargs,
            replace_existing=True,
            max_instances=job.max_instances,
            coalesce=job.coalesce,
            misfire_grace_time=job.misfire_grace_time,
        )
        logger.info(f"[TASKS] Registered job: id={job.id}, trigger={job.trigger_type}, args={job.trigger_args}")


def start_background_tasks() -> AsyncIOScheduler:
    timezone = "UTC"
    scheduler = AsyncIOScheduler(timezone=timezone)
    scheduler.add_listener(
        callback=_scheduler_listener,
        mask=EVENT_JOB_ERROR | EVENT_JOB_EXECUTED
    )
    _register_jobs(scheduler)
    scheduler.start()
    logger.info(f"[TASKS] Scheduler started, timezone={timezone}")

    # 是否开启图构建
    graph_build_enable = getattr(env, "GRAPH_BUILD_ENABLE", True)
    if graph_build_enable:
        try:
            graph_worker_count = getattr(env, "GRAPH_WORKER_COUNT", 3)
            for i in range(graph_worker_count):
                asyncio.create_task(graph_builder.process_user_queue())
            logger.info(f"[TASKS] Graph Started {graph_worker_count} process_user_queue workers.")
        except Exception as e:
            logger.error(f"[TASKS] Graph Failed to start process_user_queue workers: {e}")

    return scheduler


async def stop_background_tasks(scheduler: AsyncIOScheduler | None):
    if not scheduler:
        return

    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[TASKS] Scheduler stopped")
