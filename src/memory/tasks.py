from dataclasses import dataclass, field
from typing import Any, Callable

from agile.utils import LogHelper
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio

from src.core.config import env
from src.memory.graph import graph
from src.memory.hsg import decay

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
    # IntervalTrigger 的执行间隔，单位：秒
    seconds: int
    # 调度执行时传给 func 的关键字参数
    kwargs: dict[str, Any] = field(default_factory=dict)
    # 同一个任务允许的最大并发实例数，避免重复堆积执行
    max_instances: int = 1
    # 是否合并错过的多次触发，只执行一次补偿运行
    coalesce: bool = True
    # 错过触发后的宽限时间，超时后该次执行会被丢弃
    misfire_grace_time: int = 30


def _build_job_definitions() -> list[JobDefinition]:
    return [
        # 记忆衰减任务
        JobDefinition(
            id="decay",
            name="Periodic memory decay",
            func=decay.apply_decay,
            seconds=max(1, int(getattr(env, "DECAY_INTERVAL_SECONDS", 60 * 5) or 60 * 5)),
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        ),
        # 图构建任务
        JobDefinition(
            id="graph",
            name="Memory graph build",
            func=graph.graph_build,
            # seconds=max(1, int(getattr(env, "GRAPH_BUILD_INTERVAL_SECONDS", 60 * 30) or 60 * 30)),
            seconds=3000,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        ),
        # 每日强制图化任务
        JobDefinition(
            id="force_graph",
            name="Daily force graph build for cold users",
            func=graph.daily_force_graph_build,
            seconds=60*60*24,  # TODO：每24小时执行一次，要能定义到具体的时间
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
        )
    ]


def _scheduler_listener(event: JobExecutionEvent):
    if event.exception:
        logger.error(f"[TASKS] Job failed: id={event.job_id}", exc_info=event.exception)
    elif event.code == EVENT_JOB_EXECUTED:
        logger.debug(f"[TASKS] Job executed: id={event.job_id}")


def _register_jobs(scheduler: AsyncIOScheduler):
    for job in _build_job_definitions():
        scheduler.add_job(
            job.func,
            trigger=IntervalTrigger(seconds=job.seconds),
            id=job.id,
            name=job.name,
            kwargs=job.kwargs,
            replace_existing=True,
            max_instances=job.max_instances,
            coalesce=job.coalesce,
            misfire_grace_time=job.misfire_grace_time,
        )
        logger.info(f"[TASKS] Registered job: id={job.id}, interval={job.seconds}s")


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

    # 启动 graph worker 协程（多并发）
    try:
        graph_worker_count = getattr(env, "GRAPH_WORKER_COUNT", 3)
        for i in range(graph_worker_count):
            asyncio.create_task(graph.process_user_identity_queue())
        logger.info(f"[TASKS] Started {graph_worker_count} process_user_identity_queue workers.")
    except Exception as e:
        logger.error(f"[TASKS] Failed to start process_user_identity_queue workers: {e}")

    return scheduler


async def stop_background_tasks(scheduler: AsyncIOScheduler | None):
    if not scheduler:
        return

    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[TASKS] Scheduler stopped")
