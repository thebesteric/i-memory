import asyncio
import datetime

from agile.utils import LogHelper

from infra.db.repos import user_repo
from shared.config.settings import env
from infra.db.engine import get_session_factory
from infra.db.repos.memory_repo import mem_ops
from domain.memory.models import IMemoryUser
from services.session import session_ops
from services.session.session_extractor import SessionExtractor
from domain.session.models import SessionCollection

logger = LogHelper.get_logger(title="[SESSION_BUILD]")
session_factory = get_session_factory()

# 进程内按用户维度的锁映射表，用于避免为同一用户重复构建用户画像。
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            _SESSION_LOCKS[user_id] = lock
        return lock


async def _process_session(user: IMemoryUser, yesterday_end: datetime.datetime, semaphore: asyncio.Semaphore) -> bool:
    user_lock = await _get_user_lock(str(user.id))
    if user_lock.locked():
        logger.info(f"[SESSION_BUILD] Skip reentrant user session build, User ID: {user.id}")
        return False

    async with user_lock:
        async with semaphore:
            session_extractor = SessionExtractor()
            memories = await asyncio.to_thread(mem_ops.find_mem_by_conditions,
                                               conditions=["user_id = %s", "session_joined = 0", "created_at < %s"],
                                               params=[user.id, yesterday_end], order_by=["created_at ASC"], )

            memories_at_least = max(env.SESSION_BUILD_AT_LEAST or 10, 10)
            if not memories or len(memories) < memories_at_least:
                logger.info(
                    f"[SESSION_BUILD] No memory found for User ID: {user.id} at {yesterday_end.strftime("%Y-%m-%d %H:%M:%S")} or less than {memories_at_least}")
                return False

            sessions: SessionCollection = await session_extractor.invoke(memories=memories)
            with session_factory() as db_session:
                with db_session.begin():
                    sessions = await session_ops.insert_sessions(user, sessions, conn=db_session)
                    affected_rows = await session_ops.mark_memoires_to_session_joined([m["id"] for m in memories],
                                                                                      conn=db_session)
                    logger.info(
                        f"[SESSION_BUILD] User session created, User ID: {user.id}, Associated memories: {affected_rows}")
            return True


async def session_build():
    """
    会话总结创建
    :return:
    """
    concurrency = max(1, env.SESSION_BUILD_THREADS or 5)
    now = datetime.datetime.now()
    # 获取昨天的 23:59:59
    yesterday = now - datetime.timedelta(days=1)
    yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)

    # 查询所有用户
    users: list[IMemoryUser] = await user_repo.find_user(status=1)
    if not users:
        logger.info("[SESSION_BUILD] No active user found.")
        return

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_process_session(user, yesterday_end, semaphore) for user in users]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.exception(f"[SESSION_BUILD] Failed to create session, User ID: {users[idx].id}, Error: {result}")
            continue
        if result:
            success_count += 1

    logger.info(
        f"[SESSION_BUILD] Session created finished, total_users: {len(users)}, success_users: {success_count}, failed_users: {sum(isinstance(r, Exception) for r in results)}")
