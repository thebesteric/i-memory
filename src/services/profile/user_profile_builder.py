import asyncio
import datetime

from agile.utils import LogHelper

from infra.db.repos import user_repo
from shared.config.settings import env
from infra.db.engine import get_session_factory
from infra.db.repos.memory_repo import mem_ops
from domain.memory.models import IMemoryUser
from services.profile import user_profile_ops
from services.profile.user_profile_extractor import UserProfileExtractor
from domain.profile.models import UserProfile

logger = LogHelper.get_logger(title="[USER_PROFILE_BUILD]")
session_factory = get_session_factory()

# 进程内按用户维度的锁映射表，用于避免为同一用户重复构建用户画像。
_USER_PROFILE_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        lock = _USER_PROFILE_LOCKS.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            _USER_PROFILE_LOCKS[user_id] = lock
        return lock


async def _process_user_profile(user: IMemoryUser, yesterday_end: datetime.datetime,
                                semaphore: asyncio.Semaphore) -> bool:
    """
    构建用户画像
    :param user: 用户
    :param yesterday_end: 记忆提取时间
    :param semaphore: 异步并发控制器
    :return:
    """
    user_lock = await _get_user_lock(str(user.id))
    if user_lock.locked():
        logger.info(f"Skip reentrant user profile build, User ID: {user.id}")
        return False

    async with user_lock:
        async with semaphore:
            user_profile_extractor = UserProfileExtractor()
            memories = await asyncio.to_thread(mem_ops.find_mem_by_conditions,
                                               conditions=["user_id = %s", "profile_joined = 0", "created_at < %s"], params=[user.id, yesterday_end],
                                               order_by=["created_at ASC"], )

            memories_at_least = max(env.USER_PROFILE_AT_LEAST or 10, 10)
            if not memories:
                logger.info(f"No memory found for User ID: {user.id} at {yesterday_end.strftime("%Y-%m-%d %H:%M:%S")} or less than {memories_at_least}")
                return False

            user_profile: UserProfile = await user_profile_extractor.invoke(user, memories=memories)
            with session_factory() as db_session:
                with db_session.begin():
                    user_profile = await user_profile_ops.upsert_user_profile(user, user_profile, conn=db_session)
                    affected_rows = await user_profile_ops.mark_memoires_to_profile_joined([m["id"] for m in memories],
                                                                                           conn=db_session)
                    logger.info(f"User profile updated, User ID: {user.id}, Profile ID: {user_profile.id}, Associated memories: {affected_rows}")
            return True


async def describe_user_profile(user_ids: list[str] | None = None):
    """
    用户画像（由 jobs 的 user_profile 定时任务调用）

    :param user_ids: 用户 ID 列表，通常由用户主动调用时传入，定时任务调用时为 None
    :return:
    """
    concurrency = max(1, env.USER_PROFILE_THREADS or 5)
    now = datetime.datetime.now()
    # 获取昨天的 23:59:59
    yesterday = now - datetime.timedelta(days=1)
    yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)

    if user_ids:
        # 查指定用户
        users: list[IMemoryUser] = await user_repo.find_user(ids=user_ids, status=1, limit=9999)
    else:
        # 查询所有用户
        users: list[IMemoryUser] = await user_repo.find_user(status=1, limit=9999)

    if not users:
        logger.info("No active user found.")
        return

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_process_user_profile(user, yesterday_end, semaphore) for user in users]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.exception(f"Failed to describe user profile, User ID: {users[idx].id}, Error: {result}")
            continue
        if result:
            success_count += 1

    logger.info(f"User profile describe finished, "
                f"total_users: {len(users)}, "
                f"success_users: {success_count}, "
                f"failed_users: {sum(isinstance(r, Exception) for r in results)}")
