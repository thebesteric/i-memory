import asyncio
from typing import Any, Dict, List

from agile.utils import LogHelper
from sqlalchemy import desc, select

from src.core.config import env
from src.core.db import get_session_factory
from src.core import user_ops
from src.entity.db_schema import Memories
from src.memory.memory_models import IMemoryUser
from src.utils.json_utils import coerce_json_field

logger = LogHelper.get_logger()


def _memory_to_row(memory: Memories) -> Dict[str, Any]:
    return {k: v for k, v in vars(memory).items() if not k.startswith("_")}


def gen_user_summary(memories: List[Dict]) -> str:
    if not memories:
        return "User profile initializing... (No memories recorded yet)"

    # 用集合收集涉及的项目名，避免重复
    projects = set()
    # 用集合收集涉及的编程语言
    languages = set()
    # 用集合收集涉及的文件名（只取文件名部分）
    files = set()
    # 统计“保存”事件的次数
    saves = 0
    # 统计总事件数
    events = 0

    for m in memories:
        # 将每条记忆转为字典
        d = dict(m)
        # 读取 meta 字段
        if d.get("meta"):
            try:
                meta = coerce_json_field(m.get("meta"), {})
                if not isinstance(meta, dict): meta = {}
                # 收集项目名
                if meta.get("ide_project_name"): projects.add(meta["ide_project_name"])
                # 收集编程语言
                if meta.get("language"): languages.add(meta["language"])
                # 收集文件名（只取文件名部分）
                if meta.get("ide_file_path"):
                    files.add(meta["ide_file_path"].replace("\\", "/").split("/")[-1])
                # 统计保存事件
                if meta.get("ide_event_type") == "save": saves += 1
            except:
                pass
        # 每处理一条记忆，events + 1
        events += 1

    # 项目名集合转字符串，若无则为 “Unknown Project”
    proj_str = ", ".join(projects) if projects else "Unknown Project"
    # 编程语言集合转字符串，若无则为 “General”
    lang_str = ", ".join(languages) if languages else "General"
    # 取最多 3 个文件名，若无则为 “various files”
    recent_files = ", ".join(list(files)[:3]) if files else "various files"

    # 取第一条记忆的时间戳，格式化为日期时间字符串
    created_at = memories[0]["created_at"]
    # 格式化最后活跃时间
    last_active = created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else "Recently"

    # 生成概要字符串：Active in 项目名 using 语言. Focused on 文件. (N memories, M saves). Last active: 时间.
    return f"Active in {proj_str} using {lang_str}. Focused on {recent_files}. ({len(memories)} memories, {saves} saves). Last active: {last_active}."


async def gen_user_summary_async(user: IMemoryUser) -> str:
    """
    异步获取该用户最近 100 条记忆并生成概要字符串
    """
    query = (
        select(Memories)
        .where(Memories.user_id == user.id)
        .order_by(desc(Memories.created_at))
        .limit(100)
        .offset(0)
    )
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = [_memory_to_row(item) for item in session.execute(query).scalars().all()]
    return gen_user_summary(rows)


async def update_user_summary(user: IMemoryUser):
    """
    用于根据指定用户的记忆（memories）自动生成并更新该用户的概要信息（summary）
    :param user: 用户
    """
    try:
        # 生成用户概要
        # 格式为：Active in 项目名 using 语言. Focused on 文件. (N memories, M saves). Last active: 时间.
        summary = await gen_user_summary_async(user)
        # 更新用户概要
        await user_ops.update_user_summary(user.id, summary)
    except Exception as e:
        logger.error(f"[USER_SUMMARY] Error for {user.id}: {e}")


async def auto_update_user_summaries():
    # 查询所有用户
    users = await user_ops.find_user(status=1)
    updated = 0
    for user in users:
        await update_user_summary(user)
        updated += 1
    return {"updated": updated}


_timer_task: asyncio.Task | None = None


async def user_summary_loop():
    interval = (env.user_summary_interval or 30) * 60
    while True:
        try:
            await auto_update_user_summaries()
        except Exception as e:
            logger.error(f"[USER_SUMMARY] Loop error: {e}")
        await asyncio.sleep(interval)


def start_user_summary_reflection():
    global _timer_task
    if _timer_task: return
    _timer_task = asyncio.create_task(user_summary_loop())


def stop_user_summary_reflection():
    global _timer_task
    if _timer_task:
        _timer_task.cancel()
        _timer_task = None
