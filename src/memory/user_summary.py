import time
import json
import asyncio
from typing import Dict, List

from src.core.config import env
from src.core.db import get_db
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()

db = get_db()

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
                meta = json.loads(m["meta"]) if isinstance(m["meta"], str) else m["meta"]
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
    last_active = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at / 1000)) if created_at else "Recently"

    # 生成概要字符串：Active in 项目名 using 语言. Focused on 文件. (N memories, M saves). Last active: 时间.
    return f"Active in {proj_str} using {lang_str}. Focused on {recent_files}. ({len(memories)} memories, {saves} saves). Last active: {last_active}."


async def gen_user_summary_async(user_id: str) -> str:
    """
    异步获取该用户最近 100 条记忆并生成概要字符串
    """
    rows = db.fetchall("SELECT * FROM memories WHERE user_id=%s ORDER BY created_at DESC LIMIT 100 OFFSET 0", (user_id,))
    return gen_user_summary(rows)


async def update_user_summary(user_id: str):
    """
    用于根据指定用户的记忆（memories）自动生成并更新该用户的概要信息（summary）
    """
    try:
        # 生成用户概要
        # 格式为：Active in 项目名 using 语言. Focused on 文件. (N memories, M saves). Last active: 时间.
        summary = await gen_user_summary_async(user_id)
        now = int(time.time() * 1000)

        # 获取用户
        existing = db.fetchone("SELECT * FROM users WHERE user_id=%s", (user_id,))
        if not existing:
            # 插入新用户记录并设置概要
            db.execute("INSERT INTO users(user_id,summary,reflection_count,created_at,updated_at) VALUES (%s,%s,%s,%s,%s)",
                       (user_id, summary, 0, now, now))
        else:
            # 更新用户概要
            db.execute("UPDATE users SET summary=%s, updated_at=%s WHERE user_id=%s", (summary, now, user_id))
        db.commit()
    except Exception as e:
        logger.error(f"[USER_SUMMARY] Error for {user_id}: {e}")


async def auto_update_user_summaries():
    all_memories = db.fetchall("SELECT user_id FROM memories LIMIT 10000")
    uids = set(m["user_id"] for m in all_memories if m["user_id"])

    updated = 0
    for u in uids:
        await update_user_summary(u)
        updated += 1
    return {"updated": updated}


_timer_task = None


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
