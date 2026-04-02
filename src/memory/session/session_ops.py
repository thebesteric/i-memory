import datetime
import json
import uuid

from agile.utils import LogHelper

from src.core.db import get_db
from src.memory.memory_models import IMemoryUser
from src.memory.session.session_models import Sessions

logger = LogHelper.get_logger()
db = get_db()


async def insert_sessions(user: IMemoryUser, sessions: Sessions, conn=None) -> Sessions:
    for session in sessions.sessions:
        now = datetime.datetime.now()
        _id = str(uuid.uuid4())
        session.set_id(_id)
        session.set_user_id(user.id)

        db.execute(
            """
            INSERT INTO sessions (id, user_id, summary, dialogue_ids, key_facts, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                _id,
                user.id,
                session.summary,
                json.dumps(session.dialogue_ids or [], ensure_ascii=False),
                json.dumps(session.key_facts or [], ensure_ascii=False),
                now,
                now
            ),
            conn=conn
        )

    if conn is None:
        db.commit()
    return sessions


async def mark_memoires_to_session_joined(m_ids: list[str], conn=None) -> int:
    """
    将记忆标记为已参与会话构建
    :param m_ids:
    :param conn:
    :return:
    """
    if not m_ids:
        return 0
    format_strings = ','.join(['%s'] * len(m_ids))
    query = f"UPDATE memories SET session_joined = 1 WHERE id IN ({format_strings})"
    affected_rows = db.execute(
        query,
        tuple(m_ids),
        conn=conn
    )
    if conn is None:
        db.commit()
    return affected_rows
