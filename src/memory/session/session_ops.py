import datetime
import json
import uuid
from typing import Any

from agile.utils import LogHelper, timing

from src.core.components import get_embed_model
from src.core.db import get_db
from src.memory.memory_models import IMemoryUser
from src.memory.session.session_models import Session, Sessions

logger = LogHelper.get_logger()
db = get_db()
embed_model = get_embed_model()


def _normalize_list_payload(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


async def insert_sessions(user: IMemoryUser, sessions: Sessions, conn=None) -> Sessions:
    embed_model = get_embed_model()

    for session in sessions.sessions:
        now = datetime.datetime.now()
        # 摘要向量化
        vector = await embed_model.embed(session.summary)

        _id = str(uuid.uuid4())
        session.set_id(_id)
        session.set_user_id(user.id)
        session.set_vector(vector)

        db.execute(
            """
            INSERT INTO sessions (id, user_id, summary, vector, dialogue_ids, key_facts, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                _id,
                user.id,
                session.summary,
                vector,
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

@timing
async def session_search(user: IMemoryUser, query: str, top_k: int = 5) -> Sessions:
    if top_k <= 0:
        return Sessions(sessions=[])

    query_vector = await embed_model.embed(query)

    query_vector_str = str(query_vector)
    rows = db.fetchall(
        """
        SELECT id,
               user_id,
               summary,
               vector::text                AS vector,
               dialogue_ids,
               key_facts,
               1 - (vector <=> %s::vector) AS similarity
        FROM sessions
        WHERE user_id = %s
        ORDER BY vector <=> %s::vector
        LIMIT %s
        """,
        (query_vector_str, user.id, query_vector_str, top_k)
    )

    sessions: list[Session] = []
    for row in rows:
        session = Session(
            summary=row["summary"],
            dialogue_ids=[str(v) for v in _normalize_list_payload(row.get("dialogue_ids"))],
            key_facts=[str(v) for v in _normalize_list_payload(row.get("key_facts"))],
        )
        session.set_id(row["id"])
        session.set_user_id(row["user_id"])
        session.set_similarity(float(row.get("similarity", 0.0)))

        vector_value = row.get("vector")
        if isinstance(vector_value, str):
            try:
                vector_value = json.loads(vector_value)
            except Exception:
                vector_value = None
        if isinstance(vector_value, (list, tuple)):
            session.set_vector(list(vector_value))

        sessions.append(session)

    return Sessions(sessions=sessions)


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
