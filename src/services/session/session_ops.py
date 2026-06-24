import datetime
import uuid
from typing import Any

from agile.utils import LogHelper, timing
from sqlalchemy import Text, cast, literal, select, update

from services.memory.components import get_embed_model
from infra.db.engine import get_session_factory
from infra.db.repos.user_repo import load_user_encryption_keys
from infra.db.orm_models import Memories, Sessions as SessionEntity
from domain.memory.models import IMemoryUser
from domain.session.models import Session, SessionCollection
from services.commons.encrypt_service import decrypt_if_necessary, encrypt_if_necessary
from shared.utils.json_utils import coerce_json_field

logger = LogHelper.get_logger()
embed_model = get_embed_model()


def _normalize_list_payload(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parsed = coerce_json_field(value, [])
        return parsed if isinstance(parsed, list) else []
    return []


def _resolve_user_encryption_key(orm_session, user: IMemoryUser) -> str | None:
    if user.encryption_key:
        return user.encryption_key
    if not user.id:
        return None
    key_cache = load_user_encryption_keys(orm_session, [str(user.id)])
    return key_cache.get(str(user.id))


async def insert_sessions(user: IMemoryUser, sessions: SessionCollection, conn=None) -> SessionCollection:
    embed_model = get_embed_model()
    external_session = conn is not None
    orm_session = conn
    if orm_session is None:
        session_factory = get_session_factory()
        orm_session = session_factory()
    try:
        key_b64 = _resolve_user_encryption_key(orm_session, user)
        for session in sessions.sessions:
            now = datetime.datetime.now()
            # 摘要向量化
            vector = await embed_model.embed(session.summary)

            _id = str(uuid.uuid4())
            encrypted_summary = encrypt_if_necessary(session.summary, key_b64=key_b64, aad={"id": _id})
            session.set_id(_id)
            session.set_user_id(user.id)
            session.set_vector(vector)

            orm_session.add(
                SessionEntity(  # type: ignore[arg-type]
                    id=_id,
                    user_id=user.id,
                    summary=encrypted_summary,
                    vector=vector,
                    dialogue_ids=session.dialogue_ids or [],
                    key_facts=session.key_facts or [],
                    created_at=now,
                    updated_at=now,
                )
            )
        if not external_session:
            orm_session.commit()
    finally:
        if not external_session:
            orm_session.close()
    return sessions


@timing
async def session_search(user: IMemoryUser, query: str, top_k: int = 5) -> SessionCollection:
    if top_k <= 0:
        return SessionCollection(sessions=[])

    query_vector = await embed_model.embed(query)
    distance_expr = SessionEntity.vector.op("<=>")(query_vector)
    similarity_expr = (literal(1.0) - distance_expr).label("similarity")
    query_stmt = (
        select(
            SessionEntity.id,
            SessionEntity.user_id,
            SessionEntity.summary,
            cast(SessionEntity.vector, Text).label("vector"),
            SessionEntity.dialogue_ids,
            SessionEntity.key_facts,
            similarity_expr,
        )
        .where(SessionEntity.user_id == user.id)
        .order_by(distance_expr)
        .limit(top_k)
    )

    session_factory = get_session_factory()
    with session_factory() as orm_session:
        key_b64 = _resolve_user_encryption_key(orm_session, user)
        rows = orm_session.execute(query_stmt).mappings().all()

    sessions: list[Session] = []
    for row in rows:
        session = Session(
            summary=decrypt_if_necessary(row["summary"], key_b64=key_b64, aad={"id": row["id"]}),
            dialogue_ids=[str(v) for v in _normalize_list_payload(row.get("dialogue_ids"))],
            key_facts=[str(v) for v in _normalize_list_payload(row.get("key_facts"))],
        )
        session.set_id(row["id"])
        session.set_user_id(row["user_id"])
        session.set_similarity(float(row.get("similarity", 0.0)))

        vector_value = row.get("vector")
        vector_value = coerce_json_field(vector_value, None)
        if isinstance(vector_value, (list, tuple)):
            session.set_vector(list(vector_value))

        sessions.append(session)

    return SessionCollection(sessions=sessions)


async def mark_memoires_to_session_joined(m_ids: list[str], conn=None) -> int:
    """
    将记忆标记为已参与会话构建
    :param m_ids:
    :param conn:
    :return:
    """
    if not m_ids:
        return 0
    stmt = update(Memories).where(Memories.id.in_(m_ids)).values(session_joined=True)
    external_session = conn is not None
    orm_session = conn
    if orm_session is None:
        session_factory = get_session_factory()
        orm_session = session_factory()
    try:
        result = orm_session.execute(stmt)
        if not external_session:
            orm_session.commit()
        return int(getattr(result, "rowcount", 0) or 0)
    finally:
        if not external_session:
            orm_session.close()
