import datetime
import uuid
from typing import cast

from agile.utils import LogHelper
from sqlalchemy import asc, desc, select

from services.memory.components import USER_IDENTITY_CACHE
from services.commons.encrypt_service import decrypt_if_necessary, encrypt_if_necessary, encryption_enabled
from infra.db.engine import get_session_factory
from infra.db.orm_models import Users
from domain.memory.models import IMemoryUserIdentity, IMemoryUser
from shared.utils.encrypt_utils import EncryptionKeyTool

logger = LogHelper.get_logger()


def load_user_encryption_keys(session, user_ids: list[str]) -> dict[str, str]:
    """批量加载用户加密密钥，供仓储层透明加解密复用。"""
    if not encryption_enabled() or not user_ids:
        return {}
    rows = session.execute(
        select(Users.id, Users.encryption_key).where(Users.id.in_(user_ids))
    ).all()
    return {str(uid): encryption_key for uid, encryption_key in rows if uid and encryption_key}


def _to_memory_user(user_entity: Users) -> IMemoryUser:
    summary = decrypt_if_necessary(
        user_entity.summary,
        key_b64=user_entity.encryption_key,
        aad={"id": user_entity.id},
    )
    return IMemoryUser(
        id=user_entity.id,
        tenant_key=user_entity.tenant_key,
        project_key=user_entity.project_key,
        user_key=user_entity.user_key,
        encryption_key=user_entity.encryption_key,
        summary=summary,
        reflection_count=user_entity.reflection_count or 0,
        created_at=user_entity.created_at,
        updated_at=user_entity.updated_at,
    )


def _apply_order_by(query, order_by: list[str] | None):
    if not order_by:
        return query

    order_columns = []
    for item in order_by:
        clause = (item or "").strip()
        if not clause:
            continue
        parts = clause.split()
        field = parts[0]
        direction = parts[1].lower() if len(parts) > 1 else "asc"
        column = Users.__table__.columns.get(field)
        if column is None:
            logger.warning(f"Ignore unknown order_by field: {field}")
            continue
        order_columns.append(desc(column) if direction == "desc" else asc(column))

    if order_columns:
        query = query.order_by(*order_columns)
    return query


async def find_user(
        *,
        ids: list[str] | None = None,
        order_by: list[str] | None = None,
        status: int | None = None,
        limit=9999,
        offset=0,
) -> list[IMemoryUser]:
    """
    查询所有用户
    :param ids: 用户 ID 列表
    :param order_by: 排序调节
    :param status: 用户状态，0 = 禁用，1 = 启用，None = 全部
    :param limit: 数量限制
    :param offset: 偏移量
    :return:
    """
    query = select(Users)
    if status is not None:
        query = query.where(Users.status == status)
    if ids:
        query = query.where(Users.id.in_(ids))
    query = _apply_order_by(query, order_by)
    query = query.limit(limit).offset(offset)

    session_factory = get_session_factory()
    with session_factory() as session:
        users = session.execute(query).scalars().all()
    return [_to_memory_user(u) for u in users]


async def get_user(user_identity: IMemoryUserIdentity, using_cache: bool = False) -> IMemoryUser | None:
    """
    获取用户信息
    :param user_identity: 用户身份
    :param using_cache: 是否使用缓存
    :return:
    """
    user_key = user_identity.user_key
    tenant_key = user_identity.tenant_key
    project_key = user_identity.project_key

    # 构建缓存 key
    cache_key = f"{user_key}:{tenant_key or ''}:{project_key or ''}"
    if using_cache:
        memory_user = USER_IDENTITY_CACHE.get(cache_key)
        if memory_user:
            return memory_user

    query = select(Users).where(Users.user_key == user_key)
    if tenant_key:
        query = query.where(Users.tenant_key == tenant_key)
    if project_key:
        query = query.where(Users.project_key == project_key)

    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.execute(query).scalars().first()
    memory_user = _to_memory_user(user) if user else None

    # 加入缓存
    USER_IDENTITY_CACHE.set(cache_key, memory_user)

    # 返回用户
    return memory_user


async def get_user_by_id(_id: str) -> IMemoryUser | None:
    """
    根据 ID 获取用户信息
    :param _id: 用户 ID
    :return:
    """
    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.execute(select(Users).where(Users.id == _id)).scalars().first()
    return _to_memory_user(user) if user else None


async def add_user(user_identity: IMemoryUserIdentity, summary: str | None = None, reflection_count: int = 0) -> IMemoryUser:
    """
    添加用户
    :param user_identity: 用户身份
    :param summary: 用户概要信息
    :param reflection_count: 反思次数
    :return: 用户
    """
    user_key = user_identity.user_key
    tenant_key = user_identity.tenant_key
    project_key = user_identity.project_key

    summary = summary if summary else "User profile initializing..."
    reflection_count = reflection_count if reflection_count is not None else 0

    now = datetime.datetime.now()

    session_factory = get_session_factory()
    with session_factory() as session:
        existing_user_query = select(Users).where(Users.user_key == user_key)
        existing_user_query = (
            existing_user_query.where(Users.tenant_key == tenant_key)
            if tenant_key is not None
            else existing_user_query.where(Users.tenant_key.is_(None))
        )
        existing_user_query = (
            existing_user_query.where(Users.project_key == project_key)
            if project_key is not None
            else existing_user_query.where(Users.project_key.is_(None))
        )
        existing_user = session.execute(existing_user_query).scalars().first()
        if existing_user:
            return _to_memory_user(existing_user)

        user_entity = Users(
            id=str(uuid.uuid4()),
            tenant_key=cast(str, tenant_key),
            project_key=cast(str, project_key),
            user_key=user_key,
            encryption_key=EncryptionKeyTool.generate_aes_256_gcm_key(),
            reflection_count=reflection_count,
            created_at=now,
            updated_at=now,
        )
        user_entity.summary = encrypt_if_necessary(
            summary,
            key_b64=user_entity.encryption_key,
            aad={"id": user_entity.id},
        )
        session.add(user_entity)
        session.commit()
        return _to_memory_user(user_entity)


async def update_user_summary(_id: str, summary: str) -> None:
    """
    更新用户概要信息
    :param _id: 主键
    :param summary: 用户概要信息
    :return:
    """
    now = datetime.datetime.now()
    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.get(Users, _id)
        if not user:
            return
        user_key_b64 = getattr(user, "encryption_key", None)
        user_id = getattr(user, "id", _id)
        user.summary = encrypt_if_necessary(
            summary,
            key_b64=user_key_b64,
            aad={"id": user_id},
        )
        user.updated_at = now
        session.commit()
