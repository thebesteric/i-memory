import datetime
import uuid
from typing import Any

from agile.utils import LogHelper
from sqlalchemy import desc, select, update

from services.memory.components import USER_PROFILE_CACHE
from services.commons.encrypt_service import decrypt_if_necessary, encrypt_if_necessary
from infra.db.engine import get_session_factory
from infra.db.orm_models import Memories, UserProfiles
from infra.db.repos.user_repo import load_user_encryption_keys
from domain.memory.models import IMemoryUser
from domain.profile.models import UserProfile

logger = LogHelper.get_logger()

_JSON_ENCRYPT_PREFIX = "json_enc::"


def _resolve_user_encryption_key(orm_session, user: IMemoryUser) -> str | None:
    if user.encryption_key:
        return user.encryption_key
    key_cache = load_user_encryption_keys(orm_session, [str(user.id)])
    return key_cache.get(str(user.id))


def _json_encrypt_values(value: Any, *, key_b64: str | None, aad_base: dict[str, Any], path: str = "") -> Any:
    if isinstance(value, dict):
        return {
            k: _json_encrypt_values(v, key_b64=key_b64, aad_base=aad_base, path=f"{path}.{k}" if path else str(k))
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _json_encrypt_values(item, key_b64=key_b64, aad_base=aad_base, path=f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]

    if isinstance(value, str):
        encrypted = encrypt_if_necessary(
            value,
            key_b64=key_b64,
            aad={**aad_base, "path": path},
        )
        if encrypted != value:
            return f"{_JSON_ENCRYPT_PREFIX}{encrypted}"
    return value


def _json_decrypt_values(value: Any, *, key_b64: str | None, aad_base: dict[str, Any], path: str = "") -> Any:
    if isinstance(value, dict):
        return {
            k: _json_decrypt_values(v, key_b64=key_b64, aad_base=aad_base, path=f"{path}.{k}" if path else str(k))
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _json_decrypt_values(item, key_b64=key_b64, aad_base=aad_base, path=f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]

    if isinstance(value, str) and value.startswith(_JSON_ENCRYPT_PREFIX):
        return decrypt_if_necessary(
            value[len(_JSON_ENCRYPT_PREFIX):],
            key_b64=key_b64,
            aad={**aad_base, "path": path},
        )
    return value


def _decrypt_user_profile_row(row: dict[str, Any], *, key_b64: str | None) -> dict[str, Any]:
    profile_id = str(row.get("id") or "")
    user_id = str(row.get("user_id") or "")

    for field in ("demographic", "preferences", "attributes", "tags"):
        row[field] = _json_decrypt_values(
            row.get(field),
            key_b64=key_b64,
            aad_base={"id": profile_id, "user_id": user_id, "field": field},
        )

    return row


def _profile_entity_to_row(profile: UserProfiles) -> dict:
    return {k: v for k, v in vars(profile).items() if not k.startswith("_")}


async def get_user_profile(user: IMemoryUser, query_cache: bool = False) -> UserProfile:
    """
    获取当前用户画像
    :param user: 用户
    :param query_cache: 是否查询缓存
    :return:
    """
    if query_cache:
        # 查询缓存
        user_profile: UserProfile = USER_PROFILE_CACHE.get(user.id)
        if user_profile:
            # 返回
            return user_profile

    session_factory = get_session_factory()
    with session_factory() as session:
        key_b64 = _resolve_user_encryption_key(session, user)
        entity = session.execute(
            select(UserProfiles)
            .where(UserProfiles.is_active.is_(True), UserProfiles.user_id == user.id)
            .order_by(desc(UserProfiles.updated_at))
            .limit(1)
        ).scalars().first()
    row = _profile_entity_to_row(entity) if entity else None
    if row:
        row = _decrypt_user_profile_row(row, key_b64=key_b64)
    # 字段转 UserProfile 对象
    user_profile = UserProfile.from_dict(row)
    # 加入缓存
    USER_PROFILE_CACHE.set(user.id, user_profile)
    # 返回
    return user_profile


async def upsert_user_profile(cur_user: IMemoryUser, cur_user_profile: UserProfile, conn=None) -> UserProfile:
    """
    更新用户画像（原用户画像 is_active = False，新的用户画像 is_active = True）
    :param cur_user: 当前用户
    :param cur_user_profile: 当前画像
    :param conn: 数据库连接对象
    :return:
    """
    external_session = conn is not None
    session = conn
    if session is None:
        session_factory = get_session_factory()
        session = session_factory()

    key_b64 = _resolve_user_encryption_key(session, cur_user)

    # 用户习惯淘汰逻辑
    high_confidence_habits = [h for h in cur_user_profile.preferences.habits if h.confidence >= 0.5]
    cur_user_profile.preferences.habits = high_confidence_habits
    # 用户标签淘汰逻辑
    high_weight_tags = [t for t in cur_user_profile.tags if t.weight >= 0.5]
    cur_user_profile.tags = high_weight_tags

    demographic = cur_user_profile.demographic.model_dump(mode="json")
    preferences = cur_user_profile.preferences.model_dump(mode="json")
    attributes = cur_user_profile.attributes.model_dump(mode="json")
    tags = [tag.model_dump(mode="json") for tag in cur_user_profile.tags]

    _id = str(uuid.uuid4())
    now = datetime.datetime.now()

    demographic = _json_encrypt_values(
        demographic,
        key_b64=key_b64,
        aad_base={"id": _id, "user_id": str(cur_user.id), "field": "demographic"},
    )
    preferences = _json_encrypt_values(
        preferences,
        key_b64=key_b64,
        aad_base={"id": _id, "user_id": str(cur_user.id), "field": "preferences"},
    )
    attributes = _json_encrypt_values(
        attributes,
        key_b64=key_b64,
        aad_base={"id": _id, "user_id": str(cur_user.id), "field": "attributes"},
    )
    tags = _json_encrypt_values(
        tags,
        key_b64=key_b64,
        aad_base={"id": _id, "user_id": str(cur_user.id), "field": "tags"},
    )

    try:
        # 1. 先将原有 is_active = True 的 user_profiles 置为 False
        session.execute(
            update(UserProfiles)
            .where(UserProfiles.user_id == cur_user.id, UserProfiles.is_active.is_(True))
            .values(is_active=False, updated_at=now)
        )

        # 2. 插入新画像（is_active = True）
        session.add(
            UserProfiles(  # type: ignore[arg-type]
                id=_id,
                user_id=cur_user.id,
                demographic=demographic,
                preferences=preferences,
                attributes=attributes,
                tags=tags,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
        )
        if not external_session:
            session.commit()
    finally:
        if not external_session:
            session.close()

    cur_user_profile.set_id(_id)
    cur_user_profile.set_user_id(cur_user.id)
    cur_user_profile.set_is_active(True)

    return cur_user_profile


async def mark_memoires_to_profile_joined(m_ids: list[str], conn=None) -> int:
    """
    将记忆标记为已参与画像构建
    :param m_ids:
    :param conn:
    :return:
    """
    if not m_ids:
        return 0
    stmt = update(Memories).where(Memories.id.in_(m_ids)).values(profile_joined=True)
    external_session = conn is not None
    session = conn
    if session is None:
        session_factory = get_session_factory()
        session = session_factory()
    try:
        result = session.execute(stmt)
        if not external_session:
            session.commit()
        return int(getattr(result, "rowcount", 0) or 0)
    finally:
        if not external_session:
            session.close()
