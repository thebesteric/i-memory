import datetime
import json
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
# 列表元素的默认身份键：当路径未命中配置时使用。
_DEFAULT_LIST_IDENTITY_KEYS = ("id", "key", "name", "code", "type", "label")
# 按路径配置列表元素身份键，便于不同字段采用不同去重/upsert 策略。
_LIST_IDENTITY_KEYS_BY_PATH: dict[str, tuple[str, ...]] = {
    "tags": ("name",),
    "preferences.habits": ("name",),
    # demographic.extra 下通常是业务实体集合，优先用具名字段识别，避免仅凭 type 误合并。
    "demographic.extra.*": ("id", "key", "name", "code", "label"),
}
# 通用瘦身上限，防止画像字段无限膨胀。
_MAX_LIST_ITEMS = 50
_MAX_DICT_ITEMS = 60
_MAX_NESTED_DEPTH = 6


def _resolve_user_encryption_key(orm_session, user: IMemoryUser) -> str | None:
    # 获取用户密钥
    if user.encryption_key:
        return user.encryption_key
    key_cache = load_user_encryption_keys(orm_session, [str(user.id)])
    return key_cache.get(str(user.id))


def _json_encrypt_values(value: Any, *, key_b64: str | None, aad_base: dict[str, Any], path: str = "") -> Any:
    # 递归加密画像 JSON 中的字符串字段：
    # - dict / list 继续向下遍历；
    # - str 按当前 path 参与 AAD，保证同值在不同字段位置不可互换；
    # - 非字符串标量（数字、布尔等）保持原样。
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
    # 与 _json_encrypt_values 对应的递归解密逻辑：
    # - 仅处理带有约定前缀的密文字符串；
    # - dict / list 保持结构不变；
    # - 使用相同的 path 参与 AAD 校验，避免字段串位解密。
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
    # 将数据库查出的 user_profiles 行按字段整体解密。
    # 这里以 profile_id/user_id/field 作为 AAD 基础信息，再由递归函数补充 path，
    # 从而确保 demographic / preferences / attributes / tags 各自独立、可校验。
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
    # 将 SQLAlchemy ORM 实体拍平成普通 dict，过滤掉内部管理字段，
    # 便于后续做 JSON 解密、Pydantic 反序列化和通用 merge。
    return {k: v for k, v in vars(profile).items() if not k.startswith("_")}


def _is_empty_value(value: Any) -> bool:
    # 统一判空规则：用于压缩阶段剔除噪声字段。
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _resolve_identity_keys(path: str) -> tuple[str, ...]:
    # 支持精确路径与通配路径（例如：demographic.extra.*）。
    if path in _LIST_IDENTITY_KEYS_BY_PATH:
        return _LIST_IDENTITY_KEYS_BY_PATH[path]
    for pattern, keys in _LIST_IDENTITY_KEYS_BY_PATH.items():
        if pattern.endswith(".*") and path.startswith(pattern[:-2]):
            return keys
    return _DEFAULT_LIST_IDENTITY_KEYS


def _list_identity(item: Any, *, path: str) -> str | None:
    # 从对象中提取稳定身份，便于列表内做 upsert 合并。
    if not isinstance(item, dict):
        return None
    for key in _resolve_identity_keys(path):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip().lower()}"
        if isinstance(value, (int, float)):
            return f"{key}:{value}"
    return None


def _normalize_scalar_for_dedupe(value: Any) -> str:
    # 将标量归一化为可比较字符串，支撑“按值去重”。
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _merge_lists(base: list[Any], incoming: list[Any], *, path: str) -> list[Any]:
    # 列表中如果是对象，按通用身份键 upsert；否则按值去重拼接。
    merged: list[Any] = []
    index_by_identity: dict[str, int] = {}
    seen_scalars: set[str] = set()

    def _append_or_merge(item: Any):
        identity = _list_identity(item, path=path)
        if identity:
            if identity in index_by_identity:
                idx = index_by_identity[identity]
                merged[idx] = _deep_merge(merged[idx], item, path=f"{path}[]" if path else "[]")
            else:
                index_by_identity[identity] = len(merged)
                merged.append(item)
            return

        scalar_key = _normalize_scalar_for_dedupe(item)
        if scalar_key in seen_scalars:
            return
        seen_scalars.add(scalar_key)
        merged.append(item)

    for item in base:
        _append_or_merge(item)
    for item in incoming:
        _append_or_merge(item)

    return merged


def _deep_merge(base: Any, incoming: Any, *, path: str = "") -> Any:
    # 通用深度合并策略：dict 递归、list upsert、标量新值覆盖旧值。
    if incoming is None:
        return base
    if base is None:
        return incoming

    if isinstance(base, dict) and isinstance(incoming, dict):
        merged = dict(base)
        for key, value in incoming.items():
            child_path = f"{path}.{key}" if path else key
            merged[key] = _deep_merge(merged.get(key), value, path=child_path)
        return merged

    if isinstance(base, list) and isinstance(incoming, list):
        return _merge_lists(base, incoming, path=path)

    return incoming


def _compact_nested(value: Any, depth: int = 0, path: str = "") -> Any:
    # 通用递归压缩：去空值、去重、限长、限深。
    if depth >= _MAX_NESTED_DEPTH:
        return None

    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, raw in value.items():
            child_path = f"{path}.{key}" if path else key
            item = _compact_nested(raw, depth + 1, path=child_path)
            if _is_empty_value(item):
                continue
            compacted[key] = item
            if len(compacted) >= _MAX_DICT_ITEMS:
                break
        return compacted

    if isinstance(value, list):
        compacted_items: list[Any] = []
        for raw in value:
            item = _compact_nested(raw, depth + 1, path=path)
            if _is_empty_value(item):
                continue
            compacted_items.append(item)
        deduped = _merge_lists([], compacted_items, path=path)
        return deduped[:_MAX_LIST_ITEMS]

    return value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compact_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # 先做通用压缩，再做画像业务约束（habits/tags 的质量与数量控制）。
    compacted = _compact_nested(payload)
    if not isinstance(compacted, dict):
        return payload

    preferences_raw = compacted.get("preferences")
    preferences = preferences_raw if isinstance(preferences_raw, dict) else {}
    tags_raw = compacted.get("tags")
    tags = tags_raw if isinstance(tags_raw, list) else []

    habits_raw = preferences.get("habits")
    habits = habits_raw if isinstance(habits_raw, list) else []
    high_confidence_habits = [h for h in habits if isinstance(h, dict) and _safe_float(h.get("confidence")) >= 0.5]
    high_confidence_habits.sort(key=lambda h: _safe_float(h.get("confidence")), reverse=True)
    preferences["habits"] = high_confidence_habits[:10]

    high_weight_tags = [t for t in tags if isinstance(t, dict) and _safe_float(t.get("weight")) >= 0.5]
    high_weight_tags.sort(key=lambda t: _safe_float(t.get("weight")), reverse=True)
    compacted["tags"] = high_weight_tags[:15]
    compacted["preferences"] = preferences

    return compacted


def _build_merged_user_profile(existing_profile: UserProfile | None, new_profile: UserProfile) -> UserProfile:
    # 双策略并存：模型产物 + 程序兜底合并，最后统一压缩。
    base_payload = existing_profile.model_dump(mode="json") if existing_profile else {}
    incoming_payload = new_profile.model_dump(mode="json")
    merged_payload = _deep_merge(base_payload, incoming_payload)
    merged_payload = _compact_profile_payload(merged_payload)
    return UserProfile(**merged_payload)


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

    # 读取当前 active 画像，作为“手动合并”的基线。
    existing_entity = session.execute(
        select(UserProfiles)
        .where(UserProfiles.is_active.is_(True), UserProfiles.user_id == cur_user.id)
        .order_by(desc(UserProfiles.updated_at))
        .limit(1)
    ).scalars().first()
    existing_row = _profile_entity_to_row(existing_entity) if existing_entity else None
    existing_profile = None
    if existing_row:
        existing_profile = UserProfile.from_dict(_decrypt_user_profile_row(existing_row, key_b64=key_b64))

    # 将“旧画像 + 本次模型输出”合成最终画像，保证增量更新与结构稳定。
    merged_user_profile = _build_merged_user_profile(existing_profile, cur_user_profile)

    demographic = merged_user_profile.demographic.model_dump(mode="json")
    preferences = merged_user_profile.preferences.model_dump(mode="json")
    attributes = merged_user_profile.attributes.model_dump(mode="json")
    tags = [tag.model_dump(mode="json") for tag in merged_user_profile.tags]

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

    merged_user_profile.set_id(_id)
    merged_user_profile.set_user_id(cur_user.id)
    merged_user_profile.set_is_active(True)
    # upsert 成功后刷新缓存，后续读请求直接拿到最新画像。
    USER_PROFILE_CACHE.set(cur_user.id, merged_user_profile)

    return merged_user_profile


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
