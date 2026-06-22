import datetime
import uuid

from agile.utils import LogHelper
from sqlalchemy import desc, select, update

from services.memory.components import USER_PROFILE_CACHE
from infra.db.engine import get_session_factory
from infra.db.orm_models import Memories, UserProfiles
from domain.memory.models import IMemoryUser
from domain.profile.models import UserProfile

logger = LogHelper.get_logger()


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
        entity = session.execute(
            select(UserProfiles)
            .where(UserProfiles.is_active.is_(True), UserProfiles.user_id == user.id)
            .order_by(desc(UserProfiles.updated_at))
            .limit(1)
        ).scalars().first()
    row = _profile_entity_to_row(entity) if entity else None
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
