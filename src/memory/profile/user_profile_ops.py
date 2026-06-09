import datetime
import json
import uuid

from agile.utils import LogHelper

from src.core.components import USER_PROFILE_CACHE
from src.core.db import get_db
from src.memory.memory_models import IMemoryUser
from src.memory.profile.user_profile_models import UserProfile, Habit, Tag

logger = LogHelper.get_logger()
db = get_db()


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

    row = db.fetchone(
        "SELECT * FROM user_profiles WHERE is_active = TRUE AND user_id = %s",
        (user.id,)
    )
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
    # 1. 先将原有 is_active = True 的 user_profiles 置为 False
    db.execute(
        "UPDATE user_profiles SET is_active = FALSE WHERE user_id = %s AND is_active = TRUE",
        (cur_user.id,),
        conn=conn
    )

    # 2. 插入新画像（is_active = True）
    def json_serial(obj):
        if isinstance(obj, datetime.datetime):
            # 转成标准时间字符串
            return obj.isoformat()
        raise TypeError(f"无法序列化: {type(obj)}")

    # 用户习惯淘汰逻辑
    high_confidence_habits = [h for h in cur_user_profile.preferences.habits if h.confidence >= 0.5]
    cur_user_profile.preferences.habits = high_confidence_habits
    # 用户标签淘汰逻辑
    high_weight_tags = [t for t in cur_user_profile.tags if t.weight >= 0.5]
    cur_user_profile.tags = high_weight_tags

    demographic = json.dumps(cur_user_profile.demographic.model_dump(mode="json"), default=json_serial, ensure_ascii=False)
    preferences = json.dumps(cur_user_profile.preferences.model_dump(mode="json"), default=json_serial, ensure_ascii=False)
    attributes = json.dumps(cur_user_profile.attributes.model_dump(mode="json"), default=json_serial, ensure_ascii=False)
    tags = json.dumps([tag.model_dump() for tag in cur_user_profile.tags], default=json_serial, ensure_ascii=False)

    _id = str(uuid.uuid4())
    now = datetime.datetime.now()

    db.execute(
        """
        INSERT INTO user_profiles (id, user_id, demographic, preferences, attributes, tags, is_active, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (_id, cur_user.id, demographic, preferences, attributes, tags, True, now, now),
        conn=conn
    )
    if conn is None:
        db.commit()

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
    format_strings = ','.join(['%s'] * len(m_ids))
    query = f"UPDATE memories SET profile_joined = 1 WHERE id IN ({format_strings})"
    affected_rows = db.execute(
        query,
        tuple(m_ids),
        conn=conn
    )
    if conn is None:
        db.commit()
    return affected_rows