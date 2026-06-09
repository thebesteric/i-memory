import datetime
import uuid

from agile.utils import LogHelper

from src.core.components import USER_IDENTITY_CACHE
from src.core.db import get_db
from src.memory.memory_models import IMemoryUserIdentity, IMemoryUser

logger = LogHelper.get_logger()

db = get_db()


async def find_user(*, ids: list[str] | None = None, order_by: list[str] = None, status: int | None = None, limit=9999, offset=0) -> list[IMemoryUser]:
    """
    查询所有用户
    :param ids: 用户 ID 列表
    :param order_by: 排序调节
    :param status: 用户状态，0 = 禁用，1 = 启用，None = 全部
    :param limit: 数量限制
    :param offset: 偏移量
    :return:
    """
    sql_parts = [
        "SELECT * FROM users t"
    ]

    params = []
    where_conditions = []

    # 状态 status 筛选
    if status is not None:
        where_conditions.append("t.status = %s")
        params.append(status)

    # 用户 ID 筛选
    if ids and len(ids) > 0:
        id_placeholders = ", ".join(["%s"] * len(ids))
        where_conditions.append(f"t.id IN ({id_placeholders})")
        params.extend(ids)

    # 拼接 WHERE 条件
    if where_conditions:
        sql_parts.append("WHERE " + " AND ".join(where_conditions))

    # 拼接排序
    if order_by:
        order_by_clause = ", ".join(order_by)
        sql_parts.append(f"ORDER BY {order_by_clause}")

    # 分页
    sql_parts.append(f"LIMIT %s OFFSET %s")
    params.extend([limit, offset])

    final_sql = " ".join(sql_parts)
    users = db.fetchall(final_sql, tuple(params))
    return [IMemoryUser.from_dict(u) for u in users]


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

    sql_parts = [
        "SELECT * FROM users WHERE user_key = %s"
    ]
    params = [user_key]

    if tenant_key:
        sql_parts.append("AND tenant_key = %s")
        params.append(tenant_key)

    if project_key:
        sql_parts.append("AND project_key = %s")
        params.append(project_key)

    final_sql = " ".join(sql_parts)
    user = db.fetchone(final_sql, tuple(params))
    memory_user = IMemoryUser.from_dict(user) if user else None

    # 加入缓存
    USER_IDENTITY_CACHE.set(cache_key, memory_user)

    # 返回用户
    return memory_user


async def add_user(user_identity: IMemoryUserIdentity, summary: str = None, reflection_count: int = 0) -> IMemoryUser:
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

    # 当前时间
    now = datetime.datetime.now()
    _id = str(uuid.uuid4())

    db.execute(
        """
        INSERT INTO users(id, tenant_key, project_key, user_key, summary, reflection_count, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tenant_key, project_key, user_key) DO NOTHING
        """,
        (_id, tenant_key, project_key, user_key, summary, reflection_count, now, now)
    )
    db.commit()
    return IMemoryUser(
        id=_id,
        tenant_key=tenant_key,
        project_key=project_key,
        user_key=user_key,
        summary=summary,
        reflection_count=reflection_count,
        created_at=now,
        updated_at=now,
    )


async def update_user_summary(_id: str, summary: str) -> None:
    """
    更新用户概要信息
    :param _id: 主键
    :param summary: 用户概要信息
    :return:
    """
    # 当前时间
    now = datetime.datetime.now()
    db.execute("UPDATE users SET summary = %s, updated_at = %s WHERE id = %s", (summary, now, _id))
    db.commit()
