import datetime
import uuid

from src.core.db import get_db
from src.memory.models.memory_models import IMemoryUserIdentity, IMemoryUser
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()

db = get_db()


async def get_user(user_identity: IMemoryUserIdentity) -> IMemoryUser | None:
    """
    获取用户信息
    :param user_identity: 用户身份
    :return:
    """
    user_id = user_identity.user_id
    tenant_id = user_identity.tenant_id
    project_id = user_identity.project_id

    sql_parts = [
        "SELECT * FROM users WHERE user_id = %s"
    ]
    params = [user_id]

    if tenant_id:
        sql_parts.append("AND tenant_id = %s")
        params.append(tenant_id)

    if project_id:
        sql_parts.append("AND project_id = %s")
        params.append(project_id)

    final_sql = " ".join(sql_parts)
    user = db.fetchone(final_sql, tuple(params))
    return IMemoryUser(
        id=user["id"],
        tenant_id=user["tenant_id"],
        project_id=user["project_id"],
        user_id=user["user_id"],
        summary=user["summary"],
        reflection_count=user["reflection_count"],
        created_at=user["created_at"],
        updated_at=user["updated_at"]
    ) if user else None


async def add_user(user_identity: IMemoryUserIdentity, summary: str = None, reflection_count: int = 0) -> None:
    """
    添加用户
    :param user_identity: 用户身份
    :param summary: 用户概要信息
    :param reflection_count: 反思次数
    :return:
    """
    user_id = user_identity.user_id
    tenant_id = user_identity.tenant_id
    project_id = user_identity.project_id

    summary = summary if summary else "User profile initializing..."
    reflection_count = reflection_count if reflection_count is not None else 0

    # 当前时间
    now = datetime.datetime.now()

    db.execute(
        """
        INSERT INTO users(id, tenant_id, project_id, user_id, summary, reflection_count, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tenant_id, project_id, user_id) DO NOTHING
        """,
        (str(uuid.uuid4()), tenant_id, project_id, user_id, summary, reflection_count, now, now)
    )
    db.commit()


async def update_user_summary(id: str, summary: str) -> None:
    """
    更新用户概要信息
    :param id: 主键
    :param summary: 用户概要信息
    :return:
    """
    # 当前时间
    now = datetime.datetime.now()
    db.execute("UPDATE users SET summary = %s, updated_at = %s WHERE id = %s", (summary, now, id))
    db.commit()
