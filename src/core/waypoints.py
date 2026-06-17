import datetime
from typing import List

import numpy as np
from agile.utils import LogHelper, singleton, timing
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core import user_ops
from src.core.db import get_session_factory
from src.core.mem_ops import mem_ops
from src.exceptions.exceptions import UserNotFoundError
from src.memory.entity.db_schema import Waypoints as WaypointEntity
from src.memory.memory_models import IMemoryUserIdentity, IMemoryUser
from src.tools.vectors import buf_to_vec, cos_sim

logger = LogHelper.get_logger(title="[WAYPOINTS]")


class Expansion(BaseModel):
    id: str = Field(..., description="记忆 ID")
    weight: float = Field(..., description="扩展权重")
    path: List[str] = Field(default_factory=list, description="扩展路径")


@singleton
class Waypoints:

    def __init__(self):
        self.session_factory = get_session_factory()

    async def link(self, user_identity: IMemoryUserIdentity, rid: str, cid: str, idx: int):
        """
        创建路标（waypoint）关系
        连接记忆 ID rid 和 cid，表示从 rid 可以通过路标到达 cid
        :param rid: 记忆 ID，作为路标的起点
        :param cid: 记忆 ID，作为路标的终点
        :param idx: 路标的索引或顺序
        :param user_identity: 用户身份，标识该路标所属的用户
        :return: None
        """
        user_identity.check_legality()
        user = await user_ops.get_user(user_identity)
        if not user:
            raise UserNotFoundError(user_identity)

        now = datetime.datetime.now()
        waypoint = WaypointEntity(
            src_id=rid,
            dst_id=cid,
            user_id=user.id,
            weight=1.0,
            created_at=now,
            updated_at=now,
        )
        with self.session_factory() as session:
            session.add(waypoint)
            session.commit()

    @timing
    async def expand_via_waypoints(self, ids: List[str], max_expansion: int = 10) -> List[Expansion]:
        """
        通过路标（waypoints）关系扩展记忆 ID 列表
        :param ids: 初始记忆 ID 列表
        :param max_expansion: 最大扩展数量
        :return: 扩展后的记忆列表
        """
        expansion = []
        vis = set(ids)
        q_arr: List[Expansion] = [Expansion(id=i, weight=1.0, path=[i]) for i in ids]
        cnt = 0

        # 广度优先遍历路标关系，进行记忆扩展
        with self.session_factory() as session:
            while q_arr and cnt < max_expansion:
                cur: Expansion = q_arr.pop(0)
                # 获取当前记忆的所有邻居（通过 waypoints 关系）
                neighs = session.execute(
                    select(WaypointEntity.dst_id, WaypointEntity.weight)
                    .where(WaypointEntity.src_id == cur.id)
                    .order_by(desc(WaypointEntity.weight))
                ).all()
                for dst, weight in neighs:
                    # 已经存在则忽略
                    if dst in vis:
                        continue
                    # 获取权重
                    wt = min(1.0, max(0.0, float(weight or 0.0)))
                    # 计算扩展权重
                    exp_wt = cur.weight * wt * 0.8
                    # 权重过低则忽略
                    if exp_wt < 0.1:
                        continue
                    # 添加扩展记忆项
                    item = Expansion(id=dst, weight=exp_wt, path=cur.path + [dst])
                    expansion.append(item)
                    vis.add(dst)
                    q_arr.append(item)
                    cnt += 1

        logger.info(f"Waypoint expansion completed, expanded {cnt} memories")

        # 返回扩展结果
        return expansion

    async def create_single_waypoint(self, new_id: str, new_mean: List[float], dt: datetime.datetime, user: IMemoryUser):
        """
        用于为新记忆（new_id）在所有记忆中寻找最相似的“均值向量”，并在数据库中建立 waypoint（路标）关联
        该函数会遍历当前用户的所有记忆，计算每个记忆的均值向量与新记忆均值向量的余弦相似度，
        找到相似度最高的记忆，并在数据库中插入一条 waypoint 记录，表示新记忆指向该最相似记忆。
        如果没有找到任何相似记忆，则创建一条自指向的 waypoint 记录。
        该函数最终会提交数据库事务以保存更改。

        @param new_id: 新记忆的唯一标识符
        @param new_mean: 新记忆的均值向量（浮点数列表）
        @param ts: 当前时间戳（毫秒）
        @param user: 用户
        """
        # 获取当前用户的所有记忆
        max_result = 1000
        memories = mem_ops.all_mem_by_user(user, limit=max_result, offset=0) if user.id else mem_ops.all_mem(limit=max_result, offset=0)
        best = None
        best_sim = -1.0

        # 将新记忆的均值向量转换为 numpy 数组
        nm = np.array(new_mean, dtype=np.float32)

        # 遍历所有记忆，计算与新记忆均值向量的余弦相似度
        for mem in memories:
            # 跳过自身或没有均值向量的记忆
            if mem["id"] == new_id or not mem["mean_vec"]:
                continue
            # 将现有记忆的均值向量转换为 numpy 数组
            ex_mean = np.array(buf_to_vec(mem["mean_vec"]), dtype=np.float32)
            # 计算与当前记忆的余弦相似度
            sim = cos_sim(nm, ex_mean)
            # 如果相似度超过当前最佳相似度，更新最佳记忆 ID 和相似度
            if sim > best_sim:
                best_sim = sim
                best = mem["id"]

        # 如果找到了最佳相似记忆，创建指向该记忆的 waypoint，否则创建自指向的 waypoint
        dst_id = best if best else new_id
        weight = float(best_sim) if best else 1.0
        stmt = pg_insert(WaypointEntity).values(
            src_id=new_id,
            dst_id=dst_id,
            user_id=user.id,
            weight=weight,
            created_at=dt,
            updated_at=dt,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[WaypointEntity.src_id, WaypointEntity.dst_id],
            set_={
                "user_id": stmt.excluded.user_id,
                "weight": stmt.excluded.weight,
                "created_at": stmt.excluded.created_at,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        with self.session_factory() as session:
            session.execute(stmt)
            session.commit()
