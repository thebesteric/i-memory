import queue

from agile.utils import LogHelper

from src.core import user_ops
from src.core.config import env
from src.core.dml_ops import dml_ops
from src.memory.models.memory_models import IMemoryUser, IMemoryUserIdentity

logger = LogHelper.get_logger()

# 存放用户的队列
user_identity_queue = queue.Queue()


async def build_graph():
    candidate_users: list[IMemoryUser] = await user_ops.find_user()
    # 过滤符合条件的用户
    for candidate_user in candidate_users:
        user_identity = IMemoryUserIdentity.from_dict(candidate_user)
        # 判断是否达到最小处理阈值
        un_fact_join_count = dml_ops.count_mem_by_user(user_identity, ["fact_joined is not true"])
        if un_fact_join_count < 20:
            continue

        # 获取未参与事实处理的记忆（按创建时间正序）
        un_fact_join_memories = dml_ops.find_un_fact_join_mem_by_user(user_identity, limit=env.GRAPH_MEM_COUNT_AT_MOST or 100)
        print(un_fact_join_memories)




