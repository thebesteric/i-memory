import queue

from src.core import user_ops
from src.memory.models.memory_models import IMemoryUser

# 存放用户的队列
user_queue = queue.Queue()


async def build_graph():
    users: list[IMemoryUser] = await user_ops.find_user()
    # 过滤符合条件的用户
