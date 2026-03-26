import asyncio
from typing import Callable
from contextlib import contextmanager

from agile.utils import LogHelper

from src.core import user_ops
from src.core.config import env
from src.core.db import transaction
from src.core.dml_ops import dml_ops
from src.memory.graph import graph_ops
from src.memory.graph.fact_extract import FactExtract
from src.memory.graph.semantic_split import SemanticSplit, Dialogue, SemanticsOutput
from src.memory.models.memory_models import IMemoryUser

logger = LogHelper.get_logger()

# 存放待图化的用户队列（异步队列）
user_queue = asyncio.Queue()
# 记录已入队用户ID，防止重复入队
enqueued_user_ids = set()


async def get_un_fact_join_mem_count(callback: Callable[[IMemoryUser, int], None]):
    """
    查询所有用户，统计每个用户的 fact_join = 0 的数量，并通过 callback 处理
    """
    candidate_users: list[IMemoryUser] = await user_ops.find_user()
    for candidate_user in candidate_users:
        un_fact_join_count = dml_ops.count_mem_by_user(candidate_user, ["fact_joined = 0"])
        callback(candidate_user, un_fact_join_count)


async def graph_build():
    """
    构建图（由定时任务调用）
    """

    def enqueue_if_reach_threshold(user: IMemoryUser, un_fact_join_count: int):
        """达到阈值则入队，防止重复入队"""
        if un_fact_join_count >= env.GRAPH_MEM_COUNT_AT_LEAST and user.id not in enqueued_user_ids:
            user_queue.put_nowait(user)
            enqueued_user_ids.add(user.id)

    await get_un_fact_join_mem_count(enqueue_if_reach_threshold)


async def graph_build_daily_force():
    """
    每日强制图化：将长期未达阈值的用户也入队
    """

    def enqueue_if_cold_user(user: IMemoryUser, un_fact_join_count: int):
        """未达阈值但有记忆的冷用户入队，防止重复入队"""
        if 0 < un_fact_join_count < env.GRAPH_MEM_COUNT_AT_LEAST and user.id not in enqueued_user_ids:
            user_queue.put_nowait(user)
            enqueued_user_ids.add(user.id)

    await get_un_fact_join_mem_count(enqueue_if_cold_user)


async def process_user_queue():
    semantic_split = SemanticSplit()
    fact_extract = FactExtract()
    while True:
        # 异步阻塞式获取队列元素
        user: IMemoryUser = await user_queue.get()
        try:
            logger.info(f"[GRAPH] Processing user: {user.id}")
            # 获取未参与事实处理的记忆（按创建时间正序）
            un_fact_join_memories = dml_ops.find_un_fact_join_mem_by_user(user, limit=env.GRAPH_MEM_COUNT_AT_MOST or 100)
            un_fact_join_memories_ids = [mem["id"] for mem in un_fact_join_memories]

            try:
                # 将记忆进行语义切分，得到主题对象列表
                dialogues = [Dialogue.mem_to_dialogue(mem) for mem in un_fact_join_memories]
                semantic_output: SemanticsOutput = await semantic_split.invoke(dialogues=dialogues)

                # 并发生成所有 fact
                topics = semantic_output.topics
                fact_tasks = [fact_extract.invoke(topic=topic) for topic in topics]
                facts = await asyncio.gather(*fact_tasks)

                # 事务包裹所有写操作，数据库串行
                with contextmanager(transaction)() as conn:
                    for topic, fact in zip(topics, facts):
                        # 将 Topic 入库
                        topic = await graph_ops.add_topic(user, topic, conn=conn)
                        # 将 Fact 入库
                        fact = await graph_ops.add_fact(user, fact, topic, conn=conn)
                        # 将 Entity 入库
                        await graph_ops.link_fact_entities(user, fact, conn=conn)

                    # 提取已经参与生成事实的记忆 ID 列表
                    mem_ids = set()
                    for topic in topics:
                        mem_ids.update(topic.dialogue_ids or [])

                    # 将记忆更新为已参与事实处理
                    await graph_ops.mark_memoires_to_fact_joined(list(mem_ids), conn=conn)

                    # 将所有召回的记忆 joined_count + 1，超过一定次数的记忆将被丢弃
                    await graph_ops.increment_memoires_join_count(un_fact_join_memories_ids, env.GRAPH_MEM_DISCARD_THRESHOLD, conn=conn)

                    logger.info(f"===============================")

            except Exception as e:
                logger.error(f"[GRAPH] Error processing user: {user.id}, error: {e}")

        except Exception as e:
            logger.error(f"Error processing user: {user.id}, error: {e}")
        finally:
            # 处理完成后移除用户 ID，允许后续重新入队
            enqueued_user_ids.discard(user.id)
            user_queue.task_done()
