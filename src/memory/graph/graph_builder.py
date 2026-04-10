import asyncio
from typing import Callable
from contextlib import contextmanager

from agile.utils import LogHelper

from src.core import user_ops
from src.core.config import env
from src.core.db import transaction
from src.core.mem_ops import mem_ops
from src.memory.graph import graph_ops
from src.memory.graph.fact_extractor import FactExtractor
from src.memory.graph.semantic_spliter import SemanticSpliter, Dialogue, SemanticsOutput
from src.memory.memory_models import IMemoryUser

logger = LogHelper.get_logger(title="[GRAPH]")

# 存放待图化的用户队列（异步队列）
user_queue = asyncio.Queue()
# 记录已入队用户ID，防止重复入队
enqueued_user_ids = set()


async def get_un_fact_join_mem_count(callback: Callable[[IMemoryUser, int], None]):
    """
    查询所有用户，统计每个用户的 fact_join = 0 的数量，并通过 callback 处理
    """
    candidate_users: list[IMemoryUser] = await user_ops.find_user(status=1)
    for candidate_user in candidate_users:
        un_fact_join_count = mem_ops.count_mem_by_user(candidate_user, ["fact_joined = 0"])
        callback(candidate_user, un_fact_join_count)


async def graph_build():
    """
    构建图（由定时任务调用）
    """

    def enqueue_if_reach_threshold(user: IMemoryUser, un_fact_join_count: int):
        """
        达到阈值则入队（（记忆大于 GRAPH_MEM_COUNT_AT_LEAST 条）
        :param user:
        :param un_fact_join_count:
        :return:
        """
        if un_fact_join_count >= env.GRAPH_MEM_COUNT_AT_LEAST and user.id not in enqueued_user_ids:
            user_queue.put_nowait(user)
            enqueued_user_ids.add(user.id)

    await get_un_fact_join_mem_count(enqueue_if_reach_threshold)


async def graph_build_daily_force():
    """
    每日强制图化：将长期未达阈值的用户也入队
    """

    def enqueue_if_cold_user(user: IMemoryUser, un_fact_join_count: int):
        """
        未达阈值但有记忆的冷用户入队（记忆大于 10 条，小于 GRAPH_MEM_COUNT_AT_LEAST 条）
        :param user:
        :param un_fact_join_count:
        :return:
        """
        if 10 < un_fact_join_count < env.GRAPH_MEM_COUNT_AT_LEAST and user.id not in enqueued_user_ids:
            user_queue.put_nowait(user)
            enqueued_user_ids.add(user.id)

    await get_un_fact_join_mem_count(enqueue_if_cold_user)


async def process_user_queue():
    semantic_spliter = SemanticSpliter()
    fact_extractor = FactExtractor()
    while True:
        # 异步阻塞式获取队列元素
        user: IMemoryUser = await user_queue.get()
        try:
            logger.info(f"Processing user: {user.id}")
            # 获取未参与事实处理的记忆（按创建时间正序）
            un_fact_join_memories = mem_ops.find_mem_by_conditions(
                conditions=["fact_joined = 0", "user_id = %s"],
                order_by=["created_at ASC"],
                params=[user.id],
                limit=env.GRAPH_MEM_COUNT_AT_MOST or 100
            )
            un_fact_join_memories_ids: list[str] = [mem["id"] for mem in un_fact_join_memories]

            # 将记忆进行语义切分，得到主题对象列表
            dialogues = [Dialogue.mem_to_dialogue(mem) for mem in un_fact_join_memories]
            semantic_output: SemanticsOutput = await semantic_spliter.invoke(dialogues=dialogues)

            # 并发生成所有 fact
            topics = semantic_output.topics
            fact_tasks = [fact_extractor.invoke(topic=topic) for topic in topics]
            facts = await asyncio.gather(*fact_tasks)

            # 事务包裹所有写操作，数据库串行
            with contextmanager(transaction)() as conn:
                for topic, fact in zip(topics, facts):
                    # 将 Topic 入库
                    topic = await graph_ops.add_topic(user, topic, conn=conn)
                    # 将 Fact 入库
                    fact = await graph_ops.add_fact(user, fact, topic, conn=conn)
                    # 将 Entity 入库，并添加与 Fact 的关系映射，标准化实体对象
                    await graph_ops.link_fact_entities(user, fact, conn=conn)
                    # 基于 canonical_id 推断实体共现关系边
                    await graph_ops.infer_canonical_relations_for_fact(user, fact, conn=conn)

                # 提取已经参与生成事实的记忆 ID 列表
                mem_ids = set()
                for topic in topics:
                    mem_ids.update(topic.dialogue_ids or [])

                # 将记忆更新为已参与事实处理
                await graph_ops.mark_memoires_to_fact_joined(list(un_fact_join_memories_ids), conn=conn)

                logger.info(
                    f"Finished processing user: {user.id}, processed memories: {len(un_fact_join_memories_ids)}, fact joined memories: {len(mem_ids)}, generated facts: {len(facts)}"
                )

        except Exception as e:
            logger.error(f"Error processing user: {user.id}, error: {e}")
            raise e
        finally:
            # 处理完成后移除用户 ID，允许后续重新入队
            enqueued_user_ids.discard(user.id)
            user_queue.task_done()
