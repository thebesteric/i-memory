import asyncio
from typing import Callable

from agile.utils import LogHelper

from infra.db.repos import user_repo
from shared.config.settings import env
from infra.db.engine import get_session_factory
from infra.db.repos.memory_repo import mem_ops
from services.graph import graph_ops
from services.graph.fact_extractor import FactExtractor
from services.graph.semantic_spliter import SemanticSpliter, Dialogue, SemanticsOutput
from domain.memory.models import IMemoryUser

logger = LogHelper.get_logger(title="[GRAPH_BUILD]")
session_factory = get_session_factory()

# 存放待图化的用户队列（异步队列）
user_queue = asyncio.Queue()
# 记录已入队用户ID，防止重复入队
enqueued_user_ids = set()
# 记忆最小容忍度数量
GRAPH_MEM_COUNT_AT_LEAST_TOLERANCE = 10


async def get_un_fact_join_mem_count(callback: Callable[[IMemoryUser, int], None]):
    """
    查询所有用户，统计每个用户的 fact_join = 0 的数量，并通过 callback 处理
    """
    candidate_users: list[IMemoryUser] = await user_repo.find_user(status=1)
    for candidate_user in candidate_users:
        un_fact_join_count = mem_ops.count_mem_by_user(candidate_user, ["fact_joined = 0"])
        callback(candidate_user, un_fact_join_count)


async def graph_build():
    """
    构建图（由 jobs 的 graph_build 定时任务调用）
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
    每日强制图化：将长期未达阈值的用户也入队（由 jobs 的 force_graph_build 调用）
    """

    def enqueue_if_cold_user(user: IMemoryUser, un_fact_join_count: int):
        """
        未达阈值但有记忆的冷用户入队（记忆大于 10 条，小于 GRAPH_MEM_COUNT_AT_LEAST 条）
        :param user:
        :param un_fact_join_count:
        :return:
        """
        if GRAPH_MEM_COUNT_AT_LEAST_TOLERANCE < un_fact_join_count < env.GRAPH_MEM_COUNT_AT_LEAST and user.id not in enqueued_user_ids:
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

            # 记忆小于 10 条的用户不处理，避免过度处理冷用户
            if not un_fact_join_memories_ids or len(un_fact_join_memories_ids) < GRAPH_MEM_COUNT_AT_LEAST_TOLERANCE:
                logger.warning(
                    f"User {user.id} has {len(un_fact_join_memories_ids)} un-fact-joined memories, which is less than {GRAPH_MEM_COUNT_AT_LEAST_TOLERANCE}. Skipping processing."
                )
                continue

            # 将记忆进行语义切分，得到主题对象列表
            dialogues = [Dialogue.mem_to_dialogue(mem) for mem in un_fact_join_memories]
            semantic_output: SemanticsOutput = await semantic_spliter.invoke(dialogues=dialogues)

            # 并发生成所有 fact
            topics = semantic_output.topics
            fact_tasks = [fact_extractor.invoke(topic=topic) for topic in topics]
            facts = await asyncio.gather(*fact_tasks)

            # 事务包裹所有写操作，数据库串行
            with session_factory() as db_session:
                with db_session.begin():
                    for topic, fact in zip(topics, facts):
                        # 将 Topic 入库
                        topic = await graph_ops.add_topic(user, topic, conn=db_session)
                        # 将 Fact 入库
                        fact = await graph_ops.add_fact(user, fact, topic, conn=db_session)
                        # 将 Entity 入库，并添加与 Fact 的关系映射，标准化实体对象
                        await graph_ops.link_fact_entities(user, fact, conn=db_session)
                        # 基于 canonical_id 推断实体共现关系边
                        await graph_ops.infer_canonical_relations_for_fact(user, fact, conn=db_session)

                    # 提取已经参与生成事实的记忆 ID 列表
                    mem_ids = set()
                    for topic in topics:
                        mem_ids.update(topic.dialogue_ids or [])

                    # 将记忆更新为已参与事实处理
                    await graph_ops.mark_memoires_to_fact_joined(list(un_fact_join_memories_ids), conn=db_session)

                    logger.info(f"Finished processing user: {user.id}, "
                                f"processed memories: {len(un_fact_join_memories_ids)}, "
                                f"fact joined memories: {len(mem_ids)}, generated facts: {len(facts)}")

        except Exception as e:
            logger.error(f"Error processing user: {user.id}, error: {e}")
            raise e
        finally:
            # 处理完成后移除用户 ID，允许后续重新入队
            enqueued_user_ids.discard(user.id)
            user_queue.task_done()
