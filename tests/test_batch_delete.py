import asyncio
import unittest
import uuid
import datetime
import json
from src.core import user_ops
from src.core.db import get_db
from src.imemory import IMemory
from src.memory.memory_models import IMemoryUserIdentity, IMemoryUser
from src.memory.session import session_ops
from src.memory.session.session_models import Session, Sessions
from src.memory.graph import graph_ops
from src.memory.graph.graph_models import Fact
from src.memory.graph.semantic_spliter import Topic


class TestBatchDelete(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # 强制重置单例，确保每个测试方法都有干净的环境且绑定到正确的 Event Loop
        from src.imemory import IMemory
        from src.core.components import get_vector_store
        from src.core.vector.postgres_vector_store import PostgresVectorStore
        
        # 重置 IMemory 单例 (SingletonWrapper 的特殊处理)
        if hasattr(IMemory, "_instance"):
            IMemory._instance = None
        
        # 重置 PostgresVectorStore 单例
        vs = get_vector_store()
        if isinstance(vs, PostgresVectorStore):
            await vs.close()
            if hasattr(PostgresVectorStore, "_instance"):
                PostgresVectorStore._instance = None

        self.user_identity: IMemoryUserIdentity = IMemoryUserIdentity(
            user_key="test_batch_user",
            tenant_key="test_tenant",
            project_key="test_project"
        )
        # 重新创建实例，此时会触发 __init__ 并在当前 loop 中初始化
        self.mem = IMemory(user_identity=self.user_identity)
        # 等待初始化完成
        await self.mem._prepare_resource()
        
        self.db = get_db()
        
        # Ensure user exists
        self.user: IMemoryUser = await user_ops.get_user(self.user_identity)
        if not self.user:
            await user_ops.add_user(self.user_identity)
            self.user = await user_ops.get_user(self.user_identity)

    async def asyncTearDown(self):
        # 清理连接池资源
        from src.core.components import get_vector_store
        from src.core.vector.postgres_vector_store import PostgresVectorStore
        vs = get_vector_store()
        if isinstance(vs, PostgresVectorStore):
            await vs.close()
        await asyncio.sleep(0.2)

    async def test_all_delete_scenarios(self):
        # 场景 A: 按 batch_id 级联删除
        batch_id = f"test_batch_{uuid.uuid4().hex[:8]}"
        
        # 1. Add memories under this batch_id
        r1 = await self.mem.add(
            "今天是星期三，明天我要去图书馆借几本书来看。",
            user_identity=self.user_identity,
            batch_id=batch_id
        )
        r2 = await self.mem.add(
            "下午天气很好，适合骑自行车在公园转转，晒晒太阳。",
            user_identity=self.user_identity,
            batch_id=batch_id
        )
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)

        m_ids = [r1["id"], r2["id"]]

        # 2. Insert a simulated Session for these memories
        session = Session(
            summary="讨论日常活动和天气",
            dialogue_ids=m_ids,
            key_facts=["周三活动安排", "去图书馆", "公园骑车"]
        )
        await session_ops.insert_sessions(self.user, Sessions(sessions=[session]))
        session_id = session.id
        
        # 3. Insert a simulated Fact and Topic
        topic = Topic(
            name="周三安排",
            summary="计划去图书馆并骑车",
            keywords=["周三", "图书馆"],
            dialogue_ids=m_ids
        )
        await graph_ops.add_topic(self.user, topic)
        
        fact = Fact(
            what="计划去图书馆",
            who="我",
            when="明天",
            where="图书馆",
            why="看书",
            confidence=0.9
        )
        await graph_ops.add_fact(self.user, fact, topic)

        # 4. Now run cascading delete by batch
        affected_rows = await self.mem.delete_by_batch(self.user_identity, batch_id)
        print(f"Cascading delete by batch completed, affected rows (memories): {affected_rows}")
        self.assertEqual(affected_rows, 2)

        # 5. Assert database has clean state for this batch_id
        self.assertEqual(len(self.db.fetchall("SELECT * FROM memories WHERE batch_id = %s", (batch_id,))), 0)
        self.assertEqual(len(self.db.fetchall("SELECT * FROM sessions WHERE id = %s", (session_id,))), 0)
        self.assertEqual(len(self.db.fetchall("SELECT * FROM graph_facts WHERE topic_id = %s", (topic.id,))), 0)

        # 场景 B: 精确按 memory_ids 级联删除
        # 1. Add two memories
        r3 = await self.mem.add(
            "这是第一条独立记忆内容。",
            user_identity=self.user_identity
        )
        r4 = await self.mem.add(
            "这是第二条独立记忆内容。",
            user_identity=self.user_identity
        )
        m3_id = r3["id"]
        m4_id = r4["id"]

        # 2. Insert session referencing both memories
        session_b = Session(
            summary="测试独立删除会话汇总",
            dialogue_ids=[m3_id, m4_id],
            key_facts=["独立事实1", "独立事实2"]
        )
        await session_ops.insert_sessions(self.user, Sessions(sessions=[session_b]))
        session_b_id = session_b.id

        # 3. Insert topic & fact referencing both memories
        topic_b = Topic(
            name="独立主题",
            summary="测试独立主题汇总",
            keywords=["测试"],
            dialogue_ids=[m3_id, m4_id]
        )
        await graph_ops.add_topic(self.user, topic_b)
        
        fact_b = Fact(
            what="测试独立事实",
            who="测试员",
            when="现在",
            where="办公室",
            why="测试删除",
            confidence=0.95
        )
        await graph_ops.add_fact(self.user, fact_b, topic_b)

        # 4. Perform cascade deletion of FIRST memory ID ONLY (m3_id)
        affected = await self.mem.delete_mems([m3_id])
        self.assertEqual(affected, 1)

        # Check m3 is deleted, m4 remains
        self.assertIsNone(self.db.fetchone("SELECT * FROM memories WHERE id = %s", (m3_id,)))
        self.assertIsNotNone(self.db.fetchone("SELECT * FROM memories WHERE id = %s", (m4_id,)))

        # Check session is updated, topic is updated
        session_after = self.db.fetchone("SELECT * FROM sessions WHERE id = %s", (session_b_id,))
        self.assertIsNotNone(session_after)
        
        # 5. Perform cascade deletion of SECOND memory ID (m4_id)
        affected_2 = await self.mem.delete_mems([m4_id])
        self.assertEqual(affected_2, 1)

        # Check everything is cleared
        self.assertIsNone(self.db.fetchone("SELECT * FROM sessions WHERE id = %s", (session_b_id,)))
        self.assertIsNone(self.db.fetchone("SELECT * FROM graph_topics WHERE id = %s", (topic_b.id,)))
        self.assertIsNone(self.db.fetchone("SELECT * FROM graph_facts WHERE id = %s", (fact_b.id,)))


if __name__ == '__main__':
    unittest.main()
