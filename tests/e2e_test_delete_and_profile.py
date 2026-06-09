"""
端到端测试：两种删除方式 + 4维度验证 (1个方法避免 event loop 冲突)
  方式A: delete_by_batch — 删除整轮会话
  方式B: delete_mems   — 删除单条消息
"""
import asyncio, unittest, uuid
from src.imemory import IMemory
from src.memory.memory_models import IMemoryUserIdentity
from src.core import user_ops
from src.core.db import get_db
from src.memory.session import session_ops
from src.memory.session.session_models import Session, Sessions
from src.memory.graph import graph_ops
from src.memory.graph.graph_models import Fact
from src.memory.graph.semantic_spliter import Topic

db = get_db()

DIALOGUE = [
    "我平时喜欢喝拿铁咖啡，特别是加了燕麦奶的那种",
    "好的，记住了：您喜欢燕麦奶拿铁。还有其他口味偏好吗？",
    "有时候我也会试试新口味，比如加了火龙果的拿铁也不错",
    "了解！火龙果拿铁很有创意，已加入您的偏好记录。",
    "我通常在早上的咖啡馆里看会儿书，感觉很惬意",
    "早上的咖啡馆确实是阅读好时光，咖啡香气配书本很享受。",
    "周末我喜欢去公园跑步，大概跑5公里左右",
    "跑步是好习惯！5公里是很健康的距离，继续保持。",
    "我的工作主要用Python和TypeScript开发后端服务",
    "Python+TypeScript，很现代的技术栈。",
    "最近在学习机器学习，对自然语言处理特别感兴趣",
    "NLP是非常有趣的方向。",
    "我每天下班后会弹半小时吉他放松一下",
    "弹吉他是很好的放松方式。",
]


class TestBothDeleteModes(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        from src.core.components import get_vector_store
        from src.core.vector.postgres_vector_store import PostgresVectorStore
        for cls in [IMemory]:
            if hasattr(cls, "_instance"): cls._instance = None
        vs = get_vector_store()
        if isinstance(vs, PostgresVectorStore):
            await vs.close()
            if hasattr(PostgresVectorStore, "_instance"): PostgresVectorStore._instance = None

    async def asyncTearDown(self):
        from src.core.components import get_vector_store
        from src.core.vector.postgres_vector_store import PostgresVectorStore
        vs = get_vector_store()
        if isinstance(vs, PostgresVectorStore):
            await vs.close()
        await asyncio.sleep(0.2)

    async def _make_user(self):
        key = f"e2e_{uuid.uuid4().hex[:8]}"
        identity = IMemoryUserIdentity(user_key=key, tenant_key="test_tenant", project_key="test_project")
        user = await user_ops.get_user(identity)
        if not user:
            await user_ops.add_user(identity)
            user = await user_ops.get_user(identity)
        return identity, user

    async def _add_dialogue(self, identity, batch_id=None):
        mem = IMemory(user_identity=identity)
        await mem._prepare_resource()
        ids = []
        for i, c in enumerate(DIALOGUE):
            role = "human" if i % 2 == 0 else "assistant"
            r = await mem.add(c, user_identity=identity, batch_id=batch_id, qa_role=role)
            ids.append(r["id"])
        return ids

    async def _search_ids(self, query, identity):
        mem = IMemory(user_identity=identity)
        await mem._prepare_resource()
        r = await mem.search(query, limit=10)
        return {it.id for it in (r.memories or [])}

    def _db_has(self, table, col, keyword, user_id):
        rows = db.fetchall(f"SELECT 1 FROM {table} WHERE user_id=%s AND {col}::text ILIKE %s LIMIT 1", (user_id, f"%{keyword}%"))
        return len(rows) > 0

    def _db_session_refs(self, memory_id):
        rows = db.fetchall("SELECT 1 FROM sessions WHERE dialogue_ids::text LIKE %s LIMIT 1", (f"%{memory_id}%",))
        return len(rows) > 0

    # ================================================================
    async def test_all_delete_scenarios(self):
        # ==================== 方式A: delete_by_batch ====================
        print("\n🅰️ 方式A: delete_by_batch (删除整轮会话)")
        identity_A, user_A = await self._make_user()
        batch = "batch_" + uuid.uuid4().hex[:8]

        # A1: 添加对话
        mem_ids_A = await self._add_dialogue(identity_A, batch_id=batch)
        self.assertGreaterEqual(len(mem_ids_A), 10)
        bad_id_A = next((mid for mid in mem_ids_A if identity_A.user_key in mid and mid), mem_ids_A[2])
        # 找火龙果记忆
        search_ids_A = await self._search_ids("火龙果", identity_A)
        self.assertTrue(len(search_ids_A) > 0, "A: 删除前应搜到火龙果")

        # A2: 插入图谱 & 会话
        topic_A = Topic(name="咖啡口味", summary="燕麦奶和火龙果", keywords=["火龙果"], dialogue_ids=mem_ids_A)
        await graph_ops.add_topic(user_A, topic_A)
        fact_A = Fact(what="用户喜欢火龙果拿铁", who="用户", when="最近", where="咖啡馆", why="创新", confidence=0.9)
        await graph_ops.add_fact(user_A, fact_A, topic_A)
        session_A = Session(summary="讨论火龙果咖啡偏好", dialogue_ids=mem_ids_A, key_facts=["火龙果拿铁"])
        await session_ops.insert_sessions(user_A, Sessions(sessions=[session_A]))

        # 验证插入成功
        self.assertTrue(self._db_has("graph_facts", "what", "火龙果", user_A.id), "A: graph_facts 应含火龙果")
        self.assertTrue(self._db_has("graph_topics", "summary", "火龙果", user_A.id), "A: graph_topics 应含火龙果")
        self.assertTrue(self._db_session_refs(mem_ids_A[0]), "A: sessions 应引用这些记忆")

        # A3: delete_by_batch
        mem = IMemory(user_identity=identity_A)
        await mem._prepare_resource()
        affected = await mem.delete_by_batch(identity_A, batch)
        self.assertGreaterEqual(affected, len(mem_ids_A), "A: delete_by_batch 行数不够")

        # A4: 搜索不含火龙果（用新 query 避缓存）
        search_after_A = await self._search_ids("咖啡火龙果口味", identity_A)
        self.assertEqual(len(search_after_A), 0, "A: 删除后不应搜到火龙果")

        # A5: 级联清理验证
        self.assertFalse(self._db_session_refs(mem_ids_A[0]), "A: sessions 应被清空")
        self.assertFalse(self._db_has("graph_facts", "what", "火龙果", user_A.id), "A: graph_facts 应不含火龙果")
        self.assertFalse(self._db_has("graph_topics", "name", "咖啡", user_A.id), "A: graph_topics 应被级联删")
        rows = db.fetchall("SELECT is_active FROM user_profiles WHERE user_id=%s ORDER BY created_at DESC LIMIT 1", (user_A.id,))
        if rows:
            self.assertFalse(rows[0]["is_active"], "A: 画像应为 is_active=FALSE")

        print("  ✅ 方式A 全部通过")

        # ==================== 方式B: delete_mems ====================
        print("\n🅱️ 方式B: delete_mems (删除单条消息)")
        identity_B, user_B = await self._make_user()

        # B1: 添加对话
        mem_ids_B = await self._add_dialogue(identity_B)
        self.assertGreaterEqual(len(mem_ids_B), 10)
        search_before_B = await self._search_ids("火龙果", identity_B)
        self.assertTrue(len(search_before_B) > 0, "B: 删除前应搜到火龙果")
        # 用对话列表中的火龙果记忆 ID（索引2），确保 session 引用一致
        bad_id_B = mem_ids_B[2]  # DIALOGUE[2] = "有时候我也会试试新口味，比如加了火龙果的拿铁也不错"

        # B2: 插入图谱 & 会话
        topic_B = Topic(name="咖啡口味", summary="燕麦奶和火龙果", keywords=["火龙果"], dialogue_ids=mem_ids_B)
        await graph_ops.add_topic(user_B, topic_B)
        fact_B = Fact(what="用户喜欢火龙果拿铁", who="用户", when="最近", where="咖啡馆", why="创新", confidence=0.9)
        await graph_ops.add_fact(user_B, fact_B, topic_B)
        session_B = Session(summary="讨论火龙果咖啡", dialogue_ids=mem_ids_B, key_facts=["火龙果"])
        await session_ops.insert_sessions(user_B, Sessions(sessions=[session_B]))

        self.assertTrue(self._db_has("graph_facts", "what", "火龙果", user_B.id), "B: graph_facts 应含火龙果")
        self.assertTrue(self._db_session_refs(bad_id_B), "B: sessions 应引用记忆")

        # B3: 删除单条
        mem_B = IMemory(user_identity=identity_B)
        await mem_B._prepare_resource()
        affected = await mem_B.delete_mems([bad_id_B])
        self.assertEqual(affected, 1, "B: delete_mems 应删除1条")

        # B4: 搜索不含火龙果（用新 query 避缓存）
        search_after_B = await self._search_ids("火龙果新口味咖啡", identity_B)
        self.assertEqual(len(search_after_B), 0, "B: 删除后不应搜到火龙果")

        # B5: 级联清理验证
        remaining = len(db.fetchall("SELECT 1 FROM memories WHERE user_id=%s", (user_B.id,)))
        self.assertEqual(remaining, len(mem_ids_B) - 1, "B: 剩余记忆数应少1条")
        self.assertFalse(self._db_session_refs(bad_id_B), "B: sessions 不应再引用已删记忆")

        # graph_topics.dialogue_ids 已更新
        ref_rows = db.fetchall("SELECT 1 FROM graph_topics WHERE user_id=%s AND dialogue_ids::text LIKE %s", (user_B.id, f"%{bad_id_B}%"))
        self.assertEqual(len(ref_rows), 0, "B: graph_topics 不应引用已删记忆")

        # 画像标记失效
        pf_rows = db.fetchall("SELECT is_active FROM user_profiles WHERE user_id=%s ORDER BY created_at DESC LIMIT 1", (user_B.id,))
        if pf_rows:
            self.assertFalse(pf_rows[0]["is_active"], "B: 画像应为 is_active=FALSE")

        print("  ✅ 方式B 全部通过")
        print("\n🎉 两种删除方式 + 4维验证 全部通过！")


if __name__ == "__main__":
    unittest.main()
