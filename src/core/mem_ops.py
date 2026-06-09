import json
import datetime
from contextlib import contextmanager
from typing import Optional, Any, Dict, List, Literal

from agile.utils import singleton, timing

from src.core.db import DB, get_db, transaction
from src.memory.memory_models import IMemoryUser

db = get_db()


@singleton
class MemOps:

    def __init__(self):
        self.db: DB = db

    def ins_mem(self, **k) -> int:
        sql = """
              INSERT INTO memories(id, user_id, segment, content, primary_sector, sectors, tags, meta, created_at,
                                   updated_at,
                                   last_seen_at, salience, decay_lambda, version, mean_dim, mean_vec, compressed_vec,
                                   feedback_score,
                                   qa_role, qa_pair_id, batch_id)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
              ON CONFLICT (id) DO UPDATE SET user_id=EXCLUDED.user_id,
                                             segment=EXCLUDED.segment,
                                             content=EXCLUDED.content,
                                             primary_sector=EXCLUDED.primary_sector,
                                             sectors=EXCLUDED.sectors,
                                             tags=EXCLUDED.tags,
                                             meta=EXCLUDED.meta,
                                             created_at=EXCLUDED.created_at,
                                             updated_at=EXCLUDED.updated_at,
                                             last_seen_at=EXCLUDED.last_seen_at,
                                             salience=EXCLUDED.salience,
                                             decay_lambda=EXCLUDED.decay_lambda,
                                             version=EXCLUDED.version,
                                             mean_dim=EXCLUDED.mean_dim,
                                             mean_vec=EXCLUDED.mean_vec,
                                             compressed_vec=EXCLUDED.compressed_vec,
                                             feedback_score=EXCLUDED.feedback_score,
                                             qa_role=EXCLUDED.qa_role,
                                             qa_pair_id=EXCLUDED.qa_pair_id,
                                             batch_id=EXCLUDED.batch_id
              """
        vals = (
            k.get("id"), k.get("user_id"), k.get("segment", 0), k.get("content"),
            k.get("primary_sector"), k.get("sectors"), k.get("tags"), k.get("meta"), k.get("created_at"),
            k.get("updated_at"),
            k.get("last_seen_at"), k.get("salience", 1.0), k.get("decay_lambda", 0.02), k.get("version", 1),
            k.get("mean_dim"), k.get("mean_vec"), k.get("compressed_vec"), k.get("feedback_score", 0),
            k.get("qa_role"), k.get("qa_pair_id"), k.get("batch_id")
        )
        affected_rows = self.db.execute(sql, vals)
        self.db.commit()
        return affected_rows

    def get_mem(self, mid: str) -> Dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM memories WHERE id = %s", (mid,))

    def all_mem(self, limit=10, offset=0) -> List[Dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM memories ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset))

    def ins_log(self, _id: str, user_id: str, mem_id: str, model: str, status: str, ts: int,
                err: Optional[str] = None) -> int:
        affected_rows = self.db.execute(
            "INSERT INTO embed_logs(id, user_id, memory_id, model, status, ts, err) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (_id, user_id, mem_id, model, status, ts, err)
        )
        self.db.commit()
        return affected_rows

    def upd_log(self, _id: str, status: str, err: Optional[str] = None) -> int:
        affected_rows = self.db.execute(
            "UPDATE embed_logs SET status = %s, err = %s WHERE id = %s",
            (status, err, _id)
        )
        self.db.commit()
        return affected_rows

    @timing
    def find_mem_by_ids(self, mids: list[str]) -> List[Dict[str, Any]]:
        if not mids:
            return []
        format_strings = ','.join(['%s'] * len(mids))
        query = f"SELECT * FROM memories WHERE id IN ({format_strings})"
        return self.db.fetchall(query, tuple(mids))

    @timing
    def find_mem_by_user(self, user: IMemoryUser, order_by: List[str], limit=10, offset=0) -> List[Dict[str, Any]]:
        sql_parts = [
            """
            SELECT *
            FROM memories t
                     LEFT JOIN vectors v on t.id = v.id
            WHERE t.user_id = %s
              AND v.v IS NOT NULL
            """,
        ]

        # 查询参数列表，初始包含 user_id
        params = [user.id]

        # 拼接排序
        if order_by:
            order_by_clause = ", ".join(order_by)
            sql_parts.append(f"ORDER BY {order_by_clause}")

        # 分页
        sql_parts.append(f"LIMIT %s OFFSET %s")
        params.extend([str(limit), str(offset)])

        final_sql = " ".join(sql_parts)
        return self.db.fetchall(final_sql, tuple(params))

    def find_mem_by_conditions(self, *, conditions: list[str], order_by: List[str] = None, params: list[Any] = None,
                               limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        if not conditions:
            return []

        sql_parts = [
            "SELECT * FROM memories WHERE 1=1"
        ]
        sql_parts.append("AND " + " AND ".join(conditions))

        if order_by:
            order_by_clause = ", ".join(order_by)
            sql_parts.append(f"ORDER BY {order_by_clause}")

        if limit is not None:
            sql_parts.append("LIMIT %s OFFSET %s")
            params = params + [limit, offset]

        final_sql = " ".join(sql_parts)
        return self.db.fetchall(final_sql, tuple(params or []))

    def all_mem_by_user(self,
                        user: IMemoryUser,
                        limit=10,
                        offset=0,
                        sort_order: Literal["asc", "desc"] = "desc") -> List[Dict[str, Any]]:

        sql = f"SELECT * FROM memories WHERE user_id = %s ORDER BY created_at {sort_order} LIMIT %s OFFSET %s"
        return self.db.fetchall(sql, (user.id, limit, offset))

    def count_mem_by_user(self, user: IMemoryUser, conditions: list[str] = None) -> int:
        sql_parts = [
            "SELECT COUNT(*) as cnt FROM memories WHERE user_id = %s"
        ]
        params = [user.id]

        if conditions:
            sql_parts.append("AND " + " AND ".join(conditions))

        final_sql = " ".join(sql_parts)
        row = self.db.fetchone(final_sql, tuple(params))

        return row["cnt"] if row else 0

    def get_waypoints_by_src(self, src_id: str) -> List[Dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM waypoints WHERE src_id = %s", (src_id,))

    def del_mem(self, mem_id: str) -> int:
        # 先查该记忆的 user_id，用于失效用户画像
        mem = self.db.fetchone("SELECT user_id FROM memories WHERE id = %s", (mem_id,))
        user_id = mem["user_id"] if mem else None

        self.db.execute("DELETE FROM vectors WHERE id = %s", (mem_id,))
        self.db.execute("DELETE FROM waypoints WHERE src_id = %s OR dst_id = %s", (mem_id, mem_id))
        self.db.execute("DELETE FROM embed_logs WHERE memory_id = %s", (mem_id,))
        affected_rows = self.db.execute("DELETE FROM memories WHERE id = %s", (mem_id,))

        # 标记用户画像需重新计算
        if user_id:
            self.db.execute(
                "UPDATE user_profiles SET is_active = FALSE, updated_at = %s WHERE user_id = %s",
                (datetime.datetime.now(), user_id)
            )

        self.db.commit()
        return affected_rows

    def del_mems(self, mem_ids: List[str]) -> int:
        if not mem_ids:
            return 0

        with contextmanager(transaction)() as conn:
            id_placeholders = ",".join(["%s"] * len(mem_ids))
            mem_id_tuple = tuple(mem_ids)

            # 0. 获取要删除的记忆所属的 user_ids，用于后续按用户过滤 sessions/topics
            user_rows = self.db.fetchall(
                f"SELECT DISTINCT user_id FROM memories WHERE id IN ({id_placeholders})",
                mem_id_tuple, conn=conn
            )
            user_ids = [r["user_id"] for r in user_rows if r.get("user_id")]
            now = datetime.datetime.now()

            # 1. 清除子衍生表 (vectors, waypoints, embed_logs)
            self.db.execute(f"DELETE FROM vectors WHERE id IN ({id_placeholders})", mem_id_tuple, conn=conn)
            self.db.execute(
                f"DELETE FROM waypoints WHERE src_id IN ({id_placeholders}) OR dst_id IN ({id_placeholders})",
                tuple(mem_ids + mem_ids), conn=conn
            )
            self.db.execute(f"DELETE FROM embed_logs WHERE memory_id IN ({id_placeholders})", mem_id_tuple, conn=conn)

            # 2. 更新/清理会话 (sessions) —— 按 user_id 过滤，防止跨用户修改
            if user_ids:
                user_placeholders = ",".join(["%s"] * len(user_ids))
                sessions = self.db.fetchall(
                    f"SELECT id, dialogue_ids FROM sessions WHERE user_id IN ({user_placeholders})",
                    tuple(user_ids), conn=conn
                )
            else:
                sessions = []
            for s in sessions:
                orig_ids = s.get("dialogue_ids") or []
                if isinstance(orig_ids, str):
                    orig_ids = json.loads(orig_ids)
                new_ids = [mid for mid in orig_ids if mid not in mem_ids]
                if len(new_ids) != len(orig_ids):
                    if not new_ids:
                        self.db.execute("DELETE FROM sessions WHERE id = %s", (s["id"],), conn=conn)
                    else:
                        self.db.execute(
                            "UPDATE sessions SET dialogue_ids = %s, updated_at = %s WHERE id = %s",
                            (json.dumps(new_ids), now, s["id"]), conn=conn
                        )

            # 3. 更新/清理图话题与图事实 (graph_topics / graph_facts) —— 按 user_id 过滤
            if user_ids:
                topics = self.db.fetchall(
                    f"SELECT id, dialogue_ids FROM graph_topics WHERE user_id IN ({user_placeholders})",
                    tuple(user_ids), conn=conn
                )
            else:
                topics = []
            topics_to_delete = []
            for t in topics:
                orig_ids = t.get("dialogue_ids") or []
                if isinstance(orig_ids, str):
                    orig_ids = json.loads(orig_ids)
                new_ids = [mid for mid in orig_ids if mid not in mem_ids]
                if len(new_ids) != len(orig_ids):
                    if not new_ids:
                        topics_to_delete.append(t["id"])
                    else:
                        self.db.execute(
                            "UPDATE graph_topics SET dialogue_ids = %s, updated_at = %s WHERE id = %s",
                            (json.dumps(new_ids), now, t["id"]), conn=conn
                        )

            if topics_to_delete:
                topic_placeholders = ",".join(["%s"] * len(topics_to_delete))
                facts = self.db.fetchall(
                    f"SELECT id FROM graph_facts WHERE topic_id IN ({topic_placeholders})",
                    tuple(topics_to_delete), conn=conn
                )
                fact_ids = [f["id"] for f in facts]
                if fact_ids:
                    fact_placeholders = ",".join(["%s"] * len(fact_ids))
                    fact_id_tuple = tuple(fact_ids)

                    # 先 UPDATE: 从 graph_entity_relations.fact_ids 数组中移除被删的 fact_ids
                    update_params = tuple(list(fact_id_tuple) + [now] + list(fact_id_tuple))
                    self.db.execute(f"""
                        UPDATE graph_entity_relations 
                        SET fact_ids = (
                            SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                            FROM jsonb_array_elements_text(fact_ids) AS elem
                            WHERE elem NOT IN ({fact_placeholders})
                        ),
                        updated_at = %s
                        WHERE EXISTS (
                            SELECT 1 
                            FROM jsonb_array_elements_text(fact_ids) AS fid
                            WHERE fid IN ({fact_placeholders})
                        )
                    """, update_params, conn=conn)

                    # 然后 DELETE: fact_ids 变为空的 relation
                    self.db.execute(
                        "DELETE FROM graph_entity_relations WHERE fact_ids = '[]'::jsonb OR fact_ids IS NULL",
                        conn=conn
                    )

                    self.db.execute(
                        f"DELETE FROM graph_fact_entities WHERE fact_id IN ({fact_placeholders})",
                        fact_id_tuple, conn=conn
                    )
                    self.db.execute(
                        f"DELETE FROM graph_facts WHERE topic_id IN ({topic_placeholders})",
                        tuple(topics_to_delete), conn=conn
                    )

                self.db.execute(
                    f"DELETE FROM graph_topics WHERE id IN ({topic_placeholders})",
                    tuple(topics_to_delete), conn=conn
                )

            # 4. 标记用户画像需重新计算
            if user_ids:
                self.db.execute(
                    f"UPDATE user_profiles SET is_active = FALSE, updated_at = %s WHERE user_id IN ({user_placeholders})",
                    tuple([now] + user_ids), conn=conn
                )

            # 5. 从 memories 主表中删除记录
            affected_rows = self.db.execute(
                f"DELETE FROM memories WHERE id IN ({id_placeholders})",
                mem_id_tuple, conn=conn
            )
            return affected_rows

    def del_mem_by_user(self, user: IMemoryUser) -> int:
        user_id = user.id
        with contextmanager(transaction)() as conn:
            self.db.execute("DELETE FROM graph_entity_relations WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM graph_fact_entities WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM graph_entities WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM graph_canonical_entities WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM graph_facts WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM graph_topics WHERE user_id = %s", (user_id,), conn=conn)

            self.db.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM user_profiles WHERE user_id = %s", (user_id,), conn=conn)

            self.db.execute("DELETE FROM vectors WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM waypoints WHERE user_id = %s", (user_id,), conn=conn)
            self.db.execute("DELETE FROM embed_logs WHERE user_id = %s", (user_id,), conn=conn)

            affected_rows = self.db.execute("DELETE FROM memories WHERE user_id = %s", (user_id,), conn=conn)
            return affected_rows

    def del_mem_by_batch(self, user_id: str, batch_id: str) -> int:
        with contextmanager(transaction)() as conn:
            now = datetime.datetime.now()

            # 0. 获取本批次要删除的 memory ids（用于后续清理 topics.dialogue_ids）
            mem_rows = self.db.fetchall(
                "SELECT id FROM memories WHERE batch_id = %s AND user_id = %s",
                (batch_id, user_id), conn=conn
            )
            mem_ids = [r["id"] for r in mem_rows]

            # 1. 更新/清理图实体关系: 从 fact_ids 数组中移除被删的 fact_ids
            self.db.execute("""
                UPDATE graph_entity_relations 
                SET fact_ids = (
                    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                    FROM jsonb_array_elements_text(fact_ids) AS elem
                    WHERE elem NOT IN (SELECT id FROM graph_facts WHERE batch_id = %s AND user_id = %s)
                ),
                updated_at = %s
                WHERE user_id = %s
                  AND EXISTS (
                      SELECT 1 
                      FROM jsonb_array_elements_text(fact_ids) AS fid
                      WHERE fid IN (SELECT id FROM graph_facts WHERE batch_id = %s AND user_id = %s)
                  )
            """, (batch_id, user_id, now, user_id, batch_id, user_id), conn=conn)

            # 删除 fact_ids 变为空的 relation
            self.db.execute(
                "DELETE FROM graph_entity_relations WHERE fact_ids = '[]'::jsonb OR fact_ids IS NULL",
                conn=conn
            )

            # 2. 删除事实实体映射
            self.db.execute("""
                DELETE FROM graph_fact_entities 
                WHERE fact_id IN (SELECT id FROM graph_facts WHERE batch_id = %s AND user_id = %s)
            """, (batch_id, user_id), conn=conn)

            # 3. 更新/清理图话题 (graph_topics): 移除被删的记忆 ID，空的删掉
            topics = self.db.fetchall(
                "SELECT id, dialogue_ids FROM graph_topics WHERE user_id = %s",
                (user_id,), conn=conn
            )
            topics_to_delete = []
            for t in topics:
                orig_ids = t.get("dialogue_ids") or []
                if isinstance(orig_ids, str):
                    orig_ids = json.loads(orig_ids)
                new_ids = [mid for mid in orig_ids if mid not in mem_ids]
                if len(new_ids) != len(orig_ids):
                    if not new_ids:
                        topics_to_delete.append(t["id"])
                    else:
                        self.db.execute(
                            "UPDATE graph_topics SET dialogue_ids = %s, updated_at = %s WHERE id = %s",
                            (json.dumps(new_ids), now, t["id"]), conn=conn
                        )
            if topics_to_delete:
                topic_placeholders = ",".join(["%s"] * len(topics_to_delete))
                # 级联删除空话题下的 facts (graph_facts 已在上一步按 batch_id 删除，这里清除未被 batch 覆盖的)
                remaining_facts = self.db.fetchall(
                    f"SELECT id FROM graph_facts WHERE topic_id IN ({topic_placeholders})",
                    tuple(topics_to_delete), conn=conn
                )
                if remaining_facts:
                    rfact_ids = [f["id"] for f in remaining_facts]
                    rfact_placeholders = ",".join(["%s"] * len(rfact_ids))
                    self.db.execute(
                        f"DELETE FROM graph_fact_entities WHERE fact_id IN ({rfact_placeholders})",
                        tuple(rfact_ids), conn=conn
                    )
                    self.db.execute(
                        f"DELETE FROM graph_facts WHERE topic_id IN ({topic_placeholders})",
                        tuple(topics_to_delete), conn=conn
                    )
                self.db.execute(
                    f"DELETE FROM graph_topics WHERE id IN ({topic_placeholders})",
                    tuple(topics_to_delete), conn=conn
                )

            # 4. 删除图事实与会话
            self.db.execute(
                "DELETE FROM graph_facts WHERE batch_id = %s AND user_id = %s",
                (batch_id, user_id), conn=conn
            )
            self.db.execute(
                "DELETE FROM sessions WHERE batch_id = %s AND user_id = %s",
                (batch_id, user_id), conn=conn
            )

            # 5. 删除向量、路标、日志 (从 memories 表衍生)
            self.db.execute("""
                DELETE FROM vectors 
                WHERE id IN (SELECT id FROM memories WHERE batch_id = %s AND user_id = %s)
            """, (batch_id, user_id), conn=conn)

            self.db.execute("""
                DELETE FROM waypoints 
                WHERE src_id IN (SELECT id FROM memories WHERE batch_id = %s AND user_id = %s) 
                   OR dst_id IN (SELECT id FROM memories WHERE batch_id = %s AND user_id = %s)
            """, (batch_id, user_id, batch_id, user_id), conn=conn)

            self.db.execute("""
                DELETE FROM embed_logs 
                WHERE memory_id IN (SELECT id FROM memories WHERE batch_id = %s AND user_id = %s)
            """, (batch_id, user_id), conn=conn)

            # 6. 标记用户画像需重新计算
            self.db.execute(
                "UPDATE user_profiles SET is_active = FALSE, updated_at = %s WHERE user_id = %s",
                (now, user_id), conn=conn
            )

            # 7. 删除记忆主表记录
            affected_rows = self.db.execute(
                "DELETE FROM memories WHERE batch_id = %s AND user_id = %s",
                (batch_id, user_id), conn=conn
            )
            return affected_rows


mem_ops = MemOps()
