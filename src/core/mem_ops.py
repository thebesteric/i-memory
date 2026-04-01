from typing import Optional, Any, Dict, List

from agile.utils import singleton, timing

from src.core.db import DB, get_db
from src.memory.memory_models import IMemoryUser

db = get_db()


@singleton
class MemOps:

    def __init__(self):
        self.db: DB = db

    def ins_mem(self, **k) -> int:
        sql = """
              INSERT INTO memories(id, user_id, segment, content, primary_sector, sectors, tags, meta, created_at, updated_at,
                                   last_seen_at, salience, decay_lambda, version, mean_dim, mean_vec, compressed_vec, feedback_score,
                                   qa_role, qa_pair_id)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                                             qa_pair_id=EXCLUDED.qa_pair_id
              """
        vals = (
            k.get("id"), k.get("user_id"), k.get("segment", 0), k.get("content"),
            k.get("primary_sector"), k.get("sectors"), k.get("tags"), k.get("meta"), k.get("created_at"), k.get("updated_at"),
            k.get("last_seen_at"), k.get("salience", 1.0), k.get("decay_lambda", 0.02), k.get("version", 1),
            k.get("mean_dim"), k.get("mean_vec"), k.get("compressed_vec"), k.get("feedback_score", 0),
            k.get("qa_role"), k.get("qa_pair_id")
        )
        affected_rows = self.db.execute(sql, vals)
        self.db.commit()
        return affected_rows

    def get_mem(self, mid: str) -> Dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM memories WHERE id = %s", (mid,))

    def all_mem(self, limit=10, offset=0) -> List[Dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM memories ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset))

    def ins_log(self, id: str, model: str, status: str, ts: int, err: Optional[str] = None) -> int:
        affected_rows = self.db.execute("INSERT INTO embed_logs(id, model, status, ts, err) VALUES (%s, %s, %s, %s, %s)", (id, model, status, ts, err))
        self.db.commit()
        return affected_rows

    def upd_log(self, id: str, status: str, err: Optional[str] = None) -> int:
        affected_rows = self.db.execute("UPDATE embed_logs SET status = %s, err = %s WHERE id = %s", (status, err, id))
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

    def find_mem_by_conditions(self, *, conditions: list[str], order_by: List[str] = None, params: list[Any] = None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
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

    def all_mem_by_user(self, user: IMemoryUser, limit=10, offset=0) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM memories WHERE user_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s"
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

    def del_mem(self, mid: str) -> int:
        self.db.execute("DELETE FROM vectors WHERE id = %s", (mid,))
        self.db.execute("DELETE FROM waypoints WHERE src_id = %s OR dst_id = %s", (mid, mid))
        self.db.execute("DELETE FROM embed_logs WHERE id = %s", (mid,))
        affected_rows = self.db.execute("DELETE FROM memories WHERE id = %s", (mid,))
        self.db.commit()
        return affected_rows

    def del_mem_by_user(self, user: IMemoryUser) -> int:
        sql = "SELECT id FROM memories WHERE user_id = %s"
        memory_ids = self.db.fetchall(sql, (user.id,))
        if not memory_ids:
            return 0
        ids = tuple(row["id"] for row in memory_ids)
        self.db.execute("DELETE FROM vectors WHERE id IN %s", (ids,))
        self.db.execute("DELETE FROM waypoints WHERE src_id IN %s OR dst_id IN %s", (ids, ids))
        self.db.execute("DELETE FROM embed_logs WHERE id IN %s", (ids,))
        affected_rows = self.db.execute("DELETE FROM memories WHERE id IN %s", (ids,))
        self.db.commit()
        return affected_rows


mem_ops = MemOps()
