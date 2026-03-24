from typing import Optional, Any, Dict, List, Literal, Tuple

from agile.utils import singleton, timing

from src.core.db import DB, get_db
from src.memory.models.memory_models import IMemoryUserIdentity

db = get_db()


@singleton
class DMLOps:

    def __init__(self):
        self.db: DB = db

    def ins_mem(self, **k) -> int:
        sql = """
              INSERT INTO memories(id, user_id, tenant_id, project_id, segment, content, primary_sector, sectors, tags, meta, created_at, updated_at,
                                   last_seen_at, salience, decay_lambda, version, mean_dim, mean_vec, compressed_vec, feedback_score,
                                   qa_role, qa_pair_id)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
              ON CONFLICT (id) DO UPDATE SET user_id=EXCLUDED.user_id,
                                             tenant_id=EXCLUDED.tenant_id,
                                             project_id=EXCLUDED.project_id,
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
            k.get("id"), k.get("user_id"), k.get("tenant_id"), k.get("project_id"), k.get("segment", 0), k.get("content"),
            k.get("primary_sector"), k.get("sectors"), k.get("tags"), k.get("meta"), k.get("created_at"), k.get("updated_at"),
            k.get("last_seen_at"), k.get("salience", 1.0), k.get("decay_lambda", 0.02), k.get("version", 1),
            k.get("mean_dim"), k.get("mean_vec"), k.get("compressed_vec"), k.get("feedback_score", 0),
            k.get("qa_role"), k.get("qa_pair_id")
        )
        affected_rows = self.db.execute(sql, vals)
        self.db.commit()
        return affected_rows

    @timing
    def find_mem(self, mids: list[str]) -> List[Dict[str, Any]]:
        if not mids:
            return []
        format_strings = ','.join(['%s'] * len(mids))
        query = f"SELECT * FROM memories WHERE id IN ({format_strings})"
        return self.db.fetchall(query, tuple(mids))

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
    def find_mem_by_user(self, user_identity: IMemoryUserIdentity, order_by: List[str], limit=10, offset=0) -> List[Dict[str, Any]]:
        user_id = user_identity.user_id
        tenant_id = user_identity.tenant_id
        project_id = user_identity.project_id

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
        params = [user_id]

        # 判断租户是否存在
        if tenant_id:
            sql_parts.append("AND t.tenant_id = %s")
            params.append(tenant_id)

        # 判断项目是否存在
        if project_id:
            sql_parts.append("AND t.project_id = %s")
            params.append(project_id)

        # 拼接排序
        if order_by:
            order_by_clause = ", ".join(order_by)
            sql_parts.append(f"ORDER BY {order_by_clause}")

        # 分页
        sql_parts.append(f"LIMIT %s OFFSET %s")
        params.extend([str(limit), str(offset)])

        final_sql = " ".join(sql_parts)
        return self.db.fetchall(final_sql, tuple(params))

    def all_mem_by_user(self, user_identity: IMemoryUserIdentity, limit=10, offset=0) -> List[Dict[str, Any]]:
        user_id = user_identity.user_id
        tenant_id = user_identity.tenant_id
        project_id = user_identity.project_id

        sql_parts = [
            "SELECT * FROM memories WHERE user_id = %s"
        ]
        params = [user_id]

        if tenant_id:
            sql_parts.append("AND tenant_id = %s")
            params.append(tenant_id)

        if project_id:
            sql_parts.append("AND project_id = %s")
            params.append(project_id)

        sql_parts.append("ORDER BY created_at DESC LIMIT %s OFFSET %s")
        params.extend([str(limit), str(offset)])

        final_sql = " ".join(sql_parts)
        return self.db.fetchall(final_sql, tuple(params))

    def find_un_fact_join_mem_by_user(self, user_identity: IMemoryUserIdentity, limit=10, offset=0) -> List[Dict[str, Any]]:
        user_id = user_identity.user_id
        tenant_id = user_identity.tenant_id
        project_id = user_identity.project_id

        sql_parts = [
            "SELECT * FROM memories WHERE fact_joined IS NOT TRUE AND user_id = %s"
        ]
        params = [user_id]

        if tenant_id:
            sql_parts.append("AND tenant_id = %s")
            params.append(tenant_id)

        if project_id:
            sql_parts.append("AND project_id = %s")
            params.append(project_id)

        sql_parts.append("ORDER BY created_at ASC LIMIT %s OFFSET %s")
        params.extend([str(limit), str(offset)])

        final_sql = " ".join(sql_parts)
        return self.db.fetchall(final_sql, tuple(params))

    def count_mem_by_user(self, user_identity: IMemoryUserIdentity, conditions: list[str] = None) -> int:
        user_id = user_identity.user_id
        tenant_id = user_identity.tenant_id
        project_id = user_identity.project_id

        sql_parts = [
            "SELECT COUNT(*) as cnt FROM memories WHERE user_id = %s"
        ]
        params = [user_id]

        if tenant_id:
            sql_parts.append("AND tenant_id = %s")
            params.append(tenant_id)

        if project_id:
            sql_parts.append("AND project_id = %s")
            params.append(project_id)

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

    def del_mem_by_user(self, user_identity: IMemoryUserIdentity) -> int:
        user_id = user_identity.user_id
        tenant_id = user_identity.tenant_id
        project_id = user_identity.project_id

        sql_parts = [
            "SELECT id FROM memories WHERE user_id = %s"
        ]
        params: List[str] = [user_id]

        if tenant_id:
            sql_parts.append("AND tenant_id = %s")
            params.append(tenant_id)

        if project_id:
            sql_parts.append("AND project_id = %s")
            params.append(project_id)

        final_sql = " ".join(sql_parts)
        memory_ids = self.db.fetchall(final_sql, tuple(params))
        if not memory_ids:
            return 0
        ids = tuple(row["id"] for row in memory_ids)
        self.db.execute("DELETE FROM vectors WHERE id IN %s", (ids,))
        self.db.execute("DELETE FROM waypoints WHERE src_id IN %s OR dst_id IN %s", (ids, ids))
        self.db.execute("DELETE FROM embed_logs WHERE id IN %s", (ids,))
        affected_rows = self.db.execute("DELETE FROM memories WHERE id IN %s", (ids,))
        self.db.commit()
        return affected_rows


dml_ops = DMLOps()
