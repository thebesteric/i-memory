from typing import Optional

from src.core.db import DB, get_db

from src.utils.singleton import singleton

db = get_db()


@singleton
class DMLOps:

    def __init__(self):
        self.db: DB = db

    def ins_mem(self, **k):
        sql = """
              INSERT INTO memories(id, user_id, segment, content, primary_sector, sectors, tags, meta, created_at, updated_at, last_seen_at, salience,
                                   decay_lambda, version, mean_dim, mean_vec, compressed_vec, feedback_score)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                                             feedback_score=EXCLUDED.feedback_score
              """
        vals = (
            k.get("id"), k.get("user_id"), k.get("segment", 0), k.get("content"),
            k.get("primary_sector"), k.get("sectors"), k.get("tags"), k.get("meta"), k.get("created_at"), k.get("updated_at"),
            k.get("last_seen_at"), k.get("salience", 1.0), k.get("decay_lambda", 0.02), k.get("version", 1),
            k.get("mean_dim"), k.get("mean_vec"), k.get("compressed_vec"), k.get("feedback_score", 0)
        )
        self.db.execute(sql, vals)
        self.db.commit()

    def find_mem(self, mids: list[str]):
        format_strings = ','.join(['%s'] * len(mids))
        query = f"SELECT * FROM memories WHERE id IN ({format_strings})"
        return self.db.fetchall(query, tuple(mids))

    def get_mem(self, mid: str):
        return self.db.fetchone("SELECT * FROM memories WHERE id = %s", (mid,))

    def all_mem(self, limit=10, offset=0):
        return self.db.fetchall("SELECT * FROM memories ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset))

    def ins_log(self, id: str, model: str, status: str, ts: int, err: Optional[str] = None):
        self.db.execute("INSERT INTO embed_logs(id, model, status, ts, err) VALUES (%s, %s, %s, %s, %s)", (id, model, status, ts, err))
        self.db.commit()

    def upd_log(self, id: str, status: str, err: Optional[str] = None):
        self.db.execute("UPDATE embed_logs SET status = %s, err = %s WHERE id = %s", (status, err, id))
        self.db.commit()

    def all_mem_by_user(self, user_id: str, limit=10, offset=0):
        return self.db.fetchall("SELECT * FROM memories WHERE user_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s", (user_id, limit, offset))

    def count_mem_by_user(self, user_id: str) -> int:
        row = self.db.fetchone("SELECT COUNT(*) as cnt FROM memories WHERE user_id = %s", (user_id,))
        return row["cnt"] if row else 0

    def get_waypoints_by_src(self, src_id: str):
        return self.db.fetchall("SELECT * FROM waypoints WHERE src_id = %s", (src_id,))

    def del_mem(self, mid: str):
        self.db.execute("DELETE FROM vectors WHERE id = %s", (mid,))
        self.db.execute("DELETE FROM waypoints WHERE src_id = %s OR dst_id = %s", (mid, mid))
        self.db.execute("DELETE FROM embed_logs WHERE id = %s", (mid,))
        self.db.execute("DELETE FROM memories WHERE id = %s", (mid,))
        self.db.commit()

    def del_mem_by_user(self, uid: str):
        memory_ids = self.db.fetchall("SELECT id FROM memories WHERE user_id = %s", (uid,))
        if not memory_ids:
            return
        ids = tuple(row["id"] for row in memory_ids)
        self.db.execute("DELETE FROM vectors WHERE id IN %s", (ids,))
        self.db.execute("DELETE FROM waypoints WHERE src_id IN %s OR dst_id IN %s", (ids, ids))
        self.db.execute("DELETE FROM embed_logs WHERE id IN %s", (ids,))
        self.db.execute("DELETE FROM memories WHERE id IN %s", (ids,))
        self.db.commit()


dml_ops = DMLOps()
