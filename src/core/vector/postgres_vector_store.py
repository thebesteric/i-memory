import asyncio
import json
import time
from typing import List, Optional

import asyncpg
from agile.utils import LogHelper, singleton, timing
from asyncpg import InvalidColumnReferenceError

from src.core.config import env
from src.core.vector.base_vector_store import BaseVectorStore, VectorRow, VectorSearch
from src.memory.memory_models import IMemoryUser

logger = LogHelper.get_logger(title="[POSTGRES]")


@singleton
class PostgresVectorStore(BaseVectorStore):
    def __init__(self, dsn: str, vector_table_name: str = "vectors"):
        self.dsn = dsn
        self.vector_table_name = vector_table_name or "vectors"
        self.pool = None
        self.initialized = False
        self._pool_init_lock = asyncio.Lock()

    async def _get_pool(self):
        """
        获取数据库连接池
        该方法会确保 pgvector 扩展已启用，并初始化表结构和索引
        :return:
        """
        if self.pool and self.initialized:
            return self.pool

        async with self._pool_init_lock:
            if self.pool and self.initialized:
                return self.pool

            init_start = time.perf_counter()
            pool = await asyncpg.create_pool(self.dsn)
            try:
                async with pool.acquire() as conn:
                    # 确认 pgvector 扩展已启用
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    logger.info("Plugin pgvector extension enabled")

                    # 初始化向量表结构
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.vector_table_name} (
                            id TEXT,
                            sector TEXT NOT NULL,
                            user_id TEXT,
                            v vector({env.VECTOR_DIM}),
                            dim INTEGER,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            PRIMARY KEY (id, sector)
                        )
                    """)

                    # 创建 HNSW 索引以加速向量相似度搜索
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.vector_table_name}_hnsw_idx
                        ON {self.vector_table_name} USING hnsw (v vector_cosine_ops)
                    """)
                    logger.info(f"HNSW index created on {self.vector_table_name} for fast ANN queries")
            except Exception:
                await pool.close()
                raise

            self.pool = pool
            self.initialized = True
            logger.info(
                f"PostgresVectorStore initialized for table={self.vector_table_name} "
                f"in {(time.perf_counter() - init_start) * 1000:.2f}ms"
            )

        return self.pool

    async def warmup(self):
        """在服务启动阶段主动初始化连接池与表结构。"""
        warmup_start = time.perf_counter()
        was_initialized = self.initialized
        await self._get_pool()
        if was_initialized:
            logger.info(
                f"PostgresVectorStore warmup skipped (already initialized), "
                f"cost {(time.perf_counter() - warmup_start) * 1000:.2f}ms"
            )
            return

        logger.info(
            f"PostgresVectorStore warmup completed, "
            f"cost {(time.perf_counter() - warmup_start) * 1000:.2f}ms"
        )

    async def store_vector(self, _id: str, sector: str, vector: List[float], dim: int, user: IMemoryUser = None):
        """
        存储向量到 PostgreSQL 数据库
        :param _id: 唯一标识
        :param sector: 扇区
        :param vector: 向量
        :param dim: 维度
        :param user: 用户
        :return:
        """
        pool = await self._get_pool()
        vec_str = str(vector)

        user_id = user.id if user else None

        sql = f"""
            INSERT INTO {self.vector_table_name} (id, sector, user_id, v, dim)
            VALUES ($1, $2, $3, $4::vector, $5)
            ON CONFLICT (id, sector) DO UPDATE SET
                user_id = COALESCE(EXCLUDED.user_id, {self.vector_table_name}.user_id),
                v = EXCLUDED.v
        """
        async with pool.acquire() as conn:
            try:
                await conn.execute(sql, _id, sector, user_id, vec_str, dim)
            except InvalidColumnReferenceError as ex:
                # This typically means there's no unique constraint on `id` for ON CONFLICT to reference.
                # Create a unique index on id and retry once. If there are duplicate ids in the table, creating
                # the unique index will fail, and we should let that error surface.
                logger.warning("ON CONFLICT failed because no unique constraint on id; creating unique index and retrying")
                await conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {self.vector_table_name}_id_idx ON {self.vector_table_name} (id)")
                await conn.execute(sql, _id, sector, user_id, vec_str, dim)

    async def get_vectors_by_id(self, id: str) -> List[VectorRow]:
        """
        获取指定 ID 的所有向量
        :param id: 唯一标识
        :return:
        """
        pool = await self._get_pool()
        sql = f"SELECT id, sector, v::text as v_txt, dim FROM {self.vector_table_name} WHERE id=$1"
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, id)

        res = []
        for r in rows:
            vec = json.loads(r["v_txt"])
            res.append(VectorRow(r["id"], r["sector"], vec, r["dim"]))
        return res

    async def get_vector(self, id: str, sector: str) -> Optional[VectorRow]:
        """
        获取指定 ID 和 sector 的向量
        :param id: 唯一标识
        :param sector: 扇区
        :return:
        """
        pool = await self._get_pool()
        sql = f"SELECT id, sector, v::text as v_txt, dim FROM {self.vector_table_name} WHERE id=$1 AND sector=$2"
        async with pool.acquire() as conn:
            r = await conn.fetchrow(sql, id, sector)

        if not r:
            return None
        vec = json.loads(r["v_txt"])
        return VectorRow(r["id"], r["sector"], vec, r["dim"])

    async def delete_vectors(self, id: str):
        """
        删除指定 ID 的所有向量
        :param id: 唯一标识
        :return:
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self.vector_table_name} WHERE id=$1", id)

    @timing
    async def search(self, user: IMemoryUser, vector: List[float], sector: str, top_k: int) -> List[VectorSearch]:
        """
        相似度搜索
        :param user: 用户
        :param vector: 向量
        :param sector: 扇区
        :param top_k: 返回的相似向量数量
        :return: 相似向量列表
        """
        pool = await self._get_pool()
        vec_str = str(vector)

        filter_sql = " AND sector=$2"
        args = [vec_str, sector]
        arg_idx = 3

        filter_sql += f" AND user_id=${arg_idx}"
        args.append(user.id)
        arg_idx += 1

        sql = f"""
            SELECT id, 1 - (v <=> $1::vector) as similarity
            FROM {self.vector_table_name}
            WHERE 1=1 {filter_sql}
            ORDER BY v <=> $1::vector
            LIMIT {top_k}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)

        return [VectorSearch(id=row["id"], similarity=float(row["similarity"])) for row in rows]
