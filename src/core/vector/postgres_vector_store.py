import asyncio
import time
from typing import Any, List, Optional, cast

from sqlalchemy import delete, literal, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from agile.utils import LogHelper, singleton, timing

from src.core.config import env
from src.core.db import get_session_factory, get_sync_engine
from src.core.vector.base_vector_store import BaseVectorStore, VectorRow, VectorSearch
from src.entity.db_schema import Base, Vectors
from src.memory.memory_models import IMemoryUser

logger = LogHelper.get_logger(title="[POSTGRES]")


@singleton
class PostgresVectorStore(BaseVectorStore):
    def __init__(self, dsn: str, vector_table_name: str = "vectors"):
        self.dsn = dsn
        self.vector_table_name = vector_table_name or "vectors"
        self.initialized = False
        self._pool_init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """
        确保 pgvector 扩展、表结构和索引已准备就绪。
        :return:
        """
        if self.initialized:
            return None

        async with self._pool_init_lock:
            if self.initialized:
                return None

            init_start = time.perf_counter()

            def _init_schema():
                engine = get_sync_engine()
                with engine.begin() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    logger.info("Plugin pgvector extension enabled")
                Base.metadata.create_all(engine, tables=[cast(Any, Vectors.__table__)])
                logger.info(f"Vector table and indexes ensured for {self.vector_table_name}")

            await asyncio.to_thread(_init_schema)
            self.initialized = True
            logger.info(
                f"PostgresVectorStore initialized for table={self.vector_table_name} "
                f"in {(time.perf_counter() - init_start) * 1000:.2f}ms"
            )

        return None

    async def warmup(self):
        """在服务启动阶段主动初始化连接池与表结构。"""
        warmup_start = time.perf_counter()
        was_initialized = self.initialized
        await self._ensure_initialized()
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

    @staticmethod
    def _target_vector_dim() -> int:
        return max(1, int(getattr(env, "VECTOR_DIM", 1536) or 1536))

    @classmethod
    def _coerce_vector_for_storage(cls, vector: List[float]) -> List[float]:
        """Pad/truncate vectors so they always fit the fixed pgvector column."""
        target_dim = cls._target_vector_dim()
        vec = list(vector) if vector is not None else []
        if len(vec) > target_dim:
            return vec[:target_dim]
        if len(vec) < target_dim:
            return vec + [0.0] * (target_dim - len(vec))
        return vec

    @staticmethod
    def _coerce_vector_for_read(vector: List[float], dim: int) -> List[float]:
        vec = list(vector) if vector is not None else []
        try:
            dim_int = int(dim)
        except (TypeError, ValueError):
            dim_int = 0
        if dim_int > 0:
            return vec[:dim_int]
        return vec

    async def store_vector(self, _id: str, sector: str, vector: List[float], dim: int, user: IMemoryUser | None = None):
        """
        存储向量到 PostgreSQL 数据库
        :param _id: 唯一标识
        :param sector: 扇区
        :param vector: 向量
        :param dim: 维度
        :param user: 用户
        :return:
        """
        await self._ensure_initialized()
        src_vec = list(vector) if vector is not None else []
        logical_dim = max(0, min(len(src_vec), int(dim) if dim is not None else len(src_vec)))
        stored_vec = self._coerce_vector_for_storage(src_vec)

        user_id = user.id if user else None

        def _store():
            stmt = pg_insert(Vectors).values(id=_id, sector=sector, user_id=user_id, v=stored_vec, dim=logical_dim)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Vectors.id, Vectors.sector],
                set_={
                    "user_id": stmt.excluded.user_id,
                    "v": stmt.excluded.v,
                    "dim": stmt.excluded.dim,
                },
            )
            session_factory = get_session_factory()
            with session_factory() as session:
                session.execute(stmt)
                session.commit()

        await asyncio.to_thread(_store)

    async def get_vectors_by_id(self, id: str) -> List[VectorRow]:
        """
        获取指定 ID 的所有向量
        :param id: 唯一标识
        :return:
        """
        await self._ensure_initialized()

        def _load():
            session_factory = get_session_factory()
            with session_factory() as session:
                rows = session.execute(select(Vectors).where(Vectors.id == id)).scalars().all()
                return [
                    VectorRow(row.id, row.sector, self._coerce_vector_for_read(row.v, row.dim), row.dim)
                    for row in rows
                ]

        return await asyncio.to_thread(_load)

    async def get_vector(self, id: str, sector: str) -> Optional[VectorRow]:
        """
        获取指定 ID 和 sector 的向量
        :param id: 唯一标识
        :param sector: 扇区
        :return:
        """
        await self._ensure_initialized()

        def _load_one():
            session_factory = get_session_factory()
            with session_factory() as session:
                row = session.execute(
                    select(Vectors).where(Vectors.id == id, Vectors.sector == sector).limit(1)
                ).scalars().first()
                if not row:
                    return None
                return VectorRow(row.id, row.sector, self._coerce_vector_for_read(row.v, row.dim), row.dim)

        return await asyncio.to_thread(_load_one)

    async def delete_vectors(self, id: str):
        """
        删除指定 ID 的所有向量
        :param id: 唯一标识
        :return:
        """
        await self._ensure_initialized()

        def _delete():
            session_factory = get_session_factory()
            with session_factory() as session:
                session.execute(delete(Vectors).where(Vectors.id == id))
                session.commit()

        await asyncio.to_thread(_delete)

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
        await self._ensure_initialized()
        stored_vec = self._coerce_vector_for_storage(vector)

        def _search():
            session_factory = get_session_factory()
            distance_expr = Vectors.v.op("<=>")(stored_vec)
            similarity_expr = (literal(1.0) - distance_expr).label("similarity")
            query = (
                select(Vectors.id, similarity_expr)
                .where(Vectors.sector == sector, Vectors.user_id == user.id)
                .order_by(distance_expr)
                .limit(top_k)
            )
            with session_factory() as session:
                rows = session.execute(query).mappings().all()
                return [VectorSearch(id=row["id"], similarity=float(row["similarity"])) for row in rows]

        return await asyncio.to_thread(_search)

