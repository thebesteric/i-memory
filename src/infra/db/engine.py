from sqlite3 import OperationalError
from typing import Optional, Tuple, Dict, List, Any
from functools import lru_cache

import psycopg2
from agile.utils import LogHelper, singleton
from psycopg2.extras import DictCursor
from psycopg2.pool import ThreadedConnectionPool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import inspect
import logging

from shared.config.settings import env
from infra.db.orm_models import normalize_sync_postgres_url

logger = LogHelper.get_logger(title="[DB]")


@lru_cache(maxsize=1)
def get_sync_engine():
    """Return a cached synchronous SQLAlchemy engine for ORM CRUD modules."""
    connect_uri, database = normalize_sync_postgres_url(env.POSTGRES_DB_URL)
    if not database:
        raise ValueError("Database name is missing from POSTGRES_DB_URL")
    engine = create_engine(f"{connect_uri}/{database}", pool_pre_ping=True)
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    """Return a cached SQLAlchemy session factory for ORM CRUD modules."""
    engine = get_sync_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)


def _get_caller_info(skip_modules: Tuple[str, ...] = ("src.core.db",)) -> str:
    """
    Return a short string describing the first non-db caller in the stack.
    Format: module:function:lineno
    """
    try:
        for frame_info in inspect.stack()[2:]:  # skip current and immediate caller
            module = inspect.getmodule(frame_info.frame)
            module_name = module.__name__ if module else frame_info.filename
            if not any(module_name.startswith(m) for m in skip_modules):
                return f"{module_name}:{frame_info.function}:{frame_info.lineno}"
        # fallback to the immediate caller frame
        fi = inspect.stack()[2]
        module = inspect.getmodule(fi.frame)
        return f"{module.__name__ if module else fi.filename}:{fi.function}:{fi.lineno}"
    except Exception as e:
        logger.warning(f"Failed to get caller_info: {e}")
        return "unknown:unknown:0"


@singleton
class DB:

    def __init__(self, db_url: str | None = None, autocommit: bool | None = True):
        self.db_url = db_url if db_url else env.POSTGRES_DB_URL
        self.autocommit = autocommit if autocommit is not None else env.POSTGRES_DB_AUTOCOMMIT
        self._pool: Optional[ThreadedConnectionPool] = None
        self._pool_min_conn = 1
        self._pool_max_conn = 10
        self.conn_kwargs = {
            "sslmode": "disable"
        }

    def get_pool(self) -> ThreadedConnectionPool:
        if self._pool:
            return self._pool
        if not self.db_url.startswith("postgresql://"):
            raise ValueError("Unsupported database URL schema: postgresql://. Only PostgreSQL is supported currently.")
        try:
            self._pool = ThreadedConnectionPool(
                self._pool_min_conn,
                self._pool_max_conn,
                self.db_url,
                **self.conn_kwargs,
            )
            logger.info(
                f"Connection pool created for [{env.POSTGRES_DB_NAME}] "
                f"(min={self._pool_min_conn}, max={self._pool_max_conn})"
            )
            if self._pool is None:
                raise OperationalError("Connection pool initialization failed unexpectedly.")
            return self._pool
        except psycopg2.OperationalError as e:
            raise OperationalError(
                f"Connect to PostgreSQL failed: {str(e)}. "
                f"Please ensure init_db_schema has initialized database/schema."
            ) from e

    def get_conn(self) -> psycopg2.extensions.connection:
        pool = self.get_pool()
        conn = pool.getconn()
        conn.autocommit = self.autocommit
        return conn

    def put_conn(self, conn: psycopg2.extensions.connection) -> None:
        if self._pool and conn and not conn.closed:
            self._pool.putconn(conn)

    def connect(self):
        """
        连接数据库并预热连接池
        :return:
        """
        conn = self.get_conn()
        self.put_conn(conn)

    def execute(self, sql: str, params: Tuple | None = None, conn=None) -> int:
        """
        执行增删改SQL语句，返回受影响行数
        :param sql: SQL语句
        :param params: 参数
        :param conn: 可选，外部传入的数据库连接
        :return:
        """
        external_conn = conn is not None
        if not external_conn:
            conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                affected_rows = cur.rowcount
                # include caller info in the log so it's clear who invoked the DB operation
                caller = _get_caller_info() if logger.isEnabledFor(logging.INFO) else "unknown:unknown:0"
                logger.info(f"Caller: ({caller}) Execution successful, affected rows: {affected_rows}")
                return affected_rows
        except Exception as e:
            conn.rollback()
            caller = _get_caller_info() if logger.isEnabledFor(logging.INFO) else "unknown:unknown:0"
            raise Exception(f"Caller: ({caller}) Execution failed: {str(e)}") from e
        finally:
            if not external_conn:
                self.put_conn(conn)

    def fetchone(self, sql: str, params: Optional[Tuple] = None, conn=None) -> Dict[str, Any] | None:
        """
        查询单条数据，返回字典格式
        :param sql: SQL查询语句
        :param params: 查询参数
        :param conn: 可选，外部传入的数据库连接
        :return: 查询结果
        """
        external_conn = conn is not None
        if not external_conn:
            conn = self.get_conn()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, params)
                result = cur.fetchone()
                if result:
                    return dict(result)
                return None
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}") from e
        finally:
            if not external_conn:
                self.put_conn(conn)

    def fetchall(self, sql: str, params: Optional[Tuple] = None, conn=None) -> List[Dict[str, Any]]:
        """
        查询多条数据，返回字典列表格式
        :param sql: SQL查询语句
        :param params: 查询参数
        :param conn: 可选，外部传入的数据库连接
        :return: 查询结果列表
        """
        external_conn = conn is not None
        if not external_conn:
            conn = self.get_conn()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, params)
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}") from e
        finally:
            if not external_conn:
                self.put_conn(conn)

    def commit(self, conn: Optional[psycopg2.extensions.connection] = None) -> None:
        """
        提交事务
        :return:
        """
        if conn and not self.autocommit:
            conn.commit()

    def close(self) -> None:
        """
        关闭数据库连接
        :return:
        """
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Database connection pool closed.")

    def __enter__(self):
        """
        上下文管理器：自动连接
        :return:
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器：自动关闭"""
        self.close()
        return False


def transaction():
    """
    获取数据库连接，用于事务处理
    :return:
    """
    db_instance: DB = get_db()
    conn = db_instance.get_conn()
    original_autocommit = db_instance.autocommit
    original_conn_autocommit = conn.autocommit
    try:
        db_instance.autocommit = False
        conn.autocommit = False
        yield conn
        db_instance.commit(conn)
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        db_instance.autocommit = original_autocommit
        conn.autocommit = original_conn_autocommit
        db_instance.put_conn(conn)


# 全局唯一实例
def get_db() -> DB:
    """
    获取数据库实例
    :return:
    """
    db_instance = DB()
    return db_instance
