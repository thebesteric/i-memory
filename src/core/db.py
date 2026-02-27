import os
import time
from pathlib import Path
from sqlite3 import OperationalError
from typing import Optional, Tuple, Dict, List, Any

import psycopg2
from psycopg2.extras import DictCursor
from psycopg2.pool import ThreadedConnectionPool
import inspect
import logging

from utils.log_helper import LogHelper
from utils.singleton import singleton

from src.core.config import env

logger = LogHelper.get_logger()


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
        logger.warning(f"[DB] Failed to get caller_info: {e}")
        return "unknown:unknown:0"


@singleton
class DB:

    def __init__(self, db_url: str = None, autocommit: bool = True):
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
        # Ensure database exists before creating the pool
        self._ensure_database_exists()
        try:
            self._pool = ThreadedConnectionPool(
                self._pool_min_conn,
                self._pool_max_conn,
                self.db_url,
                **self.conn_kwargs,
            )
            logger.info(
                f"[DB] Connection pool created for [{env.POSTGRES_DB_NAME}] "
                f"(min={self._pool_min_conn}, max={self._pool_max_conn})"
            )
            return self._pool
        except psycopg2.OperationalError as e:
            raise OperationalError(f"Connect to PostgreSQL failed: {str(e)}") from e

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
        连接数据库
        :return:
        """
        conn = self.get_conn()
        try:
            # 初始化数据库结构
            self.init_schema(conn)
        finally:
            self.put_conn(conn)

    def _ensure_database_exists(self):
        """
        确保数据库存在，如果不存在则创建
        :return:
        """
        base_url = f"postgresql://{env.POSTGRES_DB_USER}:{env.POSTGRES_DB_PASSWORD}@{env.POSTGRES_DB_HOST}:{env.POSTGRES_DB_PORT}/postgres"
        try:
            conn = psycopg2.connect(base_url, **self.conn_kwargs)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (env.POSTGRES_DB_NAME,))
            exists = cur.fetchone()
            if not exists:
                logger.info(f"[DB] Creating database {env.POSTGRES_DB_NAME} successfully.")
                cur.execute(f'CREATE DATABASE "{env.POSTGRES_DB_NAME}"')
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"[DB] Failed to create database {env.POSTGRES_DB_NAME}: {e}")
            raise

    def _run_migrations(self, conn: psycopg2.extensions.connection):
        """
        初始化数据库结构
        :return:
        """
        try:
            with conn.cursor() as c:
                c.execute("CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at BIGINT)")
                files = []
                mig_path = Path(__file__).parent.parent / "migrations"
                if mig_path.exists():
                    files = [f for f in os.listdir(mig_path) if f.endswith(".sql")]
                files.sort()
                for f in files:
                    c.execute("SELECT 1 FROM _migrations WHERE name = %s", (f,))
                    if not c.fetchone():
                        logger.info(f"[DB] Applying migration {f}")
                        sql = (mig_path / f).read_text(encoding="utf-8")
                        for statement in [s.strip() for s in sql.split(';') if s.strip()]:
                            try:
                                c.execute(statement)
                            except Exception as e:
                                logger.error(f"[DB] Migration statement failed: {statement}, Error: {e}")
                                conn.rollback()
                                raise
                        c.execute("INSERT INTO _migrations (name, applied_at) VALUES (%s, %s)", (f, int(time.time())))
            # 迁移完成后强制提交（确保会话状态同步）
            self.commit(conn)
        except Exception as e:
            conn.rollback()
            raise e

    def init_schema(self, conn: psycopg2.extensions.connection):
        """
        初始化数据库结构
        :return:
        """
        self._run_migrations(conn)

    def execute(self, sql: str, params: Tuple = None) -> int:
        """
        执行增删改SQL语句，返回受影响行数
        :param sql: SQL语句
        :param params: 参数
        :return:
        """
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                affected_rows = cur.rowcount
                # include caller info in the log so it's clear who invoked the DB operation
                caller = _get_caller_info() if logger.isEnabledFor(logging.INFO) else "unknown:unknown:0"
                logger.info(f"[DB] (caller: {caller}) Execution successful, affected rows: {affected_rows}")
                return affected_rows
        except Exception as e:
            conn.rollback()
            caller = _get_caller_info() if logger.isEnabledFor(logging.INFO) else "unknown:unknown:0"
            raise Exception(f"[DB] (caller: {caller}) Execution failed: {str(e)}") from e
        finally:
            self.put_conn(conn)

    def fetchone(self, sql: str, params: Optional[Tuple] = None) -> Dict[str, Any] | None:
        """
        查询单条数据，返回字典格式
        :param sql: SQL查询语句
        :param params: 查询参数
        :return: 查询结果
        """
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
            self.put_conn(conn)

    def fetchall(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        查询多条数据，返回字典列表格式
        :param sql: SQL查询语句
        :param params: 查询参数
        :return: 查询结果列表
        """
        conn = self.get_conn()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, params)
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}") from e
        finally:
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
            logger.info("[DB] Database connection pool closed.")

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
