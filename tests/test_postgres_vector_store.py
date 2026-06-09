import asyncio
import json

from src.core.config import env
from src.core.vector.postgres_vector_store import PostgresVectorStore


class _FakeConn:
    def __init__(self, fetchrow_result=None, fetch_result=None):
        self.fetchrow_result = fetchrow_result
        self.fetch_result = fetch_result or []
        self.execute_calls = []

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        return self.fetch_result

    async def fetchrow(self, sql, *args):
        return self.fetchrow_result


class _FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _FakeAcquire(self.conn)


def test_store_vector_pads_to_pgvector_dim_and_preserves_logical_dim(monkeypatch):
    monkeypatch.setattr(env, "VECTOR_DIM", 5, raising=False)
    store = PostgresVectorStore("postgresql://example")
    conn = _FakeConn()
    async def _fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(store, "_get_pool", _fake_get_pool, raising=False)

    asyncio.run(store.store_vector("m1", "semantic", [1.0, 2.0], 2))

    assert len(conn.execute_calls) == 1
    _, args = conn.execute_calls[0]
    stored_vec = json.loads(args[3])
    assert stored_vec == [1.0, 2.0, 0.0, 0.0, 0.0]
    assert args[4] == 2


def test_get_vector_trims_padded_storage_back_to_logical_dim(monkeypatch):
    monkeypatch.setattr(env, "VECTOR_DIM", 5, raising=False)
    store = PostgresVectorStore("postgresql://example")
    conn = _FakeConn(fetchrow_result={
        "id": "m1",
        "sector": "semantic",
        "v_txt": json.dumps([1.0, 2.0, 0.0, 0.0, 0.0]),
        "dim": 2,
    })
    async def _fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(store, "_get_pool", _fake_get_pool, raising=False)

    row = asyncio.run(store.get_vector("m1", "semantic"))

    assert row is not None
    assert row.dim == 2
    assert row.vector == [1.0, 2.0]


def test_get_vectors_by_id_trims_all_rows_to_logical_dim(monkeypatch):
    monkeypatch.setattr(env, "VECTOR_DIM", 5, raising=False)
    store = PostgresVectorStore("postgresql://example")
    conn = _FakeConn(fetch_result=[
        {
            "id": "m1",
            "sector": "semantic",
            "v_txt": json.dumps([1.0, 2.0, 0.0, 0.0, 0.0]),
            "dim": 2,
        },
        {
            "id": "m1",
            "sector": "event",
            "v_txt": json.dumps([3.0, 4.0, 5.0, 0.0, 0.0]),
            "dim": 3,
        },
    ])
    async def _fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(store, "_get_pool", _fake_get_pool, raising=False)

    rows = asyncio.run(store.get_vectors_by_id("m1"))

    assert [r.vector for r in rows] == [[1.0, 2.0], [3.0, 4.0, 5.0]]
    assert [r.dim for r in rows] == [2, 3]

