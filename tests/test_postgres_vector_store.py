import asyncio

from shared.config.settings import env
import infra.vector_store.postgres_impl as pvs
from infra.vector_store.postgres_impl import PostgresVectorStore
from domain.memory.models import IMemoryUser


class _FakeVectorEntity:
    def __init__(self, *, _id, sector, vector, dim):
        self.id = _id
        self.sector = sector
        self.v = vector
        self.dim = dim


class _FakeScalarResult:
    def __init__(self, values):
        self._values = list(values)

    def all(self):
        return list(self._values)

    def first(self):
        return self._values[0] if self._values else None


class _FakeMappingResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _FakeExecuteResult:
    def __init__(self, *, scalar_rows=None, mapping_rows=None):
        self._scalar_rows = list(scalar_rows or [])
        self._mapping_rows = list(mapping_rows or [])

    def scalars(self):
        return _FakeScalarResult(self._scalar_rows)

    def mappings(self):
        return _FakeMappingResult(self._mapping_rows)


class _FakeSession:
    def __init__(self, execute_results=None):
        self.execute_results = list(execute_results or [])
        self.executed_statements = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, stmt):
        self.executed_statements.append(stmt)
        if self.execute_results:
            return self.execute_results.pop(0)
        return _FakeExecuteResult()

    def commit(self):
        self.committed = True


def _patch_sync_to_thread(monkeypatch):
    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(pvs.asyncio, "to_thread", _fake_to_thread)


def _make_session_factory(session):
    return lambda: session


def test_store_vector_pads_to_pgvector_dim_and_preserves_logical_dim(monkeypatch):
    monkeypatch.setattr(env, "VECTOR_DIM", 5, raising=False)
    store = PostgresVectorStore("postgresql://example")
    session = _FakeSession()
    _patch_sync_to_thread(monkeypatch)

    async def _fake_ensure_initialized():
        return None

    monkeypatch.setattr(store, "_ensure_initialized", _fake_ensure_initialized, raising=False)
    monkeypatch.setattr(pvs, "get_session_factory", lambda: _make_session_factory(session))

    asyncio.run(store.store_vector("m1", "semantic", [1.0, 2.0], 2))

    assert len(session.executed_statements) == 1
    compiled = session.executed_statements[0].compile()
    assert compiled.params["v"] == [1.0, 2.0, 0.0, 0.0, 0.0]
    assert compiled.params["dim"] == 2
    assert session.committed is True


def test_get_vector_trims_padded_storage_back_to_logical_dim(monkeypatch):
    monkeypatch.setattr(env, "VECTOR_DIM", 5, raising=False)
    store = PostgresVectorStore("postgresql://example")
    session = _FakeSession(execute_results=[
        _FakeExecuteResult(scalar_rows=[
            _FakeVectorEntity(_id="m1", sector="semantic", vector=[1.0, 2.0, 0.0, 0.0, 0.0], dim=2)
        ])
    ])
    _patch_sync_to_thread(monkeypatch)

    async def _fake_ensure_initialized():
        return None

    monkeypatch.setattr(store, "_ensure_initialized", _fake_ensure_initialized, raising=False)
    monkeypatch.setattr(pvs, "get_session_factory", lambda: _make_session_factory(session))

    row = asyncio.run(store.get_vector("m1", "semantic"))

    assert row is not None
    assert row.dim == 2
    assert row.vector == [1.0, 2.0]


def test_get_vectors_by_id_trims_all_rows_to_logical_dim(monkeypatch):
    monkeypatch.setattr(env, "VECTOR_DIM", 5, raising=False)
    store = PostgresVectorStore("postgresql://example")
    session = _FakeSession(execute_results=[
        _FakeExecuteResult(scalar_rows=[
            _FakeVectorEntity(_id="m1", sector="semantic", vector=[1.0, 2.0, 0.0, 0.0, 0.0], dim=2),
            _FakeVectorEntity(_id="m1", sector="event", vector=[3.0, 4.0, 5.0, 0.0, 0.0], dim=3),
        ])
    ])
    _patch_sync_to_thread(monkeypatch)

    async def _fake_ensure_initialized():
        return None

    monkeypatch.setattr(store, "_ensure_initialized", _fake_ensure_initialized, raising=False)
    monkeypatch.setattr(pvs, "get_session_factory", lambda: _make_session_factory(session))

    rows = asyncio.run(store.get_vectors_by_id("m1"))

    assert [r.vector for r in rows] == [[1.0, 2.0], [3.0, 4.0, 5.0]]
    assert [r.dim for r in rows] == [2, 3]


def test_search_returns_similarity_rows(monkeypatch):
    monkeypatch.setattr(env, "VECTOR_DIM", 4, raising=False)
    store = PostgresVectorStore("postgresql://example")
    session = _FakeSession(execute_results=[
        _FakeExecuteResult(mapping_rows=[
            {"id": "m1", "similarity": 0.91},
            {"id": "m2", "similarity": 0.72},
        ])
    ])
    _patch_sync_to_thread(monkeypatch)

    async def _fake_ensure_initialized():
        return None

    monkeypatch.setattr(store, "_ensure_initialized", _fake_ensure_initialized, raising=False)
    monkeypatch.setattr(pvs, "get_session_factory", lambda: _make_session_factory(session))

    rows = asyncio.run(store.search(IMemoryUser(id="u1"), [0.1, 0.2], "semantic", 2))

    assert [row.id for row in rows] == ["m1", "m2"]
    assert [row.similarity for row in rows] == [0.91, 0.72]
