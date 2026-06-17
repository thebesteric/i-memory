import json

from src.core.mem_ops import mem_ops
from src.memory.entity.db_schema import Memories
from src.memory.memory_models import IMemoryUser


class _FakeResult:
    def __init__(self, *, rows=None, scalar_value=None, rowcount=None):
        self._rows = rows or []
        self._scalar_value = scalar_value
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one(self):
        return self._scalar_value


class _FakeSession:
    def __init__(self, *, results=None):
        self._results = list(results or [])
        self.commit_called = False
        self.execute_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, stmt):
        _ = stmt
        self.execute_count += 1
        if self._results:
            return self._results.pop(0)
        return _FakeResult()

    def commit(self):
        self.commit_called = True


class _SessionFactoryProvider:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


def test_find_mem_by_conditions_empty_short_circuit(monkeypatch):
    def _should_not_be_called():
        raise AssertionError("session factory should not be called for empty conditions")

    monkeypatch.setattr(mem_ops, "_session_factory", _should_not_be_called)
    assert mem_ops.find_mem_by_conditions(conditions=[]) == []


def test_find_mem_by_user_serializes_vector_list(monkeypatch):
    mem = Memories(id="m1", user_id="u1", content="hello", primary_sector="semantic")
    fake_session = _FakeSession(results=[_FakeResult(rows=[(mem, [0.1, 0.2], "semantic", 2)])])
    monkeypatch.setattr(mem_ops, "_session_factory", lambda: _SessionFactoryProvider(fake_session))

    user = IMemoryUser(id="u1", tenant_key="t", project_key="p", user_key="uk")
    result = mem_ops.find_mem_by_user(user=user, order_by=["t.created_at DESC"], limit=10, offset=0)

    assert len(result) == 1
    assert result[0]["id"] == "m1"
    assert json.loads(result[0]["v"]) == [0.1, 0.2]


def test_count_mem_by_user_returns_int(monkeypatch):
    fake_session = _FakeSession(results=[_FakeResult(scalar_value=5)])
    monkeypatch.setattr(mem_ops, "_session_factory", lambda: _SessionFactoryProvider(fake_session))

    user = IMemoryUser(id="u1", tenant_key="t", project_key="p", user_key="uk")
    count = mem_ops.count_mem_by_user(user, conditions=["fact_joined = 0"])

    assert count == 5


def test_del_mem_by_user_commits_and_returns_rowcount(monkeypatch):
    fake_results = [_FakeResult() for _ in range(11)] + [_FakeResult(rowcount=7)]
    fake_session = _FakeSession(results=fake_results)
    monkeypatch.setattr(mem_ops, "_session_factory", lambda: _SessionFactoryProvider(fake_session))

    user = IMemoryUser(id="u1", tenant_key="t", project_key="p", user_key="uk")
    affected = mem_ops.del_mem_by_user(user)

    assert affected == 7
    assert fake_session.commit_called is True
    assert fake_session.execute_count == 12

