import asyncio

from domain.memory.models import IMemoryUser
from domain.session.models import Session, SessionCollection
from services.session import session_ops


class _FakeEmbedModel:
    async def embed(self, _text: str):
        return [0.1, 0.2, 0.3]


class _FakeSelectResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, *, execute_result=None):
        self.execute_result = execute_result
        self.added = []
        self.commit_called = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def add(self, value):
        self.added.append(value)

    def execute(self, _stmt):
        if self.execute_result is None:
            raise AssertionError("Unexpected execute() call")
        return self.execute_result

    def commit(self):
        self.commit_called = True

    def close(self):
        self.closed = True


class _SessionFactoryProvider:
    def __init__(self, fake_session):
        self.fake_session = fake_session

    def __call__(self):
        return self.fake_session


def test_insert_sessions_encrypts_summary(monkeypatch):
    user = IMemoryUser(id="u1", tenant_key="t", project_key="p", user_key="uk", encryption_key="k1")
    sessions = SessionCollection(sessions=[Session(summary="plain-summary", dialogue_ids=["d1"], key_facts=["f1"])])

    fake_session = _FakeSession()

    monkeypatch.setattr(session_ops, "get_embed_model", lambda: _FakeEmbedModel())
    monkeypatch.setattr(session_ops, "get_session_factory", lambda: _SessionFactoryProvider(fake_session))
    monkeypatch.setattr(
        session_ops,
        "encrypt_if_necessary",
        lambda value, *, key_b64, aad=None: f"enc::{value}::{key_b64}::{aad.get('id')}",
    )

    asyncio.run(session_ops.insert_sessions(user, sessions))

    assert len(fake_session.added) == 1
    inserted = fake_session.added[0]
    assert inserted.summary.startswith("enc::plain-summary::k1::")
    assert fake_session.commit_called is True
    assert fake_session.closed is True


def test_session_search_decrypts_summary(monkeypatch):
    user = IMemoryUser(id="u1", tenant_key="t", project_key="p", user_key="uk", encryption_key="k1")
    fake_rows = [
        {
            "id": "s1",
            "user_id": "u1",
            "summary": "enc-payload",
            "vector": "[0.1, 0.2, 0.3]",
            "dialogue_ids": ["d1"],
            "key_facts": ["f1"],
            "similarity": 0.9,
        }
    ]
    fake_session = _FakeSession(execute_result=_FakeSelectResult(fake_rows))

    monkeypatch.setattr(session_ops, "embed_model", _FakeEmbedModel())
    monkeypatch.setattr(session_ops, "get_session_factory", lambda: _SessionFactoryProvider(fake_session))
    monkeypatch.setattr(
        session_ops,
        "decrypt_if_necessary",
        lambda value, *, key_b64, aad=None: f"dec::{value}::{key_b64}::{aad.get('id')}",
    )

    result = asyncio.run(session_ops.session_search(user, "hello", top_k=1))

    assert len(result.sessions) == 1
    assert result.sessions[0].summary == "dec::enc-payload::k1::s1"
    assert result.sessions[0].id == "s1"
    assert result.sessions[0].similarity == 0.9

