import asyncio

from domain.memory.models import IMemoryUserIdentity
from infra.db.orm_models import Users
from infra.db.repos import user_repo


class _FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value

    def all(self):
        return self._value


class _FakeExecuteResult:
    def __init__(self, scalar_value):
        self._scalar_value = scalar_value

    def scalars(self):
        return _FakeScalarResult(self._scalar_value)


class _FakeSession:
    def __init__(self, *, execute_scalars=None, get_return=None):
        self._execute_scalars = list(execute_scalars or [])
        self._get_return = get_return
        self.added = []
        self.commit_called = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, _query):
        if not self._execute_scalars:
            raise AssertionError("Unexpected execute() call")
        return _FakeExecuteResult(self._execute_scalars.pop(0))

    def add(self, entity):
        self.added.append(entity)

    def get(self, _model, _id):
        return self._get_return

    def commit(self):
        self.commit_called = True


class _SessionFactoryProvider:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


def test_add_user_encrypts_summary(monkeypatch):
    fake_session = _FakeSession(execute_scalars=[None])
    monkeypatch.setattr(user_repo, "get_session_factory", lambda: _SessionFactoryProvider(fake_session))
    monkeypatch.setattr(user_repo.EncryptionKeyTool, "generate_aes_256_gcm_key", lambda: "k1")
    monkeypatch.setattr(
        user_repo,
        "encrypt_if_necessary",
        lambda value, *, key_b64, aad=None: f"enc::{value}::{key_b64}::{aad.get('id')}",
    )
    monkeypatch.setattr(
        user_repo,
        "decrypt_if_necessary",
        lambda value, *, key_b64, aad=None: value,
    )

    user_identity = IMemoryUserIdentity(user_key="u", tenant_key="t", project_key="p")
    _ = asyncio.run(user_repo.add_user(user_identity, summary="plain-user-summary"))

    assert len(fake_session.added) == 1
    inserted = fake_session.added[0]
    assert inserted.summary.startswith("enc::plain-user-summary::k1::")
    assert fake_session.commit_called is True


def test_update_user_summary_encrypts_summary(monkeypatch):
    entity = Users(id="u1", encryption_key="k1", summary="old")
    fake_session = _FakeSession(get_return=entity)
    monkeypatch.setattr(user_repo, "get_session_factory", lambda: _SessionFactoryProvider(fake_session))
    monkeypatch.setattr(
        user_repo,
        "encrypt_if_necessary",
        lambda value, *, key_b64, aad=None: f"enc::{value}::{key_b64}::{aad.get('id')}",
    )

    asyncio.run(user_repo.update_user_summary("u1", "new-summary"))

    assert entity.summary == "enc::new-summary::k1::u1"
    assert fake_session.commit_called is True


def test_get_user_by_id_decrypts_summary(monkeypatch):
    entity = Users(id="u1", tenant_key="t", project_key="p", user_key="uk", encryption_key="k1", summary="cipher")
    fake_session = _FakeSession(execute_scalars=[entity])
    monkeypatch.setattr(user_repo, "get_session_factory", lambda: _SessionFactoryProvider(fake_session))
    monkeypatch.setattr(
        user_repo,
        "decrypt_if_necessary",
        lambda value, *, key_b64, aad=None: f"dec::{value}::{key_b64}::{aad.get('id')}",
    )

    user = asyncio.run(user_repo.get_user_by_id("u1"))

    assert user is not None
    assert user.summary == "dec::cipher::k1::u1"

