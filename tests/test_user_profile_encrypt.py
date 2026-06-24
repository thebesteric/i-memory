import asyncio

from domain.memory.models import IMemoryUser
from domain.profile.models import UserProfile, Demographic, Location, Preferences, Attributes, Tag
from infra.db.orm_models import UserProfiles
from services.profile import user_profile_ops


class _FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value


class _FakeExecuteResult:
    def __init__(self, scalar_value):
        self._scalar_value = scalar_value

    def scalars(self):
        return _FakeScalarResult(self._scalar_value)


class _FakeSession:
    def __init__(self, *, scalar_value=None):
        self.scalar_value = scalar_value
        self.added = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, _stmt):
        return _FakeExecuteResult(self.scalar_value)

    def add(self, entity):
        self.added.append(entity)


class _SessionFactoryProvider:
    def __init__(self, fake_session):
        self.fake_session = fake_session

    def __call__(self):
        return self.fake_session


def test_upsert_user_profile_encrypts_json_values(monkeypatch):
    user = IMemoryUser(id="u1", tenant_key="t", project_key="p", user_key="uk", encryption_key="k1")
    profile = UserProfile(
        demographic=Demographic(occupation="Engineer", location=Location(city="Shenzhen"), extra={"age": 30}),
        preferences=Preferences(extra={"hobby": "swim"}),
        attributes=Attributes(extra={"motto": "stay hungry"}),
        tags=[Tag(name="tech-lover", weight=0.8, sub_tags=["ai"])],
    )
    fake_session = _FakeSession()

    monkeypatch.setattr(
        user_profile_ops,
        "encrypt_if_necessary",
        lambda value, *, key_b64, aad=None: f"enc({value})",
    )

    asyncio.run(user_profile_ops.upsert_user_profile(user, profile, conn=fake_session))

    assert len(fake_session.added) == 1
    inserted = fake_session.added[0]
    assert inserted.demographic["occupation"] == "jsonenc::enc(Engineer)"
    assert inserted.demographic["location"]["city"] == "jsonenc::enc(Shenzhen)"
    assert inserted.demographic["extra"]["age"] == 30
    assert inserted.preferences["extra"]["hobby"] == "jsonenc::enc(swim)"
    assert inserted.attributes["extra"]["motto"] == "jsonenc::enc(stay hungry)"
    assert inserted.tags[0]["name"] == "jsonenc::enc(tech-lover)"


def test_get_user_profile_decrypts_json_values(monkeypatch):
    user = IMemoryUser(id="u1", tenant_key="t", project_key="p", user_key="uk", encryption_key="k1")

    entity = UserProfiles(
        id="p1",
        user_id="u1",
        demographic={
            "occupation": "jsonenc::enc(Engineer)",
            "location": {"city": "jsonenc::enc(Shenzhen)"},
            "personality": [],
            "extra": {"age": 30},
        },
        preferences={"habits": [], "extra": {"hobby": "jsonenc::enc(swim)"}},
        attributes={"extra": {"motto": "jsonenc::enc(stay hungry)"}},
        tags=[{"name": "jsonenc::enc(tech-lover)", "weight": 0.8, "sub_tags": ["ai"]}],
        is_active=True,
    )

    fake_session = _FakeSession(scalar_value=entity)
    monkeypatch.setattr(user_profile_ops, "get_session_factory", lambda: _SessionFactoryProvider(fake_session))
    monkeypatch.setattr(
        user_profile_ops,
        "decrypt_if_necessary",
        lambda value, *, key_b64, aad=None: value.removeprefix("enc(").removesuffix(")"),
    )

    result = asyncio.run(user_profile_ops.get_user_profile(user, query_cache=False))

    assert result is not None
    assert result.demographic.occupation == "Engineer"
    assert result.demographic.location.city == "Shenzhen"
    assert result.preferences.extra["hobby"] == "swim"
    assert result.attributes.extra["motto"] == "stay hungry"
    assert result.tags[0].name == "tech-lover"

