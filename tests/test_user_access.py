import asyncio

import pytest

from domain.common.exceptions import UserNotFoundError
from domain.memory.models import IMemoryUser, IMemoryUserIdentity
from services.commons import user_access


def _identity() -> IMemoryUserIdentity:
    return IMemoryUserIdentity(user_key="u", tenant_key="t", project_key="p")


def test_get_user_for_access_auto_register_when_force_disabled(monkeypatch):
    created_user = IMemoryUser(id="new-user", user_key="u", tenant_key="t", project_key="p")

    async def fake_get_user(*, user_identity, using_cache=False):
        return None

    async def fake_add_user(user_identity):
        return created_user

    monkeypatch.setattr(user_access.env, "USER_REGISTER_FORCE", False)
    monkeypatch.setattr(user_access.user_repo, "get_user", fake_get_user)
    monkeypatch.setattr(user_access.user_repo, "add_user", fake_add_user)

    user = asyncio.run(user_access.get_user_for_access(_identity(), using_cache=True))

    assert user.id == "new-user"


def test_get_user_for_access_raises_when_force_enabled(monkeypatch):
    async def fake_get_user(*, user_identity, using_cache=False):
        return None

    monkeypatch.setattr(user_access.env, "USER_REGISTER_FORCE", True)
    monkeypatch.setattr(user_access.user_repo, "get_user", fake_get_user)

    with pytest.raises(UserNotFoundError):
        asyncio.run(user_access.get_user_for_access(_identity()))

