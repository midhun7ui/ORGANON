import pytest

from core.security.base import SecurityProvider, UserContext


def test_security_provider_is_abstract():
    with pytest.raises(TypeError):
        SecurityProvider()


class FakeSecurityProvider(SecurityProvider):

    async def authenticate(self, credentials):
        return UserContext(
            user_id="user-001",
            tenant_id="tenant-001",
            permissions={"document:read"},
        )

    async def authorize(self, user, action):
        return action in user.permissions


@pytest.mark.asyncio
async def test_security_provider_authentication():
    provider = FakeSecurityProvider()

    user = await provider.authenticate({})

    assert isinstance(user, UserContext)
    assert user.user_id == "user-001"
    assert user.tenant_id == "tenant-001"
    assert "document:read" in user.permissions


@pytest.mark.asyncio
async def test_security_provider_authorization():
    provider = FakeSecurityProvider()

    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-001",
        permissions={"document:read"},
    )

    assert await provider.authorize(user, "document:read") is True
    assert await provider.authorize(user, "document:delete") is False