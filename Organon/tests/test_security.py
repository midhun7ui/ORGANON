
import pytest

from core.security.base import (
    AuthenticationProvider,
    AuthorizationProvider,
    UserContext,
)


# Verify that abstract security interfaces cannot be instantiated.
def test_security_providers_are_abstract():
    with pytest.raises(TypeError):
        AuthenticationProvider()

    with pytest.raises(TypeError):
        AuthorizationProvider()


# Fake authentication implementation for testing.
class FakeAuthenticationProvider(AuthenticationProvider):

    async def authenticate(self, credentials):
        return UserContext(
            user_id="user-001",
            tenant_id="tenant-001",
            permissions={"document:read"},
        )


# Fake authorization implementation for testing.
class FakeAuthorizationProvider(AuthorizationProvider):

    async def authorize(self, user, action):
        return action in user.permissions


# Test authentication functionality.
@pytest.mark.asyncio
async def test_authentication_provider():
    provider = FakeAuthenticationProvider()

    user = await provider.authenticate({})

    assert isinstance(user, UserContext)
    assert user.user_id == "user-001"
    assert user.tenant_id == "tenant-001"
    assert "document:read" in user.permissions


# Test authorization functionality.
@pytest.mark.asyncio
async def test_authorization_provider():
    provider = FakeAuthorizationProvider()

    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-001",
        permissions={"document:read"},
    )

    assert await provider.authorize(user, "document:read") is True
    assert await provider.authorize(user, "document:delete") is False
