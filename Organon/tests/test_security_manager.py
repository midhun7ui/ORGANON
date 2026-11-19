import pytest
from core.exceptions import AuthorizationError, AuthenticationError

from core.security.base import (
    AuthenticationProvider,
    AuthorizationProvider,
    UserContext,
)
from core.security.manager import SecurityManager
from core.engine.security_context import SecurityContext


# Fake authentication provider for testing.
class FakeAuthenticationProvider(AuthenticationProvider):

    async def authenticate(self, credentials):
        return UserContext(
            user_id="user-001",
            tenant_id="tenant-001",
            permissions={"document:read"},
        )


# Fake authorization provider for testing.
class FakeAuthorizationProvider(AuthorizationProvider):

    async def authorize(self, user, action):
        return action in user.permissions


# Test successful authentication and authorization.
@pytest.mark.asyncio
async def test_security_manager_allows_authorized_user():

    manager = SecurityManager(
        authentication=FakeAuthenticationProvider(),
        authorization=FakeAuthorizationProvider(),
    )
    context = await manager.authorize(
        credentials={"token": "test-token"},
        action="document:read",
    )

    assert isinstance(context, SecurityContext)
    assert context.user_id == "user-001"
    assert context.tenant_id == "tenant-001"
    assert context.permissions == {"document:read"}


# Test that unauthorized requests are rejected.
@pytest.mark.asyncio
async def test_security_manager_rejects_unauthorized_user():

    manager = SecurityManager(
        authentication=FakeAuthenticationProvider(),
        authorization=FakeAuthorizationProvider(),
    )

    with pytest.raises(AuthorizationError):
        await manager.authorize(
            credentials={"token": "test-token"},
            action="document:delete",
        )


class FailingAuthenticationProvider(AuthenticationProvider):

    async def authenticate(self, credentials):
        raise AuthenticationError("Invalid credentials.")


# Test that invalid credentials are rejected.
@pytest.mark.asyncio
async def test_security_manager_rejects_invalid_credentials():

    manager = SecurityManager(
        authentication=FailingAuthenticationProvider(),
        authorization=FakeAuthorizationProvider(),
    )

    with pytest.raises(AuthenticationError):
        await manager.authorize(
            credentials={"token": "invalid-token"},
            action="document:read",
        )
