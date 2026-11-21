import pytest
from core.exceptions import AuthorizationError, AuthenticationError

from core.security.base import (
    AuthenticationProvider,
    AuthorizationProvider,
    UserContext,
)
from core.security.manager import SecurityManager
from core.engine.security_context import SecurityContext
from core.security.rbac import (
    Permission,
    Role,
    RoleBasedAuthorization,
)

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
@pytest.mark.asyncio
async def test_security_manager_works_with_rbac():

    user = UserContext(
        user_id="user-002",
        tenant_id="tenant-002",
        roles={"admin"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        async def authenticate(self, credentials):
            return user

    authorization = RoleBasedAuthorization(
        roles={
            "admin": Role(
                name="admin",
                permissions=frozenset({
                    Permission("document:read"),
                    Permission("document:write"),
                }),
            )
        }
    )

    manager = SecurityManager(
        authentication=RBACAuthenticationProvider(),
        authorization=authorization,
    )

    context = await manager.authorize(
        credentials={"token": "valid-token"},
        action="document:write",
    )

    assert context.user_id == "user-002"
    assert context.tenant_id == "tenant-002"
    assert context.roles == frozenset({"admin"})


@pytest.mark.asyncio
async def test_security_manager_rejects_rbac_user_without_permission():

    user = UserContext(
        user_id="user-003",
        tenant_id="tenant-002",
        roles={"viewer"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        async def authenticate(self, credentials):
            return user

    authorization = RoleBasedAuthorization(
        roles={
            "viewer": Role(
                name="viewer",
                permissions=frozenset({
                    Permission("document:read"),
                }),
            )
        }
    )

    manager = SecurityManager(
        authentication=RBACAuthenticationProvider(),
        authorization=authorization,
    )

    with pytest.raises(AuthorizationError):
        await manager.authorize(
            credentials={"token": "valid-token"},
            action="document:write",
        )