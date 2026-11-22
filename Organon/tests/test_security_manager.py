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
from core.security.resource_acl import (
    InMemoryResourceACLProvider,
    ResourceACLRule,
    ResourceContext,
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
                permissions=frozenset(
                    {
                        Permission("document:read"),
                        Permission("document:write"),
                    }
                ),
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
                permissions=frozenset(
                    {
                        Permission("document:read"),
                    }
                ),
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


@pytest.mark.asyncio
async def test_security_manager_allows_resource_access_with_rbac_and_acl():

    user = UserContext(
        user_id="user-004",
        tenant_id="tenant-004",
        roles={"editor"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        async def authenticate(self, credentials):
            return user

    # Global permission: editor can read documents.
    authorization = RoleBasedAuthorization(
        roles={
            "editor": Role(
                name="editor",
                permissions=frozenset(
                    {
                        Permission("document:read"),
                    }
                ),
            )
        }
    )

    # Resource permission: this user can read this specific document.
    resource_authorization = InMemoryResourceACLProvider(
        rules={
            ("tenant-004", "document", "doc-004"): (
                ResourceACLRule(
                    principal_type="user",
                    principal_id="user-004",
                    actions=frozenset({"document:read"}),
                ),
            )
        }
    )

    manager = SecurityManager(
        authentication=RBACAuthenticationProvider(),
        authorization=authorization,
        resource_authorization=resource_authorization,
    )

    resource = ResourceContext(
        resource_id="doc-004",
        resource_type="document",
        tenant_id="tenant-004",
    )

    context = await manager.authorize_resource(
        credentials={"token": "valid-token"},
        action="document:read",
        resource=resource,
    )

    assert isinstance(context, SecurityContext)
    assert context.user_id == "user-004"
    assert context.tenant_id == "tenant-004"
    assert context.roles == frozenset({"editor"})


@pytest.mark.asyncio
async def test_security_manager_rejects_resource_access_when_rbac_denies():

    user = UserContext(
        user_id="user-005",
        tenant_id="tenant-005",
        roles={"viewer"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        async def authenticate(self, credentials):
            return user

    authorization = RoleBasedAuthorization(
        roles={
            "viewer": Role(
                name="viewer",
                permissions=frozenset(
                    {
                        Permission("document:read"),
                    }
                ),
            )
        }
    )

    resource_authorization = InMemoryResourceACLProvider(rules={})

    manager = SecurityManager(
        authentication=RBACAuthenticationProvider(),
        authorization=authorization,
        resource_authorization=resource_authorization,
    )

    resource = ResourceContext(
        resource_id="doc-005",
        resource_type="document",
        tenant_id="tenant-005",
    )

    with pytest.raises(AuthorizationError):
        await manager.authorize_resource(
            credentials={"token": "valid-token"},
            action="document:delete",
            resource=resource,
        )


@pytest.mark.asyncio
async def test_security_manager_rejects_resource_access_when_acl_denies():

    user = UserContext(
        user_id="user-006",
        tenant_id="tenant-006",
        roles={"editor"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        async def authenticate(self, credentials):
            return user

    authorization = RoleBasedAuthorization(
        roles={
            "editor": Role(
                name="editor",
                permissions=frozenset(
                    {
                        Permission("document:read"),
                    }
                ),
            )
        }
    )

    # ACL has no grant for this resource.
    resource_authorization = InMemoryResourceACLProvider(rules={})

    manager = SecurityManager(
        authentication=RBACAuthenticationProvider(),
        authorization=authorization,
        resource_authorization=resource_authorization,
    )

    resource = ResourceContext(
        resource_id="doc-006",
        resource_type="document",
        tenant_id="tenant-006",
    )

    with pytest.raises(AuthorizationError):
        await manager.authorize_resource(
            credentials={"token": "valid-token"},
            action="document:read",
            resource=resource,
        )


@pytest.mark.asyncio
async def test_security_manager_rejects_resource_access_without_acl_provider():

    user = UserContext(
        user_id="user-007",
        tenant_id="tenant-007",
        roles={"editor"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        async def authenticate(self, credentials):
            return user

    authorization = RoleBasedAuthorization(
        roles={
            "editor": Role(
                name="editor",
                permissions=frozenset(
                    {
                        Permission("document:read"),
                    }
                ),
            )
        }
    )

    # No resource authorization provider configured.
    manager = SecurityManager(
        authentication=RBACAuthenticationProvider(),
        authorization=authorization,
    )

    resource = ResourceContext(
        resource_id="doc-007",
        resource_type="document",
        tenant_id="tenant-007",
    )

    with pytest.raises(AuthorizationError):
        await manager.authorize_resource(
            credentials={"token": "valid-token"},
            action="document:read",
            resource=resource,
        )


# Test batch authorization.
@pytest.mark.asyncio
async def test_security_manager_authorizes_multiple_resources():

    user = UserContext(
        user_id="user-008",
        tenant_id="tenant-008",
        roles={"editor"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        def __init__(self):
            self.authentication_count = 0

        async def authenticate(self, credentials):
            self.authentication_count += 1
            return user

    authentication = RBACAuthenticationProvider()

    authorization = RoleBasedAuthorization(
        roles={
            "editor": Role(
                name="editor",
                permissions=frozenset(
                    {
                        Permission("document:write"),
                    }
                ),
            )
        }
    )

    resource_authorization = InMemoryResourceACLProvider(
        rules={
            ("tenant-008", "document", "doc-008"): (
                ResourceACLRule(
                    principal_type="role",
                    principal_id="editor",
                    actions=frozenset({"document:write"}),
                ),
            ),
            ("tenant-008", "document", "doc-009"): (
                ResourceACLRule(
                    principal_type="role",
                    principal_id="editor",
                    actions=frozenset({"document:write"}),
                ),
            ),
        }
    )

    manager = SecurityManager(
        authentication=authentication,
        authorization=authorization,
        resource_authorization=resource_authorization,
    )

    context = await manager.authorize_resources(
        credentials={"token": "valid-token"},
        action="document:write",
        resource_ids=["doc-008", "doc-009"],
        resource_type="document",
    )

    assert context.user_id == "user-008"
    assert context.tenant_id == "tenant-008"
    assert authentication.authentication_count == 1


# Test that the entire batch is rejected if one resource is denied.
@pytest.mark.asyncio
async def test_security_manager_rejects_batch_when_one_resource_is_denied():

    user = UserContext(
        user_id="user-009",
        tenant_id="tenant-009",
        roles={"editor"},
    )

    class RBACAuthenticationProvider(AuthenticationProvider):

        async def authenticate(self, credentials):
            return user

    authorization = RoleBasedAuthorization(
        roles={
            "editor": Role(
                name="editor",
                permissions=frozenset(
                    {
                        Permission("document:write"),
                    }
                ),
            )
        }
    )

    resource_authorization = InMemoryResourceACLProvider(
        rules={
            ("tenant-009", "document", "doc-010"): (
                ResourceACLRule(
                    principal_type="role",
                    principal_id="editor",
                    actions=frozenset({"document:write"}),
                ),
            ),
        }
    )

    manager = SecurityManager(
        authentication=RBACAuthenticationProvider(),
        authorization=authorization,
        resource_authorization=resource_authorization,
    )

    with pytest.raises(AuthorizationError):
        await manager.authorize_resources(
            credentials={"token": "valid-token"},
            action="document:write",
            resource_ids=["doc-010", "doc-011"],
            resource_type="document",
        )
