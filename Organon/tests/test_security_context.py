
import pytest

from core.engine.security_context import SecurityContext
from core.security.base import UserContext


def test_security_context_properties():
    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-001",
        permissions={"document:read", "document:write"},
    )

    context = SecurityContext.from_user(user)

    assert context.user_id == "user-001"
    assert context.tenant_id == "tenant-001"
    assert context.permissions == {
        "document:read",
        "document:write",
    }


def test_security_context_permissions_are_immutable():
    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-001",
        permissions={"document:read"},
    )

    context = SecurityContext.from_user(user)

    with pytest.raises(AttributeError):
        context.permissions.add("document:delete")


def test_security_context_is_frozen():
    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-001",
        permissions={"document:read"},
    )

    context = SecurityContext.from_user(user)

    with pytest.raises(AttributeError):
        context.tenant_id = "tenant-002"


def test_security_context_is_independent_of_original_user():
    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions={"document:read"},
    )

    context = SecurityContext.from_user(user)

    # Modify the original user.
    user.tenant_id = "tenant-B"
    user.permissions.add("document:delete")

    # The security context must remain unchanged.
    assert context.tenant_id == "tenant-A"
    assert context.permissions == {"document:read"}

import pytest

from core.engine.security_context import SecurityContext
from core.security.base import UserContext


def test_security_context_from_user_copies_roles():
    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions={"document:read"},
        roles={"admin", "editor"},
    )

    context = SecurityContext.from_user(user)

    assert context.user_id == "user-001"
    assert context.tenant_id == "tenant-A"
    assert context.permissions == frozenset({"document:read"})
    assert context.roles == frozenset({"admin", "editor"})


def test_security_context_roles_are_immutable():
    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-A",
        roles={"admin"},
    )

    context = SecurityContext.from_user(user)

    with pytest.raises(AttributeError):
        context.roles.add("editor")


def test_security_context_is_independent_of_user_roles():
    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-A",
        roles={"admin"},
    )

    context = SecurityContext.from_user(user)

    user.roles.add("editor")

    assert context.roles == frozenset({"admin"})


def test_security_context_supports_legacy_initialization():
    context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    assert context.roles == frozenset()