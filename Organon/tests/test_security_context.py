
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
