
import pytest

from core.security.rbac import Permission, Role
from core.security.base import UserContext
from core.security.rbac import (
    Permission,
    Role,
    RoleBasedAuthorization,
)

import pytest


def test_permission_creation():
    permission = Permission(name="document:read")

    assert permission.name == "document:read"
    assert permission.description == ""


def test_permission_rejects_empty_name():
    with pytest.raises(ValueError):
        Permission(name="")


def test_role_creation():
    read = Permission(name="document:read")
    write = Permission(name="document:write")

    role = Role(
        name="editor",
        permissions={read, write},
    )

    assert role.name == "editor"
    assert len(role.permissions) == 2
    assert isinstance(role.permissions, frozenset)


def test_role_rejects_empty_name():
    with pytest.raises(ValueError):
        Role(name="")


def test_role_rejects_invalid_permissions():
    with pytest.raises(ValueError):
        Role(name="editor", permissions={"document:read"})

@pytest.mark.asyncio
async def test_rbac_authorizes_user_with_matching_role():
    editor = Role(
        name="editor",
        permissions=frozenset({
            Permission("document:read"),
            Permission("document:update"),
        }),
    )

    authorization = RoleBasedAuthorization(
        roles={"editor": editor}
    )

    user = UserContext(
        user_id="user-1",
        tenant_id="tenant-1",
        roles={"editor"},
    )

    assert await authorization.authorize(user, "document:read")


@pytest.mark.asyncio
async def test_rbac_authorizes_user_with_multiple_roles():
    editor = Role(
        name="editor",
        permissions=frozenset({Permission("document:update")}),
    )

    viewer = Role(
        name="viewer",
        permissions=frozenset({Permission("document:read")}),
    )

    authorization = RoleBasedAuthorization(
        roles={"editor": editor, "viewer": viewer}
    )

    user = UserContext(
        user_id="user-1",
        tenant_id="tenant-1",
        roles={"editor", "viewer"},
    )

    assert await authorization.authorize(user, "document:read")
    assert await authorization.authorize(user, "document:update")


@pytest.mark.asyncio
async def test_rbac_denies_user_without_required_permission():
    viewer = Role(
        name="viewer",
        permissions=frozenset({Permission("document:read")}),
    )

    authorization = RoleBasedAuthorization(
        roles={"viewer": viewer}
    )

    user = UserContext(
        user_id="user-1",
        tenant_id="tenant-1",
        roles={"viewer"},
    )

    assert not await authorization.authorize(user, "document:delete")


@pytest.mark.asyncio
async def test_rbac_denies_user_without_roles():
    authorization = RoleBasedAuthorization(roles={})

    user = UserContext(
        user_id="user-1",
        tenant_id="tenant-1",
    )

    assert not await authorization.authorize(user, "document:read")


@pytest.mark.asyncio
async def test_rbac_denies_unknown_role():
    authorization = RoleBasedAuthorization(roles={})

    user = UserContext(
        user_id="user-1",
        tenant_id="tenant-1",
        roles={"unknown"},
    )

    assert not await authorization.authorize(user, "document:read")

def test_rbac_rejects_invalid_role_registry():
    with pytest.raises(ValueError):
        RoleBasedAuthorization(roles=[])


def test_rbac_rejects_invalid_role_definition():
    with pytest.raises(ValueError):
        RoleBasedAuthorization(roles={"editor": "invalid"})


def test_rbac_rejects_mismatched_role_name():
    editor = Role(name="editor")

    with pytest.raises(ValueError):
        RoleBasedAuthorization(roles={"admin": editor})


def test_rbac_registry_is_immutable():
    editor = Role(name="editor")
    authorization = RoleBasedAuthorization(roles={"editor": editor})

    with pytest.raises(TypeError):
        authorization.roles["admin"] = Role(name="admin")


def test_rbac_registry_is_independent_of_original_mapping():
    editor = Role(name="editor")
    roles = {"editor": editor}

    authorization = RoleBasedAuthorization(roles=roles)
    roles.clear()

    assert "editor" in authorization.roles