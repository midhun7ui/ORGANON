import pytest

from core.security.base import UserContext
from core.security.permissions import PermissionBasedAuthorization


@pytest.mark.asyncio
async def test_authorize_allowed_action():
    provider = PermissionBasedAuthorization()

    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-001",
        permissions={"document:read"},
    )

    result = await provider.authorize(user, "document:read")

    assert result is True


@pytest.mark.asyncio
async def test_authorize_denied_action():
    provider = PermissionBasedAuthorization()

    user = UserContext(
        user_id="user-001",
        tenant_id="tenant-001",
        permissions={"document:read"},
    )

    result = await provider.authorize(user, "document:delete")

    assert result is False