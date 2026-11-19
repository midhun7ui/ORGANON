
"""
ORGANON - Security Context

Provides immutable identity and tenant information
to ORGANON's internal orchestration layer.
"""

from dataclasses import dataclass

from core.security.base import UserContext


@dataclass(frozen=True)
class SecurityContext:
    """Immutable security context used by ORGANON Core."""

    user_id: str
    tenant_id: str
    permissions: frozenset[str]

    @classmethod
    def from_user(cls, user: UserContext) -> "SecurityContext":
        """Create an immutable context from an authenticated user."""

        return cls(
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            permissions=frozenset(user.permissions),
        )
