"""
ORGANON - Role-Based Access Control (RBAC)

Defines roles and permissions used by ORGANON's
authorization system.
"""

from types import MappingProxyType
from typing import Mapping
from dataclasses import dataclass, field
from core.security.base import AuthorizationProvider, UserContext


@dataclass(frozen=True)
class Permission:
    """Represents a single permission in ORGANON."""

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Permission name must be a non-empty string.")

        if not isinstance(self.description, str):
            raise ValueError("Permission description must be a string.")


@dataclass(frozen=True)
class Role:
    """Represents a role containing a collection of permissions."""

    name: str
    permissions: frozenset[Permission] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Role name must be a non-empty string.")

        if not isinstance(self.permissions, (set, frozenset)):
            raise ValueError("Role permissions must be a set.")

        if not all(
            isinstance(permission, Permission) for permission in self.permissions
        ):
            raise ValueError("Role permissions must contain Permission objects.")

        object.__setattr__(
            self,
            "permissions",
            frozenset(self.permissions),
        )


class RoleBasedAuthorization(AuthorizationProvider):
    """Authorization implementation based on user roles."""

    def __init__(self, roles: Mapping[str, Role]) -> None:
        """Initialize the authorization provider with validated role definitions."""

        if not isinstance(roles, Mapping):
            raise ValueError("Roles must be provided as a mapping.")

        validated_roles = {}

        for name, role in roles.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Role registry keys must be non-empty strings.")

            if not isinstance(role, Role):
                raise ValueError(f"Invalid role definition for '{name}'.")

            if name != role.name:
                raise ValueError(
                    f"Role registry key '{name}' does not match "
                    f"role name '{role.name}'."
                )

            validated_roles[name] = role

    # Keep a read-only copy of the role registry.
        self.roles = MappingProxyType(validated_roles)

    async def authorize(self, user: UserContext, action: str) -> bool:
        """Check whether any assigned role grants the requested action."""

        for role_name in user.roles:
            role = self.roles.get(role_name)

            # Unknown roles must not grant access.
            if role is None:
                continue

            # Check whether the role contains the requested permission.
            if any(permission.name == action for permission in role.permissions):
                return True

        return False
