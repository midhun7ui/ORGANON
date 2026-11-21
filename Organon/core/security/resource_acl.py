"""
ORGANON - Resource Access Control

Defines resource identity and access control models.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from core.security.base import UserContext
from collections.abc import Mapping
from types import MappingProxyType


@dataclass(frozen=True)
class ResourceContext:
    """Represents a resource involved in an authorization request."""

    resource_id: str
    resource_type: str
    tenant_id: str
    owner_id: str | None = None

    def __post_init__(self):
        """Validate resource identity and ownership information."""

        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError("Resource ID must be a non-empty string.")

        if not isinstance(self.resource_type, str) or not self.resource_type.strip():
            raise ValueError("Resource type must be a non-empty string.")

        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("Tenant ID must be a non-empty string.")

        if self.owner_id is not None:
            if not isinstance(self.owner_id, str) or not self.owner_id.strip():
                raise ValueError("Owner ID must be a non-empty string.")


class ResourceAuthorizationProvider(ABC):
    """Interface for resource-level access control providers."""

    @abstractmethod
    async def authorize(
        self,
        user: UserContext,
        resource: ResourceContext,
        action: str,
    ) -> bool:
        """Determine whether a user can perform an action on a resource."""
        ...


@dataclass(frozen=True)
class ResourceACLRule:
    """Defines access permissions for a resource principal."""

    principal_type: str
    principal_id: str
    actions: frozenset[str]

    def __post_init__(self):
        """Validate the resource ACL rule."""

        if not isinstance(self.principal_type, str):
            raise ValueError("Principal type must be a string.")

        if self.principal_type not in {"user", "role"}:
            raise ValueError("Principal type must be 'user' or 'role'.")

        if not isinstance(self.principal_id, str) or not self.principal_id.strip():
            raise ValueError("Principal ID must be a non-empty string.")

        if not isinstance(self.actions, (set, frozenset)):
            raise ValueError("Actions must be a set or frozenset.")

        if not self.actions:
            raise ValueError("An ACL rule must contain at least one action.")

        if not all(
            isinstance(action, str) and action.strip() for action in self.actions
        ):
            raise ValueError("Every action must be a non-empty string.")

        object.__setattr__(self, "actions", frozenset(self.actions))


class InMemoryResourceACLProvider(ResourceAuthorizationProvider):
    """In-memory implementation of resource-level access control."""

    def __init__(
        self,
        rules: Mapping[tuple[str, str, str], tuple[ResourceACLRule, ...]],
    ):
        """Initialize the provider with validated ACL rules."""

        if not isinstance(rules, Mapping):
            raise TypeError("ACL rules must be a mapping.")

        validated_rules = {}

        for resource_key, resource_rules in rules.items():

            if not isinstance(resource_key, tuple) or len(resource_key) != 3:
                raise ValueError(
                    "Resource keys must contain tenant_id, resource_type, and resource_id."
                )

            if not all(
                isinstance(value, str) and value.strip() for value in resource_key
            ):
                raise ValueError("Resource key values must be non-empty strings.")

            if not isinstance(resource_rules, (tuple, list, set, frozenset)):
                raise TypeError("Resource rules must be a collection.")

            if not all(isinstance(rule, ResourceACLRule) for rule in resource_rules):
                raise TypeError("Every resource rule must be a ResourceACLRule.")

            validated_rules[resource_key] = tuple(resource_rules)

        self.rules = MappingProxyType(validated_rules)

    async def authorize(
        self,
        user: UserContext,
        resource: ResourceContext,
        action: str,
    ) -> bool:
        """Determine whether a user can access a specific resource."""

        if not isinstance(user, UserContext):
            return False

        if not isinstance(resource, ResourceContext):
            return False

        if not isinstance(action, str) or not action.strip():
            return False

        # Prevent access across tenant boundaries.
        if user.tenant_id != resource.tenant_id:
            return False

        # Identify the requested resource.
        resource_key = (
            resource.tenant_id,
            resource.resource_type,
            resource.resource_id,
        )

        resource_rules = self.rules.get(resource_key, ())

        # Check whether any rule grants the requested action.
        for rule in resource_rules:

            if rule.principal_type == "user":
                principal_matches = rule.principal_id == user.user_id

            elif rule.principal_type == "role":
                principal_matches = rule.principal_id in user.roles

            else:
                principal_matches = False

            if principal_matches and action in rule.actions:
                return True

        # Default-deny: no matching rule means no access.
        return False
