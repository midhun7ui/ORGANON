# It should define the contract that all security implementations must follow,
# without deciding how authentication or authorization actually works.


# ORGANON - Security Base


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserContext:
    """Represents the identity and security context of an authenticated user."""

    user_id: str
    tenant_id: str
    permissions: set[str] = field(default_factory=set)


class AuthenticationProvider(ABC):
    """Base interface for all ORGANON authentication providers."""

    @abstractmethod
    async def authenticate(self, credentials: dict[str, Any]) -> UserContext:
        """Authenticate a requester and return their security context."""


class AuthorizationProvider(ABC):
    """Base interface for all ORGANON authorization providers."""

    @abstractmethod
    async def authorize(
        self,
        user: UserContext,
        action: str,
    ) -> bool:
        """Check whether a user is allowed to perform an action."""