
"""
ORGANON - Security Manager

Coordinates authentication and authorization.
"""

from typing import Any

from core.exceptions import AuthorizationError
from core.security.base import (
    AuthenticationProvider,
    AuthorizationProvider,
)
from core.engine.security_context import SecurityContext


class SecurityManager:
    """Coordinates authentication and authorization in ORGANON."""

    def __init__(
        self,
        authentication: AuthenticationProvider,
        authorization: AuthorizationProvider,
    ) -> None:

        # Store the authentication implementation.
        self.authentication = authentication

        # Store the authorization implementation.
        self.authorization = authorization

    async def authorize(
        self,
        credentials: dict[str, Any],
        action: str,
    ) -> SecurityContext:

        # Authenticate the requester.
        user = await self.authentication.authenticate(credentials)

        # Check whether the user can perform the requested action.
        is_authorized = await self.authorization.authorize(user, action)

        # Reject the request if authorization fails.
        if not is_authorized:
            raise AuthorizationError(
                f"User '{user.user_id}' is not allowed to perform '{action}'."
            )

        # Create a trusted security context after successful authorization.
        return SecurityContext.from_user(user)
