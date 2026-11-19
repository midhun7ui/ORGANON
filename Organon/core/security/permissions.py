"""
ORGANON - Permission-Based Authorization

Provides authorization using permissions assigned to a user.
"""

from core.security.base import AuthorizationProvider, UserContext


class PermissionBasedAuthorization(AuthorizationProvider):
    """Authorization implementation based on user permissions."""

    async def authorize(self, user: UserContext, action: str) -> bool:
        """Check whether the user has permission to perform an action."""

        return action in user.permissions