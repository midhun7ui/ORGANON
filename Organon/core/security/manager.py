
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
from core.security.resource_acl import (
    ResourceAuthorizationProvider,
    ResourceContext,
)


class SecurityManager:
    """Coordinates authentication and authorization in ORGANON."""

    def __init__(
        self,
        authentication: AuthenticationProvider,
        authorization: AuthorizationProvider,
        resource_authorization: ResourceAuthorizationProvider | None = None,
    ) -> None:

        # Store the authentication implementation.
        self.authentication = authentication

        # Store the global authorization implementation.
        self.authorization = authorization

        # Store the optional resource-level authorization implementation.
        self.resource_authorization = resource_authorization

    async def authorize(
        self,
        credentials: dict[str, Any],
        action: str,
    ) -> SecurityContext:
        """Authenticate a user and check global authorization."""

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

    async def authorize_resource(
        self,
        credentials: dict[str, Any],
        action: str,
        resource: ResourceContext,
    ) -> SecurityContext:
        """Authenticate and authorize access to a single resource."""

        if self.resource_authorization is None:
            raise AuthorizationError(
                "Resource-level authorization is not configured."
            )

        # Authenticate the user.
        user = await self.authentication.authenticate(credentials)

        # Check global permission through RBAC.
        is_authorized = await self.authorization.authorize(user, action)

        if not is_authorized:
            raise AuthorizationError(
                f"User '{user.user_id}' is not allowed to perform '{action}'."
            )

        # Prevent access to resources belonging to another tenant.
        if resource.tenant_id != user.tenant_id:
            raise AuthorizationError(
                "Access to resources belonging to another tenant is denied."
            )

        # Check resource-level permission through ACL.
        has_resource_access = await self.resource_authorization.authorize(
            user,
            resource,
            action,
        )

        if not has_resource_access:
            raise AuthorizationError(
                f"User '{user.user_id}' is not allowed to perform "
                f"'{action}' on resource '{resource.resource_id}'."
            )

        return SecurityContext.from_user(user)

    async def authorize_resources(
        self,
        credentials: dict[str, Any],
        action: str,
        resource_ids: list[str],
        resource_type: str,
    ) -> SecurityContext:
        """Authenticate once and authorize access to multiple resources."""

        # Ensure resource-level authorization is configured.
        if self.resource_authorization is None:
            raise AuthorizationError(
                "Resource-level authorization is not configured."
            )

        # Validate the resource ID collection.
        if not isinstance(resource_ids, list) or not resource_ids:
            raise AuthorizationError(
                "At least one resource ID must be provided."
            )

        if not all(
            isinstance(resource_id, str) and resource_id.strip()
            for resource_id in resource_ids
        ):
            raise AuthorizationError(
                "Every resource ID must be a non-empty string."
            )

        if len(resource_ids) != len(set(resource_ids)):
            raise AuthorizationError(
                "Duplicate resource IDs are not allowed."
            )

        # Validate the resource type.
        if not isinstance(resource_type, str) or not resource_type.strip():
            raise AuthorizationError(
                "Resource type must be a non-empty string."
            )

        # Authenticate the requester only once.
        user = await self.authentication.authenticate(credentials)

        # Check global permission once through RBAC.
        is_authorized = await self.authorization.authorize(user, action)

        if not is_authorized:
            raise AuthorizationError(
                f"User '{user.user_id}' is not allowed to perform '{action}'."
            )

        # Construct resource contexts using the authenticated tenant ID.
        for resource_id in resource_ids:

            resource = ResourceContext(
                resource_id=resource_id,
                resource_type=resource_type,
                tenant_id=user.tenant_id,
            )

            # Check resource-level permission through ACL.
            has_resource_access = await self.resource_authorization.authorize(
                user,
                resource,
                action,
            )

            if not has_resource_access:
                raise AuthorizationError(
                    f"User '{user.user_id}' is not allowed to perform "
                    f"'{action}' on resource '{resource_id}'."
                )

        # Return context only after every resource is authorized.
        return SecurityContext.from_user(user)
