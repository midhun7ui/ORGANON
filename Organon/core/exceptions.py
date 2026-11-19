
"""
ORGANON - Core Exception Handling

Centralized exception hierarchy for the ORGANON framework.
"""


class OrganonError(Exception):
    """Base exception for all ORGANON-specific errors."""


class RegistryError(OrganonError):
    """Raised when provider registration lookup fails."""


class ConfigurationError(OrganonError):
    """Raised when framework configuration is invalid."""


class ProviderError(OrganonError):
    """Raised when a provider encounters an error."""


class IngestionError(OrganonError):
    """Raised when document ingestion fails."""


class RetrievalError(OrganonError):
    """Raised when document retrieval fails."""


class GenerationError(OrganonError):
    """Raised when LLM generation fails."""


class SecurityError(OrganonError):
    """Base exception for all ORGANON security-related errors."""


class AuthenticationError(SecurityError):
    """Raised when a requester cannot be authenticated."""


class AuthorizationError(SecurityError):
    """Raised when a user is not authorized to perform an action."""


class ValidationError(OrganonError):
    """Raised when input data fails validation."""
