# A centralized exception system so every ORGANON module reports errors consistently

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
    """Raised when a security operation fails."""
