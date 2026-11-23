
"""
ORGANON - Logging Provider Registration

Registers built-in logging providers.
"""

from core.observability.logging.providers.standard import StandardLogger
from core.registry import Registry


def register_logging_providers(registry: Registry) -> None:
    """Register built-in logging providers."""

    registry.register(
        category="logging",
        name="standard",
        provider=StandardLogger,
    )
