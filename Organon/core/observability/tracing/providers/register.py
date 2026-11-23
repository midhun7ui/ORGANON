"""
ORGANON - Tracing Provider Registration

Registers built-in tracing providers.
"""

from core.observability.tracing.providers.standard import StandardTracer
from core.registry import Registry


def register_tracing_providers(registry: Registry) -> None:
    """Register built-in tracing providers."""

    registry.register(
        category="tracing",
        name="standard",
        provider=StandardTracer,
    )