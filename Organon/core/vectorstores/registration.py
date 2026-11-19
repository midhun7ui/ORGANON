"""
ORGANON - Vector Store Registration

Registers vector store providers with the ORGANON registry.
"""

from core.registry import Registry
from core.vectorstores.base import VectorStore
from core.vectorstores.providers.chroma import ChromaVectorStore
from typing import Any


def register_vector_stores(registry: Registry) -> None:
    """Register available vector store providers."""

    registry.register(
        category="vector_store",
        name="chroma",
        provider=ChromaVectorStore,
    )


def create_vector_store(
    registry: Registry,
    provider: str,
    config: dict[str, Any],
) -> VectorStore:
    """Create a vector store instance from the registered provider."""

    # Retrieve the provider factory from the registry.
    provider_factory = registry.get("vector_store", provider)

    # Create and return the provider instance.
    vector_store = provider_factory(**config)

    return vector_store
