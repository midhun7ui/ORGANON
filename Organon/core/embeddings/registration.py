"""
ORGANON - Embedding Provider Registration

Registers embedding providers with the ORGANON registry.
"""

from typing import Any

from core.registry import Registry
from core.embeddings.base import EmbeddingProvider
from core.embeddings.providers.sentence_transformers import (
    SentenceTransformerProvider,
)

def register_embedding_providers(registry: Registry) -> None:
    """Register available embedding providers."""

    registry.register(
        category="embedding",
        name="sentence_transformers",
        provider=SentenceTransformerProvider,
    )


def create_embedding_provider(
    registry: Registry,
    provider: str,
    config: dict[str, Any],
) -> EmbeddingProvider:
    """Create an embedding provider from the registered provider factory."""

    # Retrieve the provider factory from the registry.
    provider_factory = registry.get("embedding", provider)

    # Create the provider instance using the supplied configuration.
    embedding_provider = provider_factory(**config)

    return embedding_provider
