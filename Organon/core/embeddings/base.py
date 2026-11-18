"""
ORGANON - Embedding Provider Interface

Defines the contract that all embedding providers must follow.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        pass

    @abstractmethod
    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a single query."""
        pass
