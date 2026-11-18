"""
ORGANON - Vector Store Interface

Defines the contract that all vector store providers must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorSearchResult:
    """Represents a document returned by vector similarity search."""

    id: str
    document: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Abstract interface for vector store providers."""

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store document vectors, content, and metadata."""
        pass

    @abstractmethod
    async def search(
        self,
        query_verctor: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for documents similar to the query vector."""
        pass

    @abstractmethod
    async def delete(
        self,
        ids: list[str],
    ) -> None:
        """Delete documents using their IDs."""
        pass
