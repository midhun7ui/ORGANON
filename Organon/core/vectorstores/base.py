"""
ORGANON - Vector Store Interface

Defines the contract that all vector store providers must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorSearchResult:
    """Represents a document returned by vector similarity search."""

    id: str
    document: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract interface for vector store providers."""

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Insert new records or update existing records.

        Existing records with matching IDs will be updated.
        """

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for the most relevant documents.

        Results should be ordered by relevance,
        with the highest score first.
        """

    @abstractmethod
    async def delete(
        self,
        ids: list[str],
    ) -> None:
        """Delete documents using their IDs."""
