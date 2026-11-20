from abc import ABC, abstractmethod

from core.documents.base import Document


class Chunker(ABC):
    """Base interface for all document chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """Split a document into smaller documents."""
        pass