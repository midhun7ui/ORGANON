"""
ORGANON - Document Model

Defines the standard representation of a document
used throughout the ingestion pipeline.
"""

from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod
from pathlib import Path


@dataclass
class Document:
    """Represents a document inside ORGANON."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(ABC):
    """Base interface for all document loaders."""

    @abstractmethod
    async def load(self, source: str | Path) -> list[Document]:
        """Load documents from a source."""
        pass
