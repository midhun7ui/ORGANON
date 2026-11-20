from core.chunking.base import Chunker
from core.documents.base import Document


class FixedSizeChunker(Chunker):
    """Split documents into fixed-size character chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 0,
    ) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")

        if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
            raise ValueError("Overlap must be a non-negative integer.")

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Document]:
        if not isinstance(document, Document):
            raise TypeError("Expected a Document instance.")
        chunks = []
        step = self.chunk_size - self.overlap

        for start in range(0, len(document.content), step):
            content = document.content[start:start + self.chunk_size]

            chunks.append(
                Document(
                    content=content,
                    metadata={
                        **document.metadata,
                        "chunk_index": len(chunks),
                    },
                )
            )

        return chunks