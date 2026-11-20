from pathlib import Path

from core.documents.base import Document, DocumentLoader
from core.exceptions import IngestionError


class TextLoader(DocumentLoader):
    """Load plain text files into ORGANON documents."""

    async def load(self, source: str | Path) -> list[Document]:
        path = Path(source)

        if not path.exists():
            raise IngestionError(f"File not found: {path}")

        if not path.is_file():
            raise IngestionError(f"Source is not a file: {path}")

        if path.suffix.lower() != ".txt":
            raise IngestionError(f"Unsupported file format: {path.suffix}")

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestionError(f"Failed to read text file: {path}") from exc

        if not content.strip():
            raise IngestionError(f"Text file is empty: {path}")

        document = Document(
            content=content,
            metadata={
                "source": str(path),
                "file_type": "txt",
            },
        )

        return [document]
