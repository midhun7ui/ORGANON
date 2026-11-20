from abc import ABC, abstractmethod

from core.documents.models import DocumentRecord, DocumentVersion


class MetadataStore(ABC):
    """Abstract interface for ORGANON metadata providers."""

    @abstractmethod
    async def create_document(
        self,
        document: DocumentRecord,
    ) -> DocumentRecord:
        """Create a new document record."""
        pass

    @abstractmethod
    async def get_document(
        self,
        document_id: str,
        *,
        tenant_id: str,
    ) -> DocumentRecord | None:
        """Retrieve a document belonging to a specific tenant."""
        pass

    @abstractmethod
    async def create_version(
        self,
        document_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """Create the next version of a document atomically."""
        pass

    @abstractmethod
    async def get_version(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion | None:
        """Retrieve a specific version belonging to a tenant's document."""
        pass

    @abstractmethod
    async def list_versions(
        self,
        document_id: str,
        *,
        tenant_id: str,
        limit: int = 50,
        before_version_number: int | None = None,
    ) -> list[DocumentVersion]:
        """List document versions using cursor-based pagination."""
        pass

    @abstractmethod
    async def activate_version(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """Atomically activate a ready document version."""
        pass


    @abstractmethod
    async def mark_version_ready(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """Mark a successfully ingested version as ready for activation."""
        pass


    @abstractmethod
    async def mark_version_failed(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """Mark a failed ingestion version as failed."""
        pass
