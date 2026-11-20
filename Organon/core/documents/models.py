
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class VersionStatus(str, Enum):
    """Lifecycle states of a document version."""

    PROCESSING = "processing"
    READY = "ready"
    ACTIVE = "active"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class DocumentRecord:
    """Represents the identity of a document within a tenant."""

    id: str
    tenant_id: str
    current_version_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("Tenant ID must be a non-empty string.")

        if self.current_version_id is not None:
            if (
                not isinstance(self.current_version_id, str)
                or not self.current_version_id.strip()
            ):
                raise ValueError("Current version ID must be a non-empty string.")

        if self.created_at is not None and not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime instance.")

        if self.updated_at is not None and not isinstance(self.updated_at, datetime):
            raise TypeError("updated_at must be a datetime instance.")


@dataclass
class DocumentVersion:
    """Represents a specific version of a document."""

    id: str
    document_id: str
    version_number: int
    status: VersionStatus = VersionStatus.PROCESSING
    created_at: datetime | None = None
    activated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Version ID must be a non-empty string.")

        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        if (
            isinstance(self.version_number, bool)
            or not isinstance(self.version_number, int)
            or self.version_number <= 0
        ):
            raise ValueError("Version number must be a positive integer.")

        if not isinstance(self.status, VersionStatus):
            raise ValueError("Invalid document version status.")

        if self.created_at is not None and not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime instance.")

        if self.activated_at is not None and not isinstance(self.activated_at, datetime):
            raise TypeError("activated_at must be a datetime instance.")
