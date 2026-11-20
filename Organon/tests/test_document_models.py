
import pytest
from datetime import datetime

from core.documents.models import (
    DocumentRecord,
    DocumentVersion,
    VersionStatus,
)


def test_document_record_creation():
    document = DocumentRecord(
        id="policy-001",
        tenant_id="company-001",
    )

    assert document.id == "policy-001"
    assert document.tenant_id == "company-001"
    assert document.current_version_id is None


@pytest.mark.parametrize("invalid_id", ["", "   ", None, 123])
def test_document_record_rejects_invalid_id(invalid_id):
    with pytest.raises(ValueError):
        DocumentRecord(id=invalid_id, tenant_id="company-001")


def test_document_version_creation():
    version = DocumentVersion(
        id="version-001",
        document_id="policy-001",
        version_number=1,
    )

    assert version.version_number == 1
    assert version.status == VersionStatus.PROCESSING


@pytest.mark.parametrize("invalid_number", [0, -1, True, 1.5, "1"])
def test_document_version_rejects_invalid_number(invalid_number):
    with pytest.raises(ValueError):
        DocumentVersion(
            id="version-001",
            document_id="policy-001",
            version_number=invalid_number,
        )


def test_document_version_rejects_invalid_status():
    with pytest.raises(ValueError):
        DocumentVersion(
            id="version-001",
            document_id="policy-001",
            version_number=1,
            status="unknown",
        )


def test_document_record_accepts_datetime():
    now = datetime.now()

    document = DocumentRecord(
        id="policy-001",
        tenant_id="company-001",
        created_at=now,
        updated_at=now,
    )

    assert document.created_at == now
    assert document.updated_at == now
