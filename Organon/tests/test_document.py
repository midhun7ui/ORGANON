
from core.documents.base import Document


def test_document_creation():
    document = Document(content="Hello ORGANON")

    assert document.content == "Hello ORGANON"
    assert document.metadata == {}


def test_document_metadata():
    metadata = {"source": "example.pdf", "page": 1}

    document = Document(
        content="Document content",
        metadata=metadata,
    )

    assert document.metadata == metadata


def test_documents_have_independent_metadata():
    first = Document(content="First document")
    second = Document(content="Second document")

    first.metadata["source"] = "first.pdf"

    assert "source" not in second.metadata
