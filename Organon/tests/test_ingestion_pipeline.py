
from unittest.mock import AsyncMock, Mock

import pytest

from core.exceptions import IngestionError, DocumentAlreadyExistsError
from core.documents.base import Document
from core.documents.models import DocumentRecord, DocumentVersion
from core.ingestion.pipeline import IngestionPipeline


@pytest.mark.asyncio
async def test_load_documents():
    loader = AsyncMock()
    loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={
                "source": "organon.txt",
                "file_type": "txt",
            },
        )
    ]

    pipeline = IngestionPipeline(
        document_loader=loader,
        chunker=None,
        embedding_provider=None,
        vector_store=None,
        metadata_store=AsyncMock(),
    )

    documents = await pipeline._load_documents(
        source="organon.txt",
        document_id="organon-001",
    )

    assert len(documents) == 1
    assert documents[0].content == "ORGANON is a modular RAG framework."
    assert documents[0].metadata == {
        "source": "organon.txt",
        "file_type": "txt",
        "document_id": "organon-001",
    }

    loader.load.assert_awaited_once_with("organon.txt")


def test_chunk_documents():
    chunker = Mock()
    chunker.chunk.return_value = [
        Document(
            content="First chunk",
            metadata={"document_id": "organon-001"},
        ),
        Document(
            content="Second chunk",
            metadata={"document_id": "organon-001"},
        ),
    ]

    pipeline = IngestionPipeline(
        document_loader=None,
        chunker=chunker,
        embedding_provider=None,
        vector_store=None,
        metadata_store=AsyncMock(),
    )

    document = Document(
        content="Original document content",
        metadata={"document_id": "organon-001"},
    )

    chunks = pipeline._chunk_documents([document])

    assert len(chunks) == 2
    assert chunks[0].content == "First chunk"
    assert chunks[1].content == "Second chunk"

    chunker.chunk.assert_called_once_with(document)


@pytest.mark.asyncio
async def test_embed_chunks():
    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    pipeline = IngestionPipeline(
        document_loader=None,
        chunker=None,
        embedding_provider=embedding_provider,
        vector_store=None,
        metadata_store=AsyncMock(),
    )

    chunks = [
        Document(content="First chunk"),
        Document(content="Second chunk"),
    ]

    embeddings = await pipeline._embed_chunks(chunks)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    embedding_provider.embed_documents.assert_awaited_once_with(
        ["First chunk", "Second chunk"]
    )


@pytest.mark.asyncio
async def test_store_chunks():
    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=None,
        chunker=None,
        embedding_provider=None,
        vector_store=vector_store,
        metadata_store=AsyncMock(),
    )

    chunks = [
        Document(
            content="First chunk",
            metadata={
                "document_id": "organon-001",
                "chunk_index": 99,
            },
        ),
        Document(
            content="Second chunk",
            metadata={
                "document_id": "organon-001",
                "chunk_index": 99,
            },
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    await pipeline._store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        tenant_id="tenant-001",
        version_id="version-001",
    )

    vector_store.upsert.assert_awaited_once()

    stored_data = vector_store.upsert.await_args.kwargs

    assert stored_data["ids"] == [
        "version-001:0",
        "version-001:1",
    ]

    assert stored_data["vectors"] == embeddings

    assert stored_data["documents"] == [
        "First chunk",
        "Second chunk",
    ]

    assert stored_data["metadatas"] == [
        {
            "document_id": "organon-001",
            "chunk_index": 0,
            "version_id": "version-001",
        },
        {
            "document_id": "organon-001",
            "chunk_index": 1,
            "version_id": "version-001",
        },
    ]

    assert stored_data["tenant_id"] == "tenant-001"


@pytest.mark.asyncio
async def test_ingest_complete_workflow():
    document_loader = AsyncMock()
    document_loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"source": "organon.txt"},
        )
    ]

    chunker = Mock()
    chunker.chunk.return_value = [
        Document(
            content="ORGANON is a modular",
            metadata={
                "document_id": "organon-001",
                "chunk_index": 0,
            },
        ),
        Document(
            content="RAG framework.",
            metadata={
                "document_id": "organon-001",
                "chunk_index": 1,
            },
        ),
    ]

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    vector_store = AsyncMock()

    # Configure metadata store.
    metadata_store = AsyncMock()

    metadata_store.get_document.return_value = None

    metadata_store.create_document.return_value = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )

    metadata_store.create_version.return_value = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    result = await pipeline.ingest(
        source="organon.txt",
        document_id="organon-001",
        tenant_id="tenant-001",
    )

    assert result == 2

    # Verify document loading.
    document_loader.load.assert_awaited_once_with("organon.txt")

    # Verify chunking.
    chunker.chunk.assert_called_once()

    # Verify embedding generation.
    embedding_provider.embed_documents.assert_awaited_once_with(
        ["ORGANON is a modular", "RAG framework."]
    )

    # Verify vector storage.
    vector_store.upsert.assert_awaited_once()

    stored_data = vector_store.upsert.await_args.kwargs

    assert stored_data["ids"] == [
        "version-001:0",
        "version-001:1",
    ]

    assert stored_data["tenant_id"] == "tenant-001"

    # Verify version lifecycle.
    metadata_store.mark_version_ready.assert_awaited_once_with(
        document_id="organon-001",
        version_id="version-001",
        tenant_id="tenant-001",
    )

    metadata_store.activate_version.assert_awaited_once_with(
        document_id="organon-001",
        version_id="version-001",
        tenant_id="tenant-001",
    )


@pytest.mark.asyncio
async def test_store_chunks_generates_unique_ids_across_documents():
    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=AsyncMock(),
        chunker=Mock(),
        embedding_provider=AsyncMock(),
        vector_store=vector_store,
        metadata_store=AsyncMock(),
    )

    chunks = [
        Document(
            content="First document - chunk 1",
            metadata={"document_id": "company-policy-001"},
        ),
        Document(
            content="First document - chunk 2",
            metadata={"document_id": "company-policy-001"},
        ),
        Document(
            content="Second source - chunk 1",
            metadata={"document_id": "company-policy-001"},
        ),
    ]

    embeddings = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]

    await pipeline._store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        tenant_id="company_a",
        version_id="version-002",
    )

    call_kwargs = vector_store.upsert.await_args.kwargs

    assert call_kwargs["ids"] == [
        "version-002:0",
        "version-002:1",
        "version-002:2",
    ]

    assert all(
        metadata["version_id"] == "version-002"
        for metadata in call_kwargs["metadatas"]
    )


@pytest.mark.asyncio
async def test_get_or_create_document_returns_existing_document():
    metadata_store = AsyncMock()

    existing_document = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )

    metadata_store.get_document.return_value = existing_document

    pipeline = IngestionPipeline(
        document_loader=None,
        chunker=None,
        embedding_provider=None,
        vector_store=None,
        metadata_store=metadata_store,
    )

    result = await pipeline._get_or_create_document(
        document_id="organon-001",
        tenant_id="tenant-001",
    )

    assert result is existing_document

    metadata_store.get_document.assert_awaited_once_with(
        document_id="organon-001",
        tenant_id="tenant-001",
    )

    metadata_store.create_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_or_create_document_creates_new_document():
    metadata_store = AsyncMock()
    metadata_store.get_document.return_value = None

    created_document = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )

    metadata_store.create_document.return_value = created_document

    pipeline = IngestionPipeline(
        document_loader=None,
        chunker=None,
        embedding_provider=None,
        vector_store=None,
        metadata_store=metadata_store,
    )

    result = await pipeline._get_or_create_document(
        document_id="organon-001",
        tenant_id="tenant-001",
    )

    assert result is created_document

    metadata_store.create_document.assert_awaited_once_with(
        DocumentRecord(
            id="organon-001",
            tenant_id="tenant-001",
        )
    )


@pytest.mark.asyncio
async def test_get_or_create_document_handles_concurrent_creation():
    metadata_store = AsyncMock()

    existing_document = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )

    metadata_store.get_document.side_effect = [
        None,
        existing_document,
    ]

    metadata_store.create_document.side_effect = DocumentAlreadyExistsError(
        "Document already exists."
    )

    pipeline = IngestionPipeline(
        document_loader=None,
        chunker=None,
        embedding_provider=None,
        vector_store=None,
        metadata_store=metadata_store,
    )

    result = await pipeline._get_or_create_document(
        document_id="organon-001",
        tenant_id="tenant-001",
    )

    assert result is existing_document
    assert metadata_store.get_document.await_count == 2

    metadata_store.create_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_document_version():
    metadata_store = AsyncMock()

    document = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )

    expected_version = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )

    metadata_store.create_version.return_value = expected_version

    pipeline = IngestionPipeline(
        document_loader=None,
        chunker=None,
        embedding_provider=None,
        vector_store=None,
        metadata_store=metadata_store,
    )

    result = await pipeline._create_document_version(document)

    assert result is expected_version
    assert result.status.value == "processing"

    metadata_store.create_version.assert_awaited_once_with(
        document_id="organon-001",
        tenant_id="tenant-001",
    )


@pytest.mark.asyncio
async def test_ingest_marks_version_failed_when_loading_fails():
    document_loader = AsyncMock()
    document_loader.load.side_effect = RuntimeError("Document loading failed.")

    metadata_store = AsyncMock()

    metadata_store.get_document.return_value = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )

    metadata_store.create_version.return_value = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )

    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=Mock(),
        embedding_provider=AsyncMock(),
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    with pytest.raises(RuntimeError, match="Document loading failed."):
        await pipeline.ingest(
            source="organon.txt",
            document_id="organon-001",
            tenant_id="tenant-001",
        )

    # The unsuccessful version must be marked as FAILED.
    metadata_store.mark_version_failed.assert_awaited_once_with(
        document_id="organon-001",
        version_id="version-001",
        tenant_id="tenant-001",
    )

    # No vectors should be stored or deleted.
    vector_store.upsert.assert_not_awaited()
    vector_store.delete.assert_not_awaited()

    # The version must not become READY or ACTIVE.
    metadata_store.mark_version_ready.assert_not_awaited()
    metadata_store.activate_version.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_marks_version_failed_when_embedding_fails():
    document_loader = AsyncMock()
    document_loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"source": "organon.txt"},
        )
    ]

    chunker = Mock()
    chunker.chunk.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"document_id": "organon-001"},
        )
    ]

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.side_effect = RuntimeError(
        "Embedding generation failed."
    )

    metadata_store = AsyncMock()
    metadata_store.get_document.return_value = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )
    metadata_store.create_version.return_value = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )

    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    with pytest.raises(RuntimeError, match="Embedding generation failed."):
        await pipeline.ingest(
            source="organon.txt",
            document_id="organon-001",
            tenant_id="tenant-001",
        )

    metadata_store.mark_version_failed.assert_awaited_once_with(
        document_id="organon-001",
        version_id="version-001",
        tenant_id="tenant-001",
    )

    # Cleanup is attempted because chunk IDs were prepared.
    # Cleanup is unnecessary because vector storage was never attempted.
    vector_store.delete.assert_not_awaited()

    vector_store.upsert.assert_not_awaited()
    metadata_store.mark_version_ready.assert_not_awaited()
    metadata_store.activate_version.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_marks_version_failed_when_vector_storage_fails():
    document_loader = AsyncMock()
    document_loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"source": "organon.txt"},
        )
    ]

    chunker = Mock()
    chunker.chunk.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"document_id": "organon-001"},
        )
    ]

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    metadata_store = AsyncMock()
    metadata_store.get_document.return_value = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )
    metadata_store.create_version.return_value = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )

    vector_store = AsyncMock()
    vector_store.upsert.side_effect = RuntimeError("Vector storage failed.")

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    with pytest.raises(RuntimeError, match="Vector storage failed."):
        await pipeline.ingest(
            source="organon.txt",
            document_id="organon-001",
            tenant_id="tenant-001",
        )

    vector_store.upsert.assert_awaited_once()
    vector_store.delete.assert_awaited_once_with(
        ids=["version-001:0"],
        tenant_id="tenant-001",
    )

    metadata_store.mark_version_failed.assert_awaited_once_with(
        document_id="organon-001",
        version_id="version-001",
        tenant_id="tenant-001",
    )

    metadata_store.mark_version_ready.assert_not_awaited()
    metadata_store.activate_version.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_preserves_original_error_when_cleanup_fails():
    document_loader = AsyncMock()
    document_loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"source": "organon.txt"},
        )
    ]

    chunker = Mock()
    chunker.chunk.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"document_id": "organon-001"},
        )
    ]

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    metadata_store = AsyncMock()
    metadata_store.get_document.return_value = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )
    metadata_store.create_version.return_value = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )

    vector_store = AsyncMock()
    vector_store.upsert.side_effect = RuntimeError("Vector storage failed.")
    vector_store.delete.side_effect = RuntimeError("Cleanup failed.")

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    with pytest.raises(RuntimeError, match="Vector storage failed.") as exc_info:
        await pipeline.ingest(
            source="organon.txt",
            document_id="organon-001",
            tenant_id="tenant-001",
        )

    assert any(
        "Cleanup failed" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )

    vector_store.delete.assert_awaited_once()
    metadata_store.mark_version_failed.assert_awaited_once()

    metadata_store.mark_version_ready.assert_not_awaited()
    metadata_store.activate_version.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_marks_version_failed_when_chunking_fails():
    document_loader = AsyncMock()
    document_loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"source": "organon.txt"},
        )
    ]

    chunker = Mock()
    chunker.chunk.side_effect = RuntimeError("Chunking failed.")

    metadata_store = AsyncMock()
    metadata_store.get_document.return_value = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )
    metadata_store.create_version.return_value = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )

    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=chunker,
        embedding_provider=AsyncMock(),
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    with pytest.raises(RuntimeError, match="Chunking failed."):
        await pipeline.ingest(
            source="organon.txt",
            document_id="organon-001",
            tenant_id="tenant-001",
        )

    metadata_store.mark_version_failed.assert_awaited_once_with(
        document_id="organon-001",
        version_id="version-001",
        tenant_id="tenant-001",
    )

    vector_store.upsert.assert_not_awaited()
    vector_store.delete.assert_not_awaited()

    metadata_store.mark_version_ready.assert_not_awaited()
    metadata_store.activate_version.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_preserves_ready_version_when_activation_fails():
    document_loader = AsyncMock()
    document_loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"source": "organon.txt"},
        )
    ]

    chunker = Mock()
    chunker.chunk.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"document_id": "organon-001"},
        )
    ]

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    metadata_store = AsyncMock()
    metadata_store.get_document.return_value = DocumentRecord(
        id="organon-001",
        tenant_id="tenant-001",
    )
    metadata_store.create_version.return_value = DocumentVersion(
        id="version-001",
        document_id="organon-001",
        version_number=1,
    )
    metadata_store.activate_version.side_effect = RuntimeError(
        "Version activation failed."
    )

    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    with pytest.raises(RuntimeError, match="Version activation failed."):
        await pipeline.ingest(
            source="organon.txt",
            document_id="organon-001",
            tenant_id="tenant-001",
        )

    vector_store.upsert.assert_awaited_once()

    metadata_store.mark_version_ready.assert_awaited_once_with(
        document_id="organon-001",
        version_id="version-001",
        tenant_id="tenant-001",
    )

    metadata_store.activate_version.assert_awaited_once()

    # Preserve the successfully stored READY version.
    vector_store.delete.assert_not_awaited()
    metadata_store.mark_version_failed.assert_not_awaited()
