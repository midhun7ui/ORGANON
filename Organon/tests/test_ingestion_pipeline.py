from unittest.mock import AsyncMock, Mock

import pytest
from core.exceptions import IngestionError
from core.documents.base import Document
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
    chunker = AsyncMock()

    # Chunking is synchronous, so use a regular mock.
    from unittest.mock import Mock

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
    )

    chunks = [
        Document(
            content="First chunk",
            metadata={
                "document_id": "organon-001",
                "chunk_index": 0,
            },
        ),
        Document(
            content="Second chunk",
            metadata={
                "document_id": "organon-001",
                "chunk_index": 1,
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
    )

    vector_store.upsert.assert_awaited_once_with(
        ids=[
            "organon-001:0",
            "organon-001:1",
        ],
        vectors=embeddings,
        documents=[
            "First chunk",
            "Second chunk",
        ],
        metadatas=[
            {
                "document_id": "organon-001",
                "chunk_index": 0,
            },
            {
                "document_id": "organon-001",
                "chunk_index": 1,
            },
        ],
        tenant_id="tenant-001",
    )


@pytest.mark.asyncio
async def test_ingest_complete_workflow():
    # Mock document loader
    document_loader = AsyncMock()
    document_loader.load.return_value = [
        Document(
            content="ORGANON is a modular RAG framework.",
            metadata={"source": "organon.txt"},
        )
    ]

    # Mock chunker
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

    # Mock embedding provider
    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    # Mock vector store
    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=document_loader,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = await pipeline.ingest(
        source="organon.txt",
        document_id="organon-001",
        tenant_id="tenant-001",
    )

    # Verify returned chunk count
    assert result == 2

    # Verify document loading
    document_loader.load.assert_awaited_once_with("organon.txt")

    # Verify chunking
    chunker.chunk.assert_called_once()

    # Verify embedding generation
    embedding_provider.embed_documents.assert_awaited_once_with(
        ["ORGANON is a modular", "RAG framework."]
    )

    # Verify vector storage
    vector_store.upsert.assert_awaited_once()

    stored_data = vector_store.upsert.await_args.kwargs

    assert stored_data["ids"] == [
        "organon-001:0",
        "organon-001:1",
    ]

    assert stored_data["tenant_id"] == "tenant-001"

@pytest.mark.asyncio
async def test_store_chunks_generates_unique_ids_across_documents():
    vector_store = AsyncMock()

    pipeline = IngestionPipeline(
        document_loader=AsyncMock(),
        chunker=Mock(),
        embedding_provider=AsyncMock(),
        vector_store=vector_store,
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

    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    await pipeline._store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        tenant_id="company_a",
    )

    call_kwargs = vector_store.upsert.await_args.kwargs

    assert call_kwargs["ids"] == [
        "company-policy-001:0",
        "company-policy-001:1",
        "company-policy-001:2",
    ]