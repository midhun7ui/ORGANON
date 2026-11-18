import pytest
import pytest_asyncio

from core.vectorstores.providers.chroma import ChromaVectorStore


@pytest_asyncio.fixture
async def chroma_store():
    store = ChromaVectorStore(
        host="localhost",
        port=8000,
        collection_name="organon_pytest",
    )

    await store.connect()

    yield store

    # Clean up test data after each test.
    await store.delete(ids=["test-doc-1", "test-doc-2"])


@pytest.mark.asyncio
async def test_chroma_upsert(chroma_store):
    await chroma_store.upsert(
        ids=["test-doc-1", "test-doc-2"],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        documents=[
            "Python is a programming language.",
            "React is a JavaScript library.",
        ],
        metadatas=[
            {"category": "python"},
            {"category": "javascript"},
        ],
    )

    assert chroma_store.collection is not None
    assert await chroma_store.collection.count() == 2

@pytest.mark.asyncio
async def test_chroma_search(chroma_store):
    await chroma_store.upsert(
        ids=["test-doc-1", "test-doc-2"],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        documents=[
            "Python is a programming language.",
            "React is a JavaScript library.",
        ],
    )

    results = await chroma_store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].id == "test-doc-1"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_chroma_metadata_filter(chroma_store):
    await chroma_store.upsert(
        ids=["test-doc-1", "test-doc-2"],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        documents=[
            "Python is a programming language.",
            "Django is a Python web framework.",
        ],
        metadatas=[
            {"category": "python"},
            {"category": "django"},
        ],
    )

    results = await chroma_store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        filters={"category": "django"},
    )

    assert len(results) == 1
    assert results[0].id == "test-doc-2"
    assert results[0].metadata["category"] == "django"


@pytest.mark.asyncio
async def test_chroma_delete(chroma_store):
    await chroma_store.upsert(
        ids=["test-doc-1", "test-doc-2"],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        documents=[
            "Python documentation.",
            "React documentation.",
        ],
    )

    # Delete one document.
    await chroma_store.delete(ids=["test-doc-1"])

    # Search for the remaining documents.
    results = await chroma_store.search(
        query_vector=[0.0, 1.0, 0.0],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].id == "test-doc-2"