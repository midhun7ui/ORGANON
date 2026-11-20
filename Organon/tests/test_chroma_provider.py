
import pytest
import pytest_asyncio
from core.exceptions import SecurityError
from core.vectorstores.providers.chroma import ChromaVectorStore
from core.exceptions import ConfigurationError


@pytest_asyncio.fixture
async def chroma_store():
    store = ChromaVectorStore(
        host="localhost",
        port=8000,
        collection_name="organon_pytest",
    )

    await store.connect()

    async def cleanup():
        # Clean up standard test documents.
        await store.delete(
            ids=["test-doc-1", "test-doc-2"],
            tenant_id="tenant-001",
        )

        # Clean up Tenant A test documents.
        await store.delete(
            ids=["shared-doc-001", "shared-delete-001", "security-test-3"],
            tenant_id="tenant-A",
        )

        # Clean up Tenant B test documents.
        await store.delete(
            ids=["shared-doc-001", "shared-delete-001", "tenant-filter-test"],
            tenant_id="tenant-B",
        )

    # Start every test with a clean state.
    await cleanup()

    try:
        yield store
    finally:
        # Clean up after every test, even if it fails.
        await cleanup()


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
        tenant_id="tenant-001",
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
        tenant_id="tenant-001",
    )

    results = await chroma_store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=2,
        tenant_id="tenant-001",
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
        tenant_id="tenant-001",
    )

    results = await chroma_store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        filters={"category": "django"},
        tenant_id="tenant-001",
    )

    assert len(results) == 1
    assert results[0].id == "test-doc-2"
    assert results[0].metadata["category"] == "django"
    assert results[0].metadata["tenant_id"] == "tenant-001"


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
        tenant_id="tenant-001",
    )

    # Delete one document within the specified tenant.
    await chroma_store.delete(
        ids=["test-doc-1"],
        tenant_id="tenant-001",
    )

    # Search for the remaining documents.
    results = await chroma_store.search(
        query_vector=[0.0, 1.0, 0.0],
        top_k=5,
        tenant_id="tenant-001",
    )

    assert len(results) == 1
    assert results[0].id == "test-doc-2"


@pytest.mark.asyncio
async def test_chroma_tenant_duplicate_ids(chroma_store):
    # Tenant A stores a document.
    await chroma_store.upsert(
        ids=["shared-doc-001"],
        vectors=[[1.0, 0.0, 0.0]],
        documents=["Private document belonging to Tenant A."],
        tenant_id="tenant-A",
    )

    # Tenant B stores a document with the same original ID.
    await chroma_store.upsert(
        ids=["shared-doc-001"],
        vectors=[[0.0, 1.0, 0.0]],
        documents=["Private document belonging to Tenant B."],
        tenant_id="tenant-B",
    )

    # Search as Tenant A.
    tenant_a_results = await chroma_store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        tenant_id="tenant-A",
    )

    # Search as Tenant B.
    tenant_b_results = await chroma_store.search(
        query_vector=[0.0, 1.0, 0.0],
        top_k=5,
        tenant_id="tenant-B",
    )

    assert len(tenant_a_results) == 1
    assert len(tenant_b_results) == 1

    assert tenant_a_results[0].id == "shared-doc-001"
    assert tenant_b_results[0].id == "shared-doc-001"

    assert tenant_a_results[0].document == (
        "Private document belonging to Tenant A."
    )

    assert tenant_b_results[0].document == (
        "Private document belonging to Tenant B."
    )


@pytest.mark.asyncio
async def test_chroma_tenant_delete_isolation(chroma_store):
    # Both tenants store a document with the same original ID.
    await chroma_store.upsert(
        ids=["shared-delete-001"],
        vectors=[[1.0, 0.0, 0.0]],
        documents=["Tenant A private document."],
        tenant_id="tenant-A",
    )

    await chroma_store.upsert(
        ids=["shared-delete-001"],
        vectors=[[0.0, 1.0, 0.0]],
        documents=["Tenant B private document."],
        tenant_id="tenant-B",
    )

    # Tenant A deletes its document.
    await chroma_store.delete(
        ids=["shared-delete-001"],
        tenant_id="tenant-A",
    )

    # Search as Tenant A.
    tenant_a_results = await chroma_store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        tenant_id="tenant-A",
    )

    # Search as Tenant B.
    tenant_b_results = await chroma_store.search(
        query_vector=[0.0, 1.0, 0.0],
        top_k=5,
        tenant_id="tenant-B",
    )

    assert len(tenant_a_results) == 0

    assert len(tenant_b_results) == 1
    assert tenant_b_results[0].document == "Tenant B private document."


@pytest.mark.asyncio
async def test_chroma_tenant_filter_cannot_be_overridden(chroma_store):
    # Store a document belonging to Tenant B.
    await chroma_store.upsert(
        ids=["tenant-filter-test"],
        vectors=[[1.0, 0.0, 0.0]],
        documents=["Private document belonging to Tenant B."],
        tenant_id="tenant-B",
    )

    # Tenant A attempts to search using Tenant B's tenant_id filter.
    results = await chroma_store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        filters={"tenant_id": "tenant-B"},
        tenant_id="tenant-A",
    )

    # Tenant A must never receive Tenant B's document.
    assert len(results) == 0


@pytest.mark.asyncio
async def test_chroma_rejects_reserved_tenant_metadata(chroma_store):

    with pytest.raises(SecurityError):
        await chroma_store.upsert(
            ids=["security-test-1"],
            vectors=[[0.1, 0.2, 0.3]],
            documents=["Test document"],
            metadatas=[{"tenant_id": "tenant-B"}],
            tenant_id="tenant-A",
        )


@pytest.mark.asyncio
async def test_chroma_rejects_reserved_document_id_metadata(chroma_store):

    with pytest.raises(SecurityError):
        await chroma_store.upsert(
            ids=["security-test-2"],
            vectors=[[0.1, 0.2, 0.3]],
            documents=["Test document"],
            metadatas=[{"organon_document_id": "fake-id"}],
            tenant_id="tenant-A",
        )


@pytest.mark.asyncio
async def test_chroma_accepts_custom_metadata(chroma_store):

    await chroma_store.upsert(
        ids=["security-test-3"],
        vectors=[[0.1, 0.2, 0.3]],
        documents=["Test document"],
        metadatas=[{"category": "research"}],
        tenant_id="tenant-A",
    )

    results = await chroma_store.search(
        query_vector=[0.1, 0.2, 0.3],
        tenant_id="tenant-A",
    )

    assert len(results) > 0
    assert results[0].metadata["category"] == "research"
    assert results[0].metadata["tenant_id"] == "tenant-A"

def test_chroma_rejects_invalid_port():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="localhost",
            port=70000,
            collection_name="organon_test",
        )

def test_chroma_rejects_invalid_host():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="",
            port=8000,
            collection_name="organon_test",
        )

def test_chroma_rejects_whitespace_host():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="   ",
            port=8000,
            collection_name="organon_test",
        )

def test_chroma_rejects_invalid_collection_name():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="localhost",
            port=8000,
            collection_name="   ",
        )

def test_chroma_rejects_boolean_port():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="localhost",
            port=True,
            collection_name="organon_test",
        )

def test_chroma_rejects_string_port():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="localhost",
            port="8000",
            collection_name="organon_test",
        )

def test_chroma_rejects_zero_port():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="localhost",
            port=0,
            collection_name="organon_test",
        )

def test_chroma_accepts_maximum_valid_port():
    store = ChromaVectorStore(
        host="localhost",
        port=65535,
        collection_name="organon_test",
    )

    assert store.port == 65535

def test_chroma_accepts_minimum_valid_port():
    store = ChromaVectorStore(
        host="localhost",
        port=1,
        collection_name="organon_test",
    )

    assert store.port == 1

def test_chroma_rejects_port_above_maximum():
    with pytest.raises(ConfigurationError):
        ChromaVectorStore(
            host="localhost",
            port=65536,
            collection_name="organon_test",
        )