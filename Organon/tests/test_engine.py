from unittest.mock import AsyncMock

import pytest

from core.engine.engine import OrganonEngine
from core.engine.security_context import SecurityContext
from core.exceptions import AuthorizationError
from core.exceptions import ValidationError


@pytest.mark.asyncio
async def test_store_documents_uses_authenticated_tenant():

    # Create a trusted security context.
    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    # Mock the security manager.
    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Store documents through the engine.
    await engine.store_documents(
        credentials={"token": "test-token"},
        ids=["doc-001"],
        documents=["ORGANON is a modular RAG framework."],
    )

    # Verify authorization.
    security_manager.authorize.assert_awaited_once_with(
        credentials={"token": "test-token"},
        action="document:write",
    )

    # Verify document embedding generation.
    embedding_provider.embed_documents.assert_awaited_once_with(
        ["ORGANON is a modular RAG framework."]
    )

    # Verify that the trusted tenant ID reached the vector store.
    vector_store.upsert.assert_awaited_once_with(
        ids=["doc-001"],
        vectors=[[0.1, 0.2, 0.3]],
        documents=["ORGANON is a modular RAG framework."],
        metadatas=None,
        tenant_id="tenant-A",
    )


@pytest.mark.asyncio
async def test_store_documents_rejects_unauthorized_user():

    # Mock the security manager to reject the request.
    security_manager = AsyncMock()
    security_manager.authorize.side_effect = AuthorizationError(
        "User is not allowed to write documents."
    )

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Verify that the authorization error is raised.
    with pytest.raises(AuthorizationError):
        await engine.store_documents(
            credentials={"token": "invalid-token"},
            ids=["doc-001"],
            documents=["Test document"],
        )

    # Ensure the vector store is never accessed.
    vector_store.upsert.assert_not_awaited()

    # Ensure embeddings are never generated.
    embedding_provider.embed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_documents_uses_authenticated_tenant():

    # Create a trusted security context.
    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    # Mock the security manager.
    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Retrieve documents.
    await engine.retrieve_documents(
        credentials={"token": "test-token"},
        query="What is ORGANON?",
        top_k=5,
    )

    # Verify read authorization.
    security_manager.authorize.assert_awaited_once_with(
        credentials={"token": "test-token"},
        action="document:read",
    )

    # Verify query embedding generation.
    embedding_provider.embed_query.assert_awaited_once_with("What is ORGANON?")

    # Verify tenant-scoped search.
    vector_store.search.assert_awaited_once_with(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        filters=None,
        tenant_id="tenant-A",
    )


@pytest.mark.asyncio
async def test_retrieve_documents_rejects_unauthorized_user():

    # Mock the security manager to reject the request.
    security_manager = AsyncMock()
    security_manager.authorize.side_effect = AuthorizationError(
        "User is not allowed to read documents."
    )

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Verify that retrieval raises an authorization error.
    with pytest.raises(AuthorizationError):
        await engine.retrieve_documents(
            credentials={"token": "invalid-token"},
            query="What is ORGANON?",
        )

    # Ensure the vector store is never accessed.
    vector_store.search.assert_not_awaited()

    # Ensure embeddings are never generated.
    embedding_provider.embed_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_documents_rejects_empty_documents_before_embedding():

    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(ValidationError):
        await engine.store_documents(
            credentials={"token": "test-token"},
            ids=[],
            documents=[],
        )

    # Embedding generation must never happen.
    embedding_provider.embed_documents.assert_not_awaited()

    # Vector store must never be called.
    vector_store.upsert.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, top_k",
    [
        ("", 5),
        ("   ", 5),
        ("What is ORGANON?", 0),
        ("What is ORGANON?", -1),
        ("What is ORGANON?", True),
    ],
)
async def test_retrieve_documents_rejects_invalid_inputs_before_embedding(
    query,
    top_k,
):
    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(ValidationError):
        await engine.retrieve_documents(
            credentials={"token": "test-token"},
            query=query,
            top_k=top_k,
        )

    embedding_provider.embed_query.assert_not_awaited()
    vector_store.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_documents_rejects_invalid_metadata_before_embedding():
    security_manager = AsyncMock()
    security_manager.authorize.return_value = SecurityContext(
        user_id="user-1",
        tenant_id="tenant-1",
        permissions=frozenset({"document:write"}),
    )

    embedding_provider = AsyncMock()
    vector_store = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(ValidationError):
        await engine.store_documents(
            credentials={"token": "valid"},
            ids=["doc-1"],
            documents=["ORGANON is a modular RAG framework."],
            metadatas=[{"tags": ["Python", "AI"]}],
        )

    embedding_provider.embed_documents.assert_not_awaited()
    vector_store.upsert.assert_not_awaited()
