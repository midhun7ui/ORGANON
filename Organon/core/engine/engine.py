from typing import Any

from core.embeddings.base import EmbeddingProvider
from core.security.manager import SecurityManager
from core.validation import validate_document_inputs, validate_query_input
from core.vectorstores.base import VectorStore, VectorSearchResult


class OrganonEngine:
    """Main orchestration engine for ORGANON."""

    def __init__(
        self,
        security_manager: SecurityManager,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.security_manager = security_manager
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def store_documents(
        self,
        credentials: dict[str, Any],
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Authenticate, authorize, and store documents securely."""

        # Authenticate the requester and verify document write permission.
        security_context = await self.security_manager.authorize(
            credentials=credentials,
            action="document:write",
        )

        # Validate document inputs before generating embeddings.
        validate_document_inputs(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        # Generate embeddings for the documents.
        vectors = await self.embedding_provider.embed_documents(documents)

        # Use the trusted tenant ID from the security context.
        # Never accept tenant_id directly from the requester.
        await self.vector_store.upsert(
            ids=ids,
            vectors=vectors,
            documents=documents,
            metadatas=metadatas,
            tenant_id=security_context.tenant_id,
        )

    async def retrieve_documents(
        self,
        credentials: dict[str, Any],
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Authenticate, authorize, and retrieve documents securely."""

        # Authenticate the requester and verify read permission.
        security_context = await self.security_manager.authorize(
            credentials=credentials,
            action="document:read",
        )

        # Validate retrieval inputs before generating embeddings.
        validate_query_input(
            query=query,
            top_k=top_k,
            filters=filters,
        )

        # Generate an embedding for the user's query.
        query_vector = await self.embedding_provider.embed_query(query)

        # Search only within the authenticated user's tenant.
        results = await self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            tenant_id=security_context.tenant_id,
        )

        # Return the retrieved documents.
        return results
