import asyncio
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

        # Track the engine's connection state.
        self._is_connected = False

        # Track providers that still need cleanup.
        self._providers_to_close = []

        # Prevent concurrent lifecycle operations.
        self._lifecycle_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize all providers required by the engine."""

        async with self._lifecycle_lock:

            # Avoid connecting providers more than once.
            if self._is_connected:
                return

            # Prevent startup while previous cleanup is incomplete.
            if self._providers_to_close:
                raise RuntimeError(
                    "Cannot connect while provider cleanup is pending. "
                    "Call close() first."
                )

            attempted_providers = []

            try:
                # Initialize the vector store first.
                attempted_providers.append(self.vector_store)
                await self.vector_store.connect()

                # Initialize the embedding provider next.
                attempted_providers.append(self.embedding_provider)
                await self.embedding_provider.connect()

                # Track providers in reverse initialization order.
                self._providers_to_close = [
                    self.embedding_provider,
                    self.vector_store,
                ]

                self._is_connected = True

            except BaseException as exc:

                # Roll back all providers that were attempted.
                for provider in reversed(attempted_providers):

                    try:
                        await provider.close()

                    except BaseException as cleanup_error:

                        # Preserve providers that still need cleanup.
                        self._providers_to_close.append(provider)

                        # Preserve the original startup exception.
                        exc.add_note(f"Provider cleanup failed: {cleanup_error}")

                self._is_connected = False

                # Re-raise the original exception, including cancellation.
                raise

    async def close(self) -> None:
        """Release all resources used by the engine."""

        async with self._lifecycle_lock:

            # Nothing to close if no providers need cleanup.
            if not self._providers_to_close:
                return

            remaining_providers = []
            errors = []

            # Close providers in reverse initialization order.
            for provider in self._providers_to_close:
                try:
                    await provider.close()

                except Exception as exc:
                    # Record the failure and retain the provider for retry.
                    errors.append(exc)
                    remaining_providers.append(provider)

            # Keep only providers whose cleanup failed.
            self._providers_to_close = remaining_providers

            # The engine is disconnected after shutdown is attempted.
            self._is_connected = False

            # Report cleanup errors without hiding additional failures.
            if errors:
                first_error = errors[0]

                for error in errors[1:]:
                    first_error.add_note(f"Additional cleanup failure: {error}")

                raise first_error

    async def store_documents(
        self,
        credentials: dict[str, Any],
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Authorize, validate, embed, and securely store documents."""

        # Validate inputs before performing security checks.
        validate_document_inputs(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        # Authenticate and authorize access to every document.
        security_context = await self.security_manager.authorize_resources(
            credentials=credentials,
            action="document:write",
            resource_ids=ids,
            resource_type="document",
        )

        # Generate embeddings only after authorization succeeds.
        vectors = await self.embedding_provider.embed_documents(documents)

        # Store documents within the authenticated user's tenant.
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

        # Validate inputs before generating embeddings.
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

        return results
