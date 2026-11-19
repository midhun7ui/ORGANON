"""
ORGANON - ChromaDB Provider

Implements the ORGANON VectorStore interface using ChromaDB.
"""

from typing import Any

import chromadb

from core.exceptions import SecurityError
from core.vectorstores.base import VectorStore, VectorSearchResult

from core.vectorstores.validation import (
    validate_upsert_inputs,
    validate_search_inputs,
)


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the ORGANON VectorStore interface."""

    # Metadata fields reserved for internal ORGANON operations.
    RESERVED_METADATA_FIELDS = {
        "tenant_id",
        "organon_document_id",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "organon",
    ) -> None:

        # Store connection configuration.
        self.host = host
        self.port = port

        # Store the collection name for later operations.
        self.collection_name = collection_name

        # Initialize client and collection as None.
        self.client = None
        self.collection = None

    @staticmethod
    def _make_internal_id(tenant_id: str, document_id: str) -> str:
        """Create a unique ChromaDB ID for a tenant and document."""

        return f"{len(tenant_id)}:{tenant_id}{document_id}"

    async def connect(self) -> None:
        """Connect to ChromaDB and initialize the collection."""

        # Establish an asynchronous HTTP connection to ChromaDB.
        self.client = await chromadb.AsyncHttpClient(
            host=self.host,
            port=self.port,
        )

        # Get the collection if it exists.
        # Otherwise, create a new collection.
        self.collection = await self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        tenant_id: str,
    ) -> None:
        """Insert or update documents within a specific tenant."""

        # Ensure the provider is connected.
        if self.collection is None:
            raise RuntimeError("ChromaVectorStore is not connected.")

        # Reject missing tenant identity.
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty.")

        # Validate inputs before interacting with ChromaDB.
        validate_upsert_inputs(
            ids=ids,
            vectors=vectors,
            documents=documents,
            metadatas=metadatas,
        )

        # Prepare metadata for every document.
        document_metadatas = metadatas if metadatas is not None else [{} for _ in ids]

        # Prepare secured metadata for every document.
        secured_metadatas = []

        for metadata in document_metadatas:

            # Identify fields reserved by ORGANON.
            reserved_fields = self.RESERVED_METADATA_FIELDS.intersection(
                metadata.keys()
            )

            # Reject caller-supplied internal metadata fields.
            if reserved_fields:
                raise SecurityError(
                    f"Metadata contains reserved fields: {reserved_fields}"
                )

            # Copy developer-provided metadata.
            secured_metadata = dict(metadata)

            # Attach trusted tenant identity.
            secured_metadata["tenant_id"] = tenant_id

            secured_metadatas.append(secured_metadata)

        # Generate tenant-specific internal IDs.
        internal_ids = [self._make_internal_id(tenant_id, doc_id) for doc_id in ids]

        # Preserve original document IDs in metadata.
        for metadata, doc_id in zip(secured_metadatas, ids):
            metadata["organon_document_id"] = doc_id

        # Insert or update records in ChromaDB.
        await self.collection.upsert(
            ids=internal_ids,
            embeddings=vectors,
            documents=documents,
            metadatas=secured_metadatas,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        *,
        tenant_id: str,
    ) -> list[VectorSearchResult]:
        """Search for documents belonging to a specific tenant."""

        # Ensure the provider is connected.
        if self.collection is None:
            raise RuntimeError("ChromaVectorStore is not connected.")

        # Reject missing tenant identity.
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty.")

        # Validate search inputs before querying ChromaDB.
        validate_search_inputs(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        # Build a mandatory tenant filter.
        tenant_filter = {"tenant_id": tenant_id}

        # Combine tenant isolation with optional user filters.
        if filters:
            secured_filter = {
                "$and": [
                    tenant_filter,
                    filters,
                ]
            }
        else:
            secured_filter = tenant_filter

        # Search ChromaDB using the tenant-scoped filter.
        results = await self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=secured_filter,
        )

        # Extract results for our single query.
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Convert ChromaDB results into ORGANON result objects.
        search_results = []

        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}

            # Retrieve the original ORGANON document ID.
            original_id = metadata["organon_document_id"]

            # Remove internal ORGANON metadata before returning results.
            metadata.pop("organon_document_id", None)

            search_results.append(
                VectorSearchResult(
                    id=original_id,
                    document=document,
                    score=1.0 - distance,
                    metadata=metadata,
                )
            )

        return search_results

    async def delete(
        self,
        ids: list[str],
        *,
        tenant_id: str,
    ) -> None:
        """Delete documents only within the specified tenant."""

        # Ensure the provider is connected.
        if self.collection is None:
            raise RuntimeError("ChromaVectorStore is not connected.")

        # Reject missing tenant identity.
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty.")

        # Convert original IDs into tenant-specific internal IDs.
        internal_ids = [self._make_internal_id(tenant_id, doc_id) for doc_id in ids]

        # Delete only the specified tenant's documents.
        await self.collection.delete(
            ids=internal_ids,
            where={"tenant_id": tenant_id},
        )
