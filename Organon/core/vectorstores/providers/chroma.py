"""
ORGANON - ChromaDB Provider

Implements the ORGANON VectorStore interface using ChromaDB.
"""

from typing import Any

import chromadb

from core.vectorstores.base import VectorStore, VectorSearchResult


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the ORGANON VectorStore interface."""

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
    ) -> None:
        """Insert new documents or update existing documents."""

        # Ensure the provider is connected before performing operations.
        if self.collection is None:
            raise RuntimeError("ChromaVectorStore is not connected.")

        # Insert or update records in ChromaDB.
        await self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar documents in ChromaDB."""

        # Ensure the provider is connected.
        if self.collection is None:
            raise RuntimeError("ChromaVectorStore is not connected.")

        # Search ChromaDB using the supplied query vector.
        results = await self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filters,
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
            search_results.append(
                VectorSearchResult(
                    id=doc_id,
                    document=document,
                    score=1.0 - distance,
                    metadata=metadata or {},
                )
            )

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """Delete documents from ChromaDB using their IDs."""

        # Ensure the provider is connected before performing operations.
        if self.collection is None:
            raise RuntimeError("ChromaVectorStore is not connected.")

        # Delete the requested documents from ChromaDB.
        await self.collection.delete(ids=ids)

