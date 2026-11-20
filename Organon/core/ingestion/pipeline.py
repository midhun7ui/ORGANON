
from pathlib import Path

from core.documents.base import Document, DocumentLoader
from core.chunking.base import Chunker
from core.embeddings.base import EmbeddingProvider
from core.vectorstores.base import VectorStore
from core.exceptions import IngestionError, ProviderError


class IngestionPipeline:
    """Coordinates the document ingestion workflow."""

    def __init__(
        self,
        document_loader: DocumentLoader,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.document_loader = document_loader
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def _load_documents(
        self,
        source: str | Path,
        document_id: str,
    ) -> list[Document]:
        """Load documents and attach their original document ID."""

        if not isinstance(document_id, str) or not document_id.strip():
            raise IngestionError("Document ID must be a non-empty string.")

        documents = await self.document_loader.load(source)

        if not documents:
            raise IngestionError("Document loader returned no documents.")

        loaded_documents = []

        for document in documents:
            loaded_document = Document(
                content=document.content,
                metadata={
                    **document.metadata,
                    "document_id": document_id,
                },
            )

            loaded_documents.append(loaded_document)

        return loaded_documents

    def _chunk_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Split loaded documents into smaller chunks."""

        chunks = []

        for document in documents:
            document_chunks = self.chunker.chunk(document)
            chunks.extend(document_chunks)

        if not chunks:
            raise IngestionError("Chunking produced no chunks.")

        return chunks

    async def _embed_chunks(
        self,
        chunks: list[Document],
    ) -> list[list[float]]:
        """Generate embeddings for document chunks."""

        texts = [chunk.content for chunk in chunks]

        embeddings = await self.embedding_provider.embed_documents(texts)

        if len(embeddings) != len(chunks):
            raise ProviderError(
                "Embedding provider returned an unexpected number of vectors."
            )

        return embeddings

    async def _store_chunks(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
        tenant_id: str,
    ) -> None:
        """Store document chunks and their embeddings."""

        if len(chunks) != len(embeddings):
            raise IngestionError(
                "Number of chunks and embeddings must match."
            )

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise IngestionError("Tenant ID must be a non-empty string.")

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            document_id = chunk.metadata.get("document_id")

            if not isinstance(document_id, str) or not document_id.strip():
                raise IngestionError("Chunk is missing its document ID.")

            chunk_id = f"{document_id}:{index}"

            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata.copy())

        await self.vector_store.upsert(
            ids=ids,
            vectors=embeddings,
            documents=documents,
            metadatas=metadatas,
            tenant_id=tenant_id,
        )

    async def ingest(
        self,
        source: str | Path,
        document_id: str,
        tenant_id: str,
    ) -> int:
        """Execute the complete document ingestion workflow."""

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise IngestionError("Tenant ID must be a non-empty string.")

        # Stage 1: Load documents
        documents = await self._load_documents(
            source=source,
            document_id=document_id,
        )

        # Stage 2: Split documents into chunks
        chunks = self._chunk_documents(documents)

        # Stage 3: Generate embeddings
        embeddings = await self._embed_chunks(chunks)

        # Stage 4: Store chunks and embeddings
        await self._store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            tenant_id=tenant_id,
        )

        return len(chunks)
