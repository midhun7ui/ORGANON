from pathlib import Path

from core.documents.base import Document, DocumentLoader
from core.chunking.base import Chunker
from core.embeddings.base import EmbeddingProvider
from core.vectorstores.base import VectorStore
from core.exceptions import (
    IngestionError,
    ProviderError,
    DocumentAlreadyExistsError,
)
from core.metadata.base import MetadataStore
from core.documents.models import DocumentRecord, DocumentVersion


class IngestionPipeline:
    """Coordinates the document ingestion workflow."""

    def __init__(
        self,
        document_loader: DocumentLoader,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self.document_loader = document_loader
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    async def _get_or_create_document(
        self,
        document_id: str,
        tenant_id: str,
    ) -> DocumentRecord:
        """Retrieve an existing document or create a new one."""

        # First, check whether the document already exists.
        document = await self.metadata_store.get_document(
            document_id=document_id,
            tenant_id=tenant_id,
        )

        if document is not None:
            return document

        # Create a new document if it does not exist.
        try:
            return await self.metadata_store.create_document(
                DocumentRecord(
                    id=document_id,
                    tenant_id=tenant_id,
                )
            )

        except DocumentAlreadyExistsError:
            # Another request may have created it concurrently.
            document = await self.metadata_store.get_document(
                document_id=document_id,
                tenant_id=tenant_id,
            )

        if document is None:
            raise

        return document

    async def _create_document_version(
        self,
        document: DocumentRecord,
    ) -> DocumentVersion:
        """Create a new processing version for a document."""

        return await self.metadata_store.create_version(
            document_id=document.id,
            tenant_id=document.tenant_id,
        )

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
        version_id: str,
    ) -> None:
        """Store version-specific document chunks and their embeddings."""

        if len(chunks) != len(embeddings):
            raise IngestionError("Number of chunks and embeddings must match.")

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise IngestionError("Tenant ID must be a non-empty string.")

        if not isinstance(version_id, str) or not version_id.strip():
            raise IngestionError("Version ID must be a non-empty string.")

        ids = []
        documents = []
        metadatas = []

        # Process every chunk individually.
        for index, chunk in enumerate(chunks):
            document_id = chunk.metadata.get("document_id")

            if not isinstance(document_id, str) or not document_id.strip():
                raise IngestionError("Chunk is missing its document ID.")

            # Generate a unique ID for this version's chunk.
            chunk_id = f"{version_id}:{index}"

            ids.append(chunk_id)
            documents.append(chunk.content)

            # Attach ORGANON-managed version information.
            metadata = chunk.metadata.copy()

            metadata.update(
                {
                    "document_id": document_id,
                    "version_id": version_id,
                    "chunk_index": index,
                }
            )

            metadatas.append(metadata)

        # Store all chunks together.
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
        """Ingest a document with versioning and failure handling."""

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise IngestionError("Tenant ID must be a non-empty string.")

        document = await self._get_or_create_document(
            document_id=document_id,
            tenant_id=tenant_id,
        )

        # Create a new version in PROCESSING state.
        version = await self._create_document_version(document)

        # Track chunk IDs for possible cleanup.
        chunk_ids: list[str] = []

        # Tracks whether vector storage has been attempted.
        vector_storage_attempted = False

        # A READY version must be preserved for recovery.
        version_ready = False

        try:
            # 1. Load source documents.
            documents = await self._load_documents(
                source=source,
                document_id=document.id,
            )

            # 2. Split documents into chunks.
            chunks = self._chunk_documents(documents)

            # Prepare IDs before storing vectors.
            chunk_ids = [f"{version.id}:{index}" for index in range(len(chunks))]

            # 3. Generate embeddings.
            embeddings = await self._embed_chunks(chunks)

            # 4. Store chunks in the vector database.
            vector_storage_attempted = True

            await self._store_chunks(
                chunks=chunks,
                embeddings=embeddings,
                tenant_id=document.tenant_id,
                version_id=version.id,
            )

            # 5. Mark the version READY.
            await self.metadata_store.mark_version_ready(
                document_id=document.id,
                version_id=version.id,
                tenant_id=document.tenant_id,
            )

            version_ready = True

            # 6. Activate the new version.
            await self.metadata_store.activate_version(
                document_id=document.id,
                version_id=version.id,
                tenant_id=document.tenant_id,
            )

            return len(chunks)

        except Exception as exc:

            # Preserve READY versions for later recovery.
            if version_ready:
                raise

            # Attempt to remove partially stored vector data.
            if vector_storage_attempted and chunk_ids:
                try:
                    await self.vector_store.delete(
                        ids=chunk_ids,
                        tenant_id=document.tenant_id,
                    )

                except Exception as cleanup_error:
                    exc.add_note(f"Vector cleanup failed: {cleanup_error!r}")

            # Mark the unsuccessful PROCESSING version as FAILED.
            try:
                await self.metadata_store.mark_version_failed(
                    document_id=document.id,
                    version_id=version.id,
                    tenant_id=document.tenant_id,
                )

            except Exception as failure_error:
                exc.add_note(
                    f"Failed to mark document version as FAILED: " f"{failure_error!r}"
                )

            # Re-raise the original exception.
            raise
