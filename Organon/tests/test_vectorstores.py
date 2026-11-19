import pytest
import asyncio

from core.vectorstores.base import VectorStore, VectorSearchResult


class FakeVectorStore(VectorStore):
    """Fake vector store for testing the interface."""

    async def upsert(
        self,
        ids,
        vectors,
        documents,
        metadatas=None,
    ):
        pass

    async def search(
        self,
        query_vector,
        top_k=5,
        filters=None,
    ):
        return [
            VectorSearchResult(
                id="doc_001",
                document="ORGANON is a modular RAG framework.",
                score=0.95,
                metadata={"source": "organon.pdf"},
            )
        ]

    async def delete(self, ids):
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory implementation for testing vector store operations."""

    def __init__(self):
        self.records = {}

    async def upsert(
        self,
        ids,
        vectors,
        documents,
        metadatas=None,
    ):
        if metadatas is None:
            metadatas = [{} for _ in ids]

        for doc_id, vector, document, metadata in zip(
            ids, vectors, documents, metadatas
        ):
            self.records[doc_id] = {
                "vector": vector,
                "document": document,
                "metadata": metadata,
            }

    async def search(self, query_vector, top_k=5, filters=None):
        pass

    async def delete(self, ids):
        for doc_id in ids:
            self.records.pop(doc_id, None)


def test_upsert_inserts_documents():
    store = InMemoryVectorStore()

    asyncio.run(
        store.upsert(
            ids=["doc_001"],
            vectors=[[0.1, 0.2, 0.3]],
            documents=["ORGANON is a RAG framework."],
            metadatas=[{"source": "guide.pdf"}],
        )
    )

    assert "doc_001" in store.records
    assert store.records["doc_001"]["document"] == "ORGANON is a RAG framework."


def test_upsert_updates_existing_document():
    store = InMemoryVectorStore()

    asyncio.run(
        store.upsert(
            ids=["doc_001"],
            vectors=[[0.1, 0.2, 0.3]],
            documents=["Version 1"],
            metadatas=[{"version": 1, "status": "draft"}],
        )
    )

    asyncio.run(
        store.upsert(
            ids=["doc_001"],
            vectors=[[0.4, 0.5, 0.6]],
            documents=["Version 2"],
            metadatas=[{"version": 2}],
        )
    )

    assert store.records["doc_001"]["document"] == "Version 2"
    assert store.records["doc_001"]["vector"] == [0.4, 0.5, 0.6]
    assert store.records["doc_001"]["metadata"] == {"version": 2}


def test_delete_documents():
    store = InMemoryVectorStore()

    asyncio.run(
        store.upsert(
            ids=["doc_001", "doc_002"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            documents=["Document 1", "Document 2"],
        )
    )

    asyncio.run(store.delete(["doc_001"]))

    assert "doc_001" not in store.records
    assert "doc_002" in store.records


def test_delete_nonexistent_document():
    store = InMemoryVectorStore()

    asyncio.run(store.delete(["unknown_doc"]))

    assert "unknown_doc" not in store.records


def test_cannot_instantiate_abstract_vector_store():
    with pytest.raises(TypeError):
        VectorStore()


def test_can_instantiate_valid_vector_store():
    store = FakeVectorStore()
    assert isinstance(store, VectorStore)


def test_vector_search_result():
    result = VectorSearchResult(
        id="doc_001",
        document="ORGANON is a modular RAG framework.",
        score=0.95,
        metadata={"source": "organon.pdf"},
    )

    assert result.id == "doc_001"
    assert result.score == 0.95
    assert result.metadata["source"] == "organon.pdf"
