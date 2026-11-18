import pytest

from core.vectorstores.base import VectorStore, VectorSearchResult


class FakeVectorStore(VectorStore):
    """Fake vector store for testing the interface."""

    async def add(
        self,
        ids,
        vectors,
        documents,
        metadatas=None,
    ):
        pass

    async def search(self, query_verctor, top_k=5, filters=None):
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
