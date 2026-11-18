import pytest

from core.embeddings.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Fake provider used for testing the embedding interface."""

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_cannot_instantiate_abstract_provider():
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_can_instantiate_valid_provider():
    provider = FakeEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)
