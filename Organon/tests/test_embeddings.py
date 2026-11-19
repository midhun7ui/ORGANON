import pytest

from core.exceptions import RegistryError
from core.registry import Registry

from core.embeddings.base import EmbeddingProvider
from core.embeddings.registration import (
    create_embedding_provider,
    register_embedding_providers,
)
from core.embeddings.providers.sentence_transformers import (
    SentenceTransformerProvider,
)


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


def test_create_embedding_provider():
    registry = Registry()

    registry.register(
        category="embedding",
        name="fake",
        provider=FakeEmbeddingProvider,
    )

    provider = create_embedding_provider(
        registry=registry,
        provider="fake",
        config={},
    )

    assert isinstance(provider, FakeEmbeddingProvider)
    assert isinstance(provider, EmbeddingProvider)


def test_create_unknown_embedding_provider():
    registry = Registry()

    with pytest.raises(RegistryError):
        create_embedding_provider(
            registry=registry,
            provider="unknown",
            config={},
        )


def test_register_embedding_providers():
    registry = Registry()

    register_embedding_providers(registry)

    provider_factory = registry.get(
        "embedding",
        "sentence_transformers",
    )

    assert provider_factory is SentenceTransformerProvider
