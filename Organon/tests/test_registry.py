import pytest

from core.registry import Registry
from core.exceptions import RegistryError

class FakeEmbeddingProvider:
    pass

def test_register_and_get_provider():
    registry = Registry()

    registry.register(
        "embeddings",
        "fake",
        FakeEmbeddingProvider,
    )

    provider = registry.get("embeddings", "fake")

    assert provider is FakeEmbeddingProvider

def test_duplicate_registration():
    registry = Registry()

    registry.register(
        "embeddings",
        "fake",
        FakeEmbeddingProvider,
    )

    with pytest.raises(RegistryError):
        registry.register(
            "embeddings",
            "fake",
            FakeEmbeddingProvider
        )


def test_unknown_provider():
    registry = Registry()

    with pytest.raises(RegistryError):
        registry.get("embeddings", "unknown")

def test_list_providers():
    registry = Registry()

    registry.register(
        "embeddings",
        "fake",
        FakeEmbeddingProvider,
    )

    assert registry.list_providers("embeddings") == ["fake"]