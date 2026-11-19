import pytest

from core.exceptions import RegistryError
from core.registry import Registry
from core.vectorstores.providers.chroma import ChromaVectorStore
from core.vectorstores.registration import (
    register_vector_stores,
    create_vector_store,
)


def test_register_vector_stores():
    registry = Registry()

    register_vector_stores(registry)

    providers = registry.list_providers("vector_store")

    assert "chroma" in providers


def test_create_vector_store():
    registry = Registry()

    register_vector_stores(registry)

    store = create_vector_store(
        registry=registry,
        provider="chroma",
        config={
            "host": "localhost",
            "port": 8000,
            "collection_name": "organon_test",
        },
    )

    assert isinstance(store, ChromaVectorStore)
    assert store.host == "localhost"
    assert store.port == 8000
    assert store.collection_name == "organon_test"


def test_create_unknown_vector_store():
    registry = Registry()

    register_vector_stores(registry)

    with pytest.raises(RegistryError):
        create_vector_store(
            registry=registry,
            provider="unknown",
            config={},
        )
