import pytest

from core.chunking.registration import register_chunkers
from core.chunking.providers.fixed_size import FixedSizeChunker
from core.registry import Registry
from core.chunking.registration import create_chunker
from core.exceptions import RegistryError

def test_registers_fixed_size_chunker():
    registry = Registry()

    register_chunkers(registry)

    provider = registry.get("chunker", "fixed_size")

    assert provider is FixedSizeChunker


def test_creates_fixed_size_chunker():
    registry = Registry()
    register_chunkers(registry)

    chunker = create_chunker(
        registry=registry,
        provider="fixed_size",
        config={
            "chunk_size": 500,
            "overlap": 50,
        },
    )

    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_size == 500
    assert chunker.overlap == 50

def test_unknown_chunker_provider_raises_error():
    registry = Registry()
    register_chunkers(registry)

    with pytest.raises(RegistryError):
        create_chunker(
            registry=registry,
            provider="unknown",
        )