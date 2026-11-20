from core.registry import Registry
from core.chunking.providers.fixed_size import FixedSizeChunker
from core.chunking.base import Chunker


def register_chunkers(registry: Registry) -> None:
    """Register available chunking providers."""

    registry.register(
        category="chunker",
        name="fixed_size",
        provider=FixedSizeChunker,
    )


def create_chunker(
    registry: Registry,
    provider: str,
    config: dict | None = None,
) -> Chunker:
    """Create a chunker from the registry."""

    provider_factory = registry.get("chunker", provider)

    chunker = provider_factory(**(config or {}))

    return chunker
