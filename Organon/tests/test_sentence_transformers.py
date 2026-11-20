import pytest

from core.embeddings.providers.sentence_transformers import (
    SentenceTransformerProvider,
)
from core.exceptions import ConfigurationError

from core.exceptions import ProviderError

from unittest.mock import MagicMock

import core.embeddings.providers.sentence_transformers as st_module 


@pytest.fixture
async def provider():
    provider = SentenceTransformerProvider()

    await provider.connect()

    yield provider

    await provider.close()


@pytest.mark.asyncio
async def test_embed_documents(provider):
    texts = [
        "Python is a programming language.",
        "Machine learning uses data to learn patterns.",
    ]

    embeddings = await provider.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


@pytest.mark.asyncio
async def test_embed_query(provider):
    embedding = await provider.embed_query("What is Python?")

    assert len(embedding) == 384


@pytest.mark.asyncio
async def test_embeddings_are_normalized(provider):
    embedding = await provider.embed_query("Artificial Intelligence")

    magnitude = sum(value**2 for value in embedding) ** 0.5

    assert magnitude == pytest.approx(1.0, abs=1e-5)


def test_rejects_empty_model_name():
    with pytest.raises(ConfigurationError):
        SentenceTransformerProvider(model_name="")


def test_rejects_whitespace_model_name():
    with pytest.raises(ConfigurationError):
        SentenceTransformerProvider(model_name="   ")


def test_rejects_empty_device():
    with pytest.raises(ConfigurationError):
        SentenceTransformerProvider(device="")


def test_rejects_whitespace_device():
    with pytest.raises(ConfigurationError):
        SentenceTransformerProvider(device="   ")


@pytest.mark.asyncio
async def test_embed_documents_requires_connection():
    provider = SentenceTransformerProvider()

    with pytest.raises(ProviderError):
        await provider.embed_documents(["Hello ORGANON"])


@pytest.mark.asyncio
async def test_embed_query_requires_connection():
    provider = SentenceTransformerProvider()

    with pytest.raises(ProviderError):
        await provider.embed_query("Hello ORGANON")

@pytest.mark.asyncio
async def test_connect_is_idempotent(monkeypatch):
    model = MagicMock()
    model_loader = MagicMock(return_value=model)

    monkeypatch.setattr(
        st_module,
        "SentenceTransformer",
        model_loader,
    )

    provider = SentenceTransformerProvider()

    await provider.connect()
    await provider.connect()

    model_loader.assert_called_once_with(
        provider.model_name,
        device=provider.device,
    )

    await provider.close()


@pytest.mark.asyncio
async def test_close_is_idempotent():
    provider = SentenceTransformerProvider()
    provider.model = MagicMock()

    await provider.close()

    assert provider.model is None

    # Closing an already closed provider should be safe.
    await provider.close()

    assert provider.model is None


@pytest.mark.asyncio
async def test_connect_failure_keeps_provider_disconnected(monkeypatch):
    def failing_loader(*args, **kwargs):
        raise RuntimeError("Model loading failed")

    monkeypatch.setattr(
        st_module,
        "SentenceTransformer",
        failing_loader,
    )

    provider = SentenceTransformerProvider()

    with pytest.raises(RuntimeError, match="Model loading failed"):
        await provider.connect()

    assert provider.model is None


@pytest.mark.asyncio
async def test_connect_failure_keeps_provider_disconnected(monkeypatch):
    def failing_loader(*args, **kwargs):
        raise RuntimeError("Model loading failed")

    monkeypatch.setattr(
        st_module,
        "SentenceTransformer",
        failing_loader,
    )

    provider = SentenceTransformerProvider()

    with pytest.raises(RuntimeError, match="Model loading failed"):
        await provider.connect()

    assert provider.model is None
