
import pytest

from core.embeddings.providers.sentence_transformers import (
    SentenceTransformerProvider,
)


@pytest.fixture(scope="module")
def provider():
    return SentenceTransformerProvider()


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

    magnitude = sum(value ** 2 for value in embedding) ** 0.5

    assert magnitude == pytest.approx(1.0, abs=1e-5)
