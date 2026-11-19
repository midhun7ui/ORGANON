"""
ORGANON - Sentence Transformers Provider

Local embedding provider using Sentence Transformers.
"""

import asyncio

from sentence_transformers import SentenceTransformer

from core.embeddings.base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers implementation of EmbeddingProvider."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a single query."""

        embedding = await asyncio.to_thread(
            self.model.encode,
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()
