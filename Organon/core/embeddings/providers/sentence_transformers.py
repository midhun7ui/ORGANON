"""
ORGANON - Sentence Transformers Provider

Local embedding provider using Sentence Transformers.
"""

import asyncio

from sentence_transformers import SentenceTransformer
from core.exceptions import ConfigurationError
from core.embeddings.base import EmbeddingProvider
from core.exceptions import ProviderError


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers implementation of EmbeddingProvider."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:

        if not isinstance(model_name, str) or not model_name.strip():
            raise ConfigurationError(
                "Sentence Transformers model name must be a non-empty string."
            )
        self.model_name = model_name

        # Validate device.
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError(
                "Sentence Transformers device must be a non-empty string."
            )
        self.device = device

        self.model: SentenceTransformer | None = None

    async def connect(self) -> None:
        """Load the Sentence Transformers model."""

        if self.model is not None:
            return

        self.model = await asyncio.to_thread(
            SentenceTransformer,
            self.model_name,
            device=self.device,
        )

    async def close(self) -> None:
        """Release the loaded model."""

        self.model = None

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        if self.model is None:
            raise ProviderError("Sentence Transformers provider is not connected.")

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
        if self.model is None:
            raise ProviderError(
                "Sentence Transformers provider is not connected."
            )
        
        """Generate an embedding for a single query."""

        embedding = await asyncio.to_thread(
            self.model.encode,
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()
