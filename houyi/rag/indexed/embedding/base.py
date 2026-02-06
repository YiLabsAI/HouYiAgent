"""Embedder base class and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from houyi.rag.config import EmbeddingConfig


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        raise NotImplementedError


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (default: sequential)."""
        return [await self.embed(text) for text in texts]


class APIEmbedder(BaseEmbedder):
    """Embedder using OpenAI/Anthropic API."""

    def __init__(
        self,
        provider: str,
        model: str,
        dimension: int,
    ) -> None:
        super().__init__(dimension)
        self.provider = provider
        self.model = model

    async def embed(self, text: str) -> list[float]:
        """Embed using API."""
        if self.provider == "openai":
            return await self._embed_openai(text)
        elif self.provider == "anthropic":
            # Anthropic doesn't have embedding API, fall back to OpenAI
            return await self._embed_openai(text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _embed_openai(self, text: str) -> list[float]:
        """Embed using OpenAI API."""
        try:
            import openai

            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except ImportError as err:
            raise ImportError("openai package required for API embedding") from err

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using API."""
        if self.provider == "openai":
            try:
                import openai

                client = openai.AsyncOpenAI()
                response = await client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [d.embedding for d in response.data]
            except ImportError as err:
                raise ImportError("openai package required for API embedding") from err
        else:
            return await super().embed_batch(texts)


class LocalEmbedder(BaseEmbedder):
    """Embedder using local FastEmbed model."""

    def __init__(self, model: str, dimension: int) -> None:
        super().__init__(dimension)
        self.model = model
        self._encoder = None

    def _ensure_encoder(self) -> None:
        """Ensure encoder is loaded."""
        if self._encoder is None:
            try:
                from fastembed import TextEmbedding

                self._encoder = TextEmbedding(model_name=self.model)
            except ImportError as err:
                raise ImportError(
                    "fastembed package required for local embedding. "
                    "Install with: pip install fastembed"
                ) from err

    async def embed(self, text: str) -> list[float]:
        """Embed using local model."""
        self._ensure_encoder()
        embeddings = list(self._encoder.embed([text]))
        return embeddings[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using local model."""
        self._ensure_encoder()
        embeddings = list(self._encoder.embed(texts))
        return [e.tolist() for e in embeddings]


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Create embedder from configuration.

    Args:
        config: Embedding configuration

    Returns:
        Embedder instance
    """
    if config.provider == "local":
        return LocalEmbedder(
            model=config.model,
            dimension=config.dimension,
        )
    else:
        return APIEmbedder(
            provider=config.provider,
            model=config.model,
            dimension=config.dimension,
        )
