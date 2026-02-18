"""Embedder base class and factory."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from houyi.rag.config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Progress callback type: (processed, total, current_batch) -> None
ProgressCallback = Callable[[int, int, int], None]


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        raise NotImplementedError

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
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

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Embed multiple texts (default: sequential)."""
        results = []
        for i, text in enumerate(texts):
            results.append(await self.embed(text))
            if progress_callback:
                progress_callback(i + 1, len(texts), 1)
        return results


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

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Batch embed using API."""
        if self.provider == "openai":
            try:
                import openai

                client = openai.AsyncOpenAI()
                response = await client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                if progress_callback:
                    progress_callback(len(texts), len(texts), len(texts))
                return [d.embedding for d in response.data]
            except ImportError as err:
                raise ImportError("openai package required for API embedding") from err
        else:
            return await super().embed_batch(texts, progress_callback)


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

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Batch embed using local model."""
        self._ensure_encoder()
        embeddings = list(self._encoder.embed(texts))
        if progress_callback:
            progress_callback(len(texts), len(texts), len(texts))
        return [e.tolist() for e in embeddings]


class GeminiEmbedder(BaseEmbedder):
    """Embedder using Google Vertex AI Gemini (via google-genai SDK).

    Features:
    - Rate limiting to avoid 429 errors (configurable delay between batches)
    - Exponential backoff retry on rate limit errors
    - Progress callback for UI updates
    - Configurable batch size for token limit management
    """

    # Gemini API limits:
    # - Per-text: 2048 tokens max
    # - Per-batch: 20000 tokens max
    # - Rate limit: varies by project quota (typically 60 RPM, some as low as 30 RPM)
    DEFAULT_BATCH_SIZE = 2  # Conservative: 2 chunks * ~2000 tokens = 4000 tokens
    DEFAULT_DELAY_SECONDS = 2.0  # Delay between batches (2s = 30 RPM max, safer for low quotas)
    MAX_RETRIES = 5  # Maximum retry attempts on 429 errors
    INITIAL_BACKOFF = 2.0  # Initial backoff delay in seconds
    MAX_BACKOFF = 60.0  # Maximum backoff delay
    DELAY_MULTIPLIER = 2.0  # Multiplier for adaptive delay increase

    def __init__(
        self,
        model: str,
        dimension: int,
        project: str | None = None,
        location: str | None = None,
        batch_size: int | None = None,
        delay_seconds: float | None = None,
    ) -> None:
        super().__init__(dimension)
        self.model = model
        self._client = None
        self._project = project
        self._location = location
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.delay_seconds = delay_seconds or self.DEFAULT_DELAY_SECONDS

    def _ensure_client(self) -> None:
        """Ensure Gemini client is initialized.

        Supports two authentication modes (following ``google-genai`` SDK conventions):
        1. **GOOGLE_API_KEY** — direct Gemini API access (no GCP project needed).
        2. **Vertex AI** — ``GOOGLE_CLOUD_PROJECT`` + ``GOOGLE_APPLICATION_CREDENTIALS``.

        Tries API-key auth first (simpler), then falls back to Vertex AI.
        """
        if self._client is not None:
            return

        try:
            from google import genai  # type: ignore[attr-defined]
        except ImportError as err:
            raise ImportError(
                "google-genai package required for Gemini embedding. "
                "Install with: pip install google-genai"
            ) from err

        import os

        # 1. API-key auth (no project required)
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self._client = genai.Client(api_key=api_key)
            logger.info("GeminiEmbedder: using GOOGLE_API_KEY auth")
            return

        # 2. Vertex AI auth (requires project)
        project = self._project or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = self._location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"

        if not project:
            raise ValueError(
                "Gemini embedding requires either:\n"
                "  - GOOGLE_API_KEY for direct API access, or\n"
                "  - GOOGLE_CLOUD_PROJECT for Vertex AI"
            )

        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        logger.info("GeminiEmbedder: using Vertex AI auth (project=%s)", project)

    async def embed(self, text: str) -> list[float]:
        """Embed using Gemini API with retry logic."""
        return await self._embed_with_retry(text)

    async def _embed_with_retry(self, text: str) -> list[float]:
        """Embed with exponential backoff retry on rate limit errors."""
        self._ensure_client()

        backoff = self.INITIAL_BACKOFF
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await self._client.aio.models.embed_content(
                    model=self.model,
                    contents=text,
                )
                return list(result.embeddings[0].values)
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str
                )

                if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.5 * backoff)
                    wait_time = backoff + jitter
                    logger.warning(
                        "Rate limit hit (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        wait_time,
                        e,
                    )
                    await asyncio.sleep(wait_time)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                else:
                    raise

        # Should not reach here, but just in case
        raise RuntimeError("Max retries exceeded for embedding")

    async def _embed_batch_single(self, batch: list[str]) -> tuple[list[list[float]], bool]:
        """Embed a single batch with retry logic.

        Returns:
            Tuple of (embeddings, rate_limited) where rate_limited indicates if 429 was hit
        """
        self._ensure_client()

        backoff = self.INITIAL_BACKOFF
        rate_limited = False

        for attempt in range(self.MAX_RETRIES):
            try:
                result = await self._client.aio.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                return [list(e.values) for e in result.embeddings], rate_limited
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str
                )

                if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                    rate_limited = True
                    jitter = random.uniform(0, 0.5 * backoff)
                    wait_time = backoff + jitter
                    logger.warning(
                        "Rate limit hit on batch (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        wait_time,
                        e,
                    )
                    await asyncio.sleep(wait_time)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                else:
                    raise

        raise RuntimeError("Max retries exceeded for batch embedding")

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Batch embed with adaptive rate limiting and progress reporting.

        Features:
        - Automatic delay between batches to avoid rate limits
        - Adaptive delay increase when 429 errors are encountered
        - Exponential backoff retry on rate limit errors
        - Progress callback for UI updates

        Args:
            texts: List of texts to embed
            progress_callback: Optional callback(processed, total, batch_size)

        Returns:
            List of embedding vectors
        """
        self._ensure_client()

        all_embeddings: list[list[float]] = []
        total = len(texts)
        processed = 0
        current_delay = self.delay_seconds  # Adaptive delay

        # Split into batches
        batches = []
        for i in range(0, total, self.batch_size):
            batches.append(texts[i : i + self.batch_size])

        logger.info(
            "Embedding %d texts in %d batches (batch_size=%d, delay=%.1fs)",
            total,
            len(batches),
            self.batch_size,
            current_delay,
        )

        for batch_idx, batch in enumerate(batches):
            # Embed this batch with retry
            batch_embeddings, rate_limited = await self._embed_batch_single(batch)
            all_embeddings.extend(batch_embeddings)

            # If rate limited, increase delay for subsequent batches
            if rate_limited:
                old_delay = current_delay
                current_delay = min(current_delay * self.DELAY_MULTIPLIER, 10.0)  # Max 10 seconds
                logger.info(
                    "Increasing delay from %.1fs to %.1fs due to rate limiting",
                    old_delay,
                    current_delay,
                )

            processed += len(batch)

            # Report progress
            if progress_callback:
                progress_callback(processed, total, len(batch))

            # Rate limiting delay between batches (except for last batch)
            if batch_idx < len(batches) - 1 and current_delay > 0:
                await asyncio.sleep(current_delay)

        logger.info("Completed embedding %d texts", total)
        return all_embeddings


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Create embedder from configuration.

    Args:
        config: Embedding configuration

    Returns:
        Embedder instance

    Supported providers:
        - "local": FastEmbed local model (requires fastembed)
        - "openai": OpenAI API (requires openai)
        - "gemini" or "vertex": Google Vertex AI Gemini (requires google-genai)
    """
    if config.provider == "local":
        return LocalEmbedder(
            model=config.model,
            dimension=config.dimension,
        )
    elif config.provider in ("gemini", "vertex"):
        return GeminiEmbedder(
            model=config.model,
            dimension=config.dimension,
        )
    else:
        return APIEmbedder(
            provider=config.provider,
            model=config.model,
            dimension=config.dimension,
        )
