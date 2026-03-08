from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from houyi.infrastructure.config.env_config import (
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_CLOUD_LOCATION,
    ENV_GOOGLE_CLOUD_PROJECT,
)
from houyi.rag.indexed.embedding.base import BaseEmbedder, ProgressCallback

logger = logging.getLogger(__name__)


class GeminiEmbedder(BaseEmbedder):
    """Embedder using Google Gemini with rate-limit-aware batching."""

    DEFAULT_BATCH_SIZE = 2
    DEFAULT_DELAY_SECONDS = 2.0
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 2.0
    MAX_BACKOFF = 60.0
    DELAY_MULTIPLIER = 2.0

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
        self._client: Any = None
        self._project = project
        self._location = location
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.delay_seconds = delay_seconds or self.DEFAULT_DELAY_SECONDS

    def _ensure_client(self) -> None:
        """Initialize the Gemini client using API key or Vertex AI auth."""
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

        api_key = os.getenv(ENV_GOOGLE_API_KEY)
        if api_key:
            self._client = genai.Client(api_key=api_key)
            logger.info("GeminiEmbedder: using GOOGLE_API_KEY auth")
            return

        project = self._project or os.getenv(ENV_GOOGLE_CLOUD_PROJECT)
        location = self._location or os.getenv(ENV_GOOGLE_CLOUD_LOCATION) or "us-central1"

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
        """Embed a single text using Gemini with retry logic."""
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
            except Exception as exc:
                error_str = str(exc).lower()
                is_rate_limit = (
                    "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str
                )

                if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                    jitter = random.uniform(0, 0.5 * backoff)
                    wait_time = backoff + jitter
                    logger.warning(
                        "Rate limit hit (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        wait_time,
                        exc,
                    )
                    await asyncio.sleep(wait_time)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                else:
                    raise

        raise RuntimeError("Max retries exceeded for embedding")

    async def _embed_batch_single(self, batch: list[str]) -> tuple[list[list[float]], bool]:
        """Embed a single batch with retry logic and rate-limit signaling."""
        self._ensure_client()

        backoff = self.INITIAL_BACKOFF
        rate_limited = False

        for attempt in range(self.MAX_RETRIES):
            try:
                result = await self._client.aio.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                return [list(embedding.values) for embedding in result.embeddings], rate_limited
            except Exception as exc:
                error_str = str(exc).lower()
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
                        exc,
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
        """Batch embed with adaptive rate limiting and progress reporting."""
        self._ensure_client()

        all_embeddings: list[list[float]] = []
        total = len(texts)
        processed = 0
        current_delay = self.delay_seconds
        batches = [
            texts[index : index + self.batch_size] for index in range(0, total, self.batch_size)
        ]

        logger.info(
            "Embedding %d texts in %d batches (batch_size=%d, delay=%.1fs)",
            total,
            len(batches),
            self.batch_size,
            current_delay,
        )

        for batch_idx, batch in enumerate(batches):
            batch_embeddings, rate_limited = await self._embed_batch_single(batch)
            all_embeddings.extend(batch_embeddings)

            if rate_limited:
                old_delay = current_delay
                current_delay = min(current_delay * self.DELAY_MULTIPLIER, 10.0)
                logger.info(
                    "Increasing delay from %.1fs to %.1fs due to rate limiting",
                    old_delay,
                    current_delay,
                )

            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total, len(batch))

            if batch_idx < len(batches) - 1 and current_delay > 0:
                await asyncio.sleep(current_delay)

        logger.info("Completed embedding %d texts", total)
        return all_embeddings
