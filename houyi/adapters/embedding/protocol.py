"""Embedding provider protocol and shared utilities."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Protocol for batch text embedding generation.

    All implementations must be asynchronous to keep the call-site
    uniform regardless of whether the backend is a remote HTTP API
    (httpx-based) or a local model (sentence-transformers running on
    a worker thread).
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding backend fails irrecoverably for a single call."""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 when the vectors have different lengths or one of them
    has zero magnitude. Inputs are not modified.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


__all__ = ["EmbeddingProvider", "EmbeddingProviderError", "cosine_similarity"]
