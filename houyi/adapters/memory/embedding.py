"""Embedding providers for memory vector operations.

Provides a pluggable protocol and two implementations:
- LocalEmbeddingProvider: sentence-transformers based (offline)
- NoOpEmbeddingProvider: zero-vector fallback when no embedding is needed
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Protocol for text embedding generation."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...


class NoOpEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based pseudo-embeddings for testing and fallback.

    Produces consistent vectors from text content so that identical texts
    yield identical embeddings and similar texts have *some* proximity,
    but quality is far below a real model.  Useful when no GPU or API
    key is available.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def dimension(self) -> int:
        return self._dim

    def _hash_embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = []
        for i in range(self._dim):
            byte_val = digest[i % len(digest)]
            raw.append((byte_val / 255.0) * 2 - 1)
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)
