"""Embedding providers for memory vector operations.

 introduces a dual-track stack that supersedes the original
LocalEmbeddingProvider-only design:

- SiliconFlowEmbeddingProvider — remote OpenAI-compatible endpoint
 (BAAI/bge-m3 by default). Primary path for benchmarks where API access
 is available.
- SentenceTransformerEmbeddingProvider — local sentence-transformers
 backend (BAAI/bge-small-en-v1.5 by default). Offline fallback / CI path.
- NoOpEmbeddingProvider — deterministic hash-based pseudo-embeddings
 used by unit tests; never enters production hot paths.

Provider selection is driven by the project-wide EMBEDDING_PROVIDER /
EMBEDDING_MODEL env vars (see houyi.infrastructure.config.env_config).
Use make_embedding_provider for default construction; tests should
inject providers directly via dependency injection.

All providers expose the same async embed(texts) + dimension() shape
defined by EmbeddingProvider so callers stay agnostic to the
underlying backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# NoOp (testing / fallback)
# ---------------------------------------------------------------------------


class NoOpEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based pseudo-embeddings for testing and fallback.

    Produces consistent vectors from text content so that identical texts
    yield identical embeddings, but quality is far below a real model.
    Never use this in production retrieval paths.
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


# ---------------------------------------------------------------------------
# SiliconFlow (remote, OpenAI-compatible)
# ---------------------------------------------------------------------------


DEFAULT_SILICONFLOW_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_SILICONFLOW_DIMENSION = 1024  # BGE-M3 native dimension


class SiliconFlowEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding client targeting SiliconFlow.

    The endpoint is POST {base_url}/embeddings with payload
    {"model": str, "input": list[str]}. Failures (network, 5xx, invalid
    JSON) raise EmbeddingProviderError so callers can decide whether
    to fall back to a lexical-only path or degrade to NoOpEmbeddingProvider.

    No silent degradation: degradation policy lives one layer above
    (orchestrator / backfill worker), per Embedding Backfill.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_SILICONFLOW_EMBEDDING_MODEL,
        base_url: str | None = None,
        dimension: int = DEFAULT_SILICONFLOW_DIMENSION,
        timeout_s: float = 10.0,
        max_batch: int = 32,
    ):
        if not api_key:
            raise ValueError("SiliconFlowEmbeddingProvider requires a non-empty api_key")
        from houyi.infrastructure.config.env_config import EnvConfig

        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or EnvConfig.get().siliconflow_base_url).rstrip("/")
        self._dim = dimension
        self._timeout_s = timeout_s
        self._max_batch = max_batch

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Lazy import keeps httpx out of the import graph for pure-stub tests.
        import httpx

        out: list[list[float]] = []
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            for start in range(0, len(texts), self._max_batch):
                batch = texts[start : start + self._max_batch]
                payload: dict[str, Any] = {"model": self._model, "input": batch}
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise EmbeddingProviderError(
                        f"SiliconFlow embedding request failed: {exc}"
                    ) from exc

                items = data.get("data") or []
                if len(items) != len(batch):
                    raise EmbeddingProviderError(
                        f"SiliconFlow returned {len(items)} embeddings for {len(batch)} inputs"
                    )
                for item in items:
                    vec = item.get("embedding")
                    if not isinstance(vec, list) or not vec:
                        raise EmbeddingProviderError("SiliconFlow embedding payload malformed")
                    out.append([float(x) for x in vec])
        return out


# ---------------------------------------------------------------------------
# Local (sentence-transformers)
# ---------------------------------------------------------------------------


DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers backend.

    The heavy sentence_transformers import is guarded so unit tests that
    do not need real embeddings can still import this module. encode runs
    in a worker thread to avoid blocking the event loop.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        device: str | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only on bare envs
            raise EmbeddingProviderError(
                "sentence-transformers is not installed; install it or switch "
                "EMBEDDING_PROVIDER to 'siliconflow' / 'noop'."
            ) from exc

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        # get_sentence_embedding_dimension is stable across recent ST releases.
        self._dim: int = int(self._model.get_sentence_embedding_dimension())

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def _run() -> list[list[float]]:
            vecs = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [vec.tolist() for vec in vecs]

        return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Errors + factory
# ---------------------------------------------------------------------------


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding backend fails irrecoverably for a single call."""


_PROVIDER_SILICONFLOW = "siliconflow"
_PROVIDER_LOCAL = "local"
_PROVIDER_NOOP = "noop"


def make_embedding_provider(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> EmbeddingProvider:
    """Build an EmbeddingProvider from env / explicit overrides.

    Resolution order:

    1. Explicit provider argument.
    2. EMBEDDING_PROVIDER env var (project-wide default "local").

    Supported values: "siliconflow", "local", "noop".

    Notes:
    - "siliconflow" requires SILICONFLOW_API_KEY (or api_key
    override). If the key is missing we raise rather than silently
    downgrading; orchestrator-level fallback lives elsewhere.
    - "local" requires sentence-transformers to be importable.
    - "noop" is intended for tests and stubs only.
    """
    provider = (
        (provider or os.getenv("EMBEDDING_PROVIDER", _PROVIDER_LOCAL) or _PROVIDER_LOCAL)
        .strip()
        .lower()
    )

    if provider == _PROVIDER_NOOP:
        return NoOpEmbeddingProvider()
    if provider == _PROVIDER_SILICONFLOW:
        from houyi.infrastructure.config.env_config import EnvConfig

        _env = EnvConfig.get()
        key = api_key or _env.siliconflow_api_key
        if not key:
            raise EmbeddingProviderError(
                "EMBEDDING_PROVIDER='siliconflow' but SILICONFLOW_API_KEY is unset."
            )
        return SiliconFlowEmbeddingProvider(
            api_key=key,
            model=model or _env.embedding_model or DEFAULT_SILICONFLOW_EMBEDDING_MODEL,
        )
    if provider == _PROVIDER_LOCAL:
        return SentenceTransformerEmbeddingProvider(
            model_name=model or os.getenv("EMBEDDING_MODEL") or DEFAULT_LOCAL_EMBEDDING_MODEL,
        )
    raise EmbeddingProviderError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}; expected siliconflow / local / noop."
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)
