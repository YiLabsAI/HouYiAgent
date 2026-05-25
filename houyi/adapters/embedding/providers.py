"""Concrete embedding provider implementations.

Four backends share the EmbeddingProvider protocol:

- NoOpEmbeddingProvider — deterministic hash-based pseudo embeddings
  for tests; never enters production hot paths.
- SiliconFlowEmbeddingProvider — remote OpenAI-compatible endpoint,
  BAAI/bge-m3 by default. Primary path for benchmarks and production
  where API access is available.
- DashScopeEmbeddingProvider — Alibaba Cloud Bailian remote endpoint
  compatible with OpenAI /embeddings shape.
- SentenceTransformerEmbeddingProvider — local sentence-transformers
  backend (BAAI/bge-small-en-v1.5 by default). Offline fallback / CI
  path. The heavy sentence_transformers import is guarded so unit
  tests that do not need real embeddings can still import this module.

Configuration is read from EnvConfig (houyi.infrastructure.config.env_config)
at runtime to avoid hardcoded defaults in this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import Any

from houyi.adapters.embedding.protocol import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

logger = logging.getLogger(__name__)


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


class SiliconFlowEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding client targeting SiliconFlow.

    The endpoint is POST {base_url}/embeddings with payload
    {"model": str, "input": list[str]}. Failures (network, 5xx, invalid
    JSON) raise EmbeddingProviderError so callers can decide whether
    to fall back to a lexical-only path or degrade to NoOpEmbeddingProvider.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
        timeout_s: float = 10.0,
        max_batch: int = 32,
    ):
        if not api_key:
            raise ValueError("SiliconFlowEmbeddingProvider requires a non-empty api_key")
        from houyi.infrastructure.config.env_config import (
            _DEFAULT_SILICONFLOW_EMBEDDING_MODEL,
            EnvConfig,
        )

        _env = EnvConfig.get()
        self._api_key = api_key
        # Each provider uses its own default model; generic EMBEDDING_MODEL is for local provider only.
        self._model = model or _DEFAULT_SILICONFLOW_EMBEDDING_MODEL
        self._base_url = (base_url or _env.siliconflow_base_url).rstrip("/")
        self._dim = dimension or 1024  # BGE-M3 native dimension
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
                except httpx.HTTPStatusError as exc:
                    err_body = exc.response.text[:500] if exc.response else ""
                    raise EmbeddingProviderError(
                        f"SiliconFlow embedding request failed: {exc} - {err_body}"
                    ) from exc
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
# DashScope (Bailian, Alibaba Cloud)
# ---------------------------------------------------------------------------


class DashScopeEmbeddingProvider(EmbeddingProvider):
    """DashScope embedding client for Alibaba Cloud Bailian service.

    Uses the DashScope HTTP API endpoint compatible with OpenAI format.
    Endpoint: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
        timeout_s: float = 30.0,
        max_batch: int = 25,  # DashScope batch limit
    ):
        if not api_key:
            raise ValueError("DashScopeEmbeddingProvider requires a non-empty api_key")
        from houyi.infrastructure.config.env_config import EnvConfig

        _env = EnvConfig.get()
        self._api_key = api_key
        self._model = model or _env.dashscope_embedding_model
        self._base_url = (base_url or _env.dashscope_base_url).rstrip("/")
        self._dim = dimension or 1024  # text-embedding-v3 dimension
        self._timeout_s = timeout_s
        self._max_batch = max_batch

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
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
                payload: dict[str, Any] = {
                    "model": self._model,
                    "input": batch,
                }
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as exc:
                    # Log actual error response for debugging
                    err_body = exc.response.text[:500] if exc.response else ""
                    logger.debug("DashScope embedding error: %s", err_body)
                    raise EmbeddingProviderError(
                        f"DashScope embedding request failed: {exc} - {err_body}"
                    ) from exc
                except (httpx.HTTPError, ValueError) as exc:
                    raise EmbeddingProviderError(
                        f"DashScope embedding request failed: {exc}"
                    ) from exc

                items = data.get("data") or []
                if len(items) != len(batch):
                    raise EmbeddingProviderError(
                        f"DashScope returned {len(items)} embeddings for {len(batch)} inputs"
                    )
                # Sort by index to maintain order
                items_sorted = sorted(items, key=lambda x: x.get("index", 0))
                for item in items_sorted:
                    vec = item.get("embedding")
                    if not isinstance(vec, list) or not vec:
                        raise EmbeddingProviderError("DashScope embedding payload malformed")
                    out.append([float(x) for x in vec])
        return out


# ---------------------------------------------------------------------------
# Local (sentence-transformers)
# ---------------------------------------------------------------------------


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers backend.

    The heavy sentence_transformers import is guarded so unit tests that
    do not need real embeddings can still import this module. encode runs
    in a worker thread to avoid blocking the event loop.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
    ):
        from houyi.infrastructure.config.env_config import EnvConfig

        _env = EnvConfig.get()
        _model_name = model_name or _env.embedding_model or "BAAI/bge-small-en-v1.5"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only on bare envs
            raise EmbeddingProviderError(
                "sentence-transformers is not installed; install it or switch "
                "EMBEDDING_PROVIDER to 'siliconflow' / 'noop'."
            ) from exc

        self._model_name = _model_name
        # Try cache-only first to suppress the HF Hub HEAD-check spam on every
        # startup. Fall back to network on cache miss so the first run still
        # downloads the model.
        try:
            self._model = SentenceTransformer(_model_name, device=device, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(_model_name, device=device)
        self._dim: int = int(self._model.get_embedding_dimension())

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


__all__ = [
    "DashScopeEmbeddingProvider",
    "NoOpEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SiliconFlowEmbeddingProvider",
]
