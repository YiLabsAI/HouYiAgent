"""Embedding provider factory.

make_embedding_provider is the canonical entry point for picking
an EmbeddingProvider from the project-wide EMBEDDING_PROVIDER
env var or an explicit override.
"""

from __future__ import annotations

import os

from houyi.adapters.embedding.protocol import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from houyi.adapters.embedding.providers import (
    DashScopeEmbeddingProvider,
    NoOpEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    SiliconFlowEmbeddingProvider,
)


# Re-exported for callers (and tests) that need to assert against the
# default SiliconFlow embedding model. Sourced from the canonical
# env_config defaults so there is exactly one source of truth.
def _siliconflow_default_model() -> str:
    from houyi.infrastructure.config.env_config import (
        _DEFAULT_SILICONFLOW_EMBEDDING_MODEL,
    )

    return _DEFAULT_SILICONFLOW_EMBEDDING_MODEL


DEFAULT_SILICONFLOW_EMBEDDING_MODEL: str = _siliconflow_default_model()


_PROVIDER_SILICONFLOW = "siliconflow"
_PROVIDER_DASHSCOPE = "dashscope"
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
    2. EMBEDDING_PROVIDER env var (project-wide default local).

    Supported values: siliconflow / dashscope / local / noop.

    Notes:
        - siliconflow requires SILICONFLOW_API_KEY (or the api_key
          override). If the key is missing we raise rather than
          silently downgrading; orchestrator-level fallback lives
          elsewhere.
        - dashscope requires DASHSCOPE_API_KEY.
        - local requires sentence-transformers to be importable.
        - noop is intended for tests and stubs only.
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
            model=model or _env.embedding_model,
        )
    if provider == _PROVIDER_DASHSCOPE:
        from houyi.infrastructure.config.env_config import EnvConfig

        _env = EnvConfig.get()
        key = api_key or _env.dashscope_api_key
        if not key:
            raise EmbeddingProviderError(
                "EMBEDDING_PROVIDER='dashscope' but DASHSCOPE_API_KEY is unset."
            )
        return DashScopeEmbeddingProvider(
            api_key=key,
            model=model or _env.dashscope_embedding_model,
        )
    if provider == _PROVIDER_LOCAL:
        from houyi.infrastructure.config.env_config import EnvConfig

        _env = EnvConfig.get()
        return SentenceTransformerEmbeddingProvider(
            model_name=model or _env.embedding_model,
        )
    raise EmbeddingProviderError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}; expected siliconflow / dashscope / local / noop."
    )


__all__ = ["DEFAULT_SILICONFLOW_EMBEDDING_MODEL", "make_embedding_provider"]
