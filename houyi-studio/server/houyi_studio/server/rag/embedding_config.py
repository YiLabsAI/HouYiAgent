"""Embedding configuration resolution and storage-path helpers.

Responsibilities:
    - Define storage-directory constants and path helpers used across RAG modules.
    - Resolve the embedding provider/model/dimension via a clear priority chain:
      explicit override → environment variables → auto-detection.

Dependencies:
    - houyi.infrastructure.config.env_config for environment variable names.
    - houyi.rag.config.EmbeddingConfig (lazy import inside helpers).

Thread Safety:
    All functions are stateless and safe to call from any thread/task.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from houyi.infrastructure.config import (
    ENV_EMBEDDING_MODEL,
    ENV_EMBEDDING_PROVIDER,
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_APPLICATION_CREDENTIALS,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_KNOWLEDGE_STORAGE,
    ENV_OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage-directory constants
# ---------------------------------------------------------------------------

UPLOADS_SUBDIR = "uploads"
INDEX_SUBDIR = "index"

_LIB_ID_PREFIX = "lib_"

KNOWLEDGE_STORAGE_DIR: Path = Path(os.getenv(ENV_KNOWLEDGE_STORAGE, ".houyi/knowledge")).resolve()


def _default_storage_dir() -> Path:
    """Lazily resolve the default storage directory.

    Returns the path at call time so that tests can set
    HOUYI_KNOWLEDGE_STORAGE *before* importing.
    """
    return Path(os.getenv(ENV_KNOWLEDGE_STORAGE, ".houyi/knowledge")).resolve()


def get_library_storage_dir(library_id: str) -> Path:
    """Return the root storage directory for *library_id*.

    For test isolation prefer the instance methods on
    ~.library_repository.LibraryRepository.
    """
    return _default_storage_dir() / library_id


def get_library_upload_dir(library_id: str) -> Path:
    """Return the uploads sub-directory for *library_id*."""
    return get_library_storage_dir(library_id) / UPLOADS_SUBDIR


def get_library_index_dir(library_id: str) -> Path:
    """Return the index sub-directory for *library_id*."""
    return get_library_storage_dir(library_id) / INDEX_SUBDIR


def is_index_path(path: Path) -> bool:
    """Check whether *path* is an internal index file that should be skipped.

    Index files live under {storage}/<lib>/index/.  Upload files under
    {storage}/<lib>/uploads/ are **not** skipped.

    Args:
        path: Filesystem path to inspect.

    Returns:
        True when the path belongs to an index directory inside
        .houyi.
    """
    path_str = str(path)
    index_unix = f"/{INDEX_SUBDIR}/"
    index_win = f"\\{INDEX_SUBDIR}\\"
    has_index = index_unix in path_str or index_win in path_str
    return has_index and ".houyi" in path_str


# ---------------------------------------------------------------------------
# Embedding provider resolution
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, tuple[str, int]] = {
    "local": ("BAAI/bge-small-en-v1.5", 384),
    "gemini": ("text-embedding-004", 768),
    "vertex": ("text-embedding-004", 768),
    "openai": ("text-embedding-3-small", 1536),
}


def _is_provider_runtime_available(provider: str) -> bool:
    """Return whether runtime deps for *provider* are importable."""
    if provider == "local":
        try:
            import fastembed  # noqa: F401

            return True
        except ImportError:
            return False
    if provider in ("gemini", "vertex"):
        try:
            from google import genai  # noqa: F401

            return True
        except ImportError:
            return False
    if provider == "openai":
        try:
            import openai  # noqa: F401

            return True
        except ImportError:
            return False
    return True


def _raise_or_warn_provider_unavailable(provider: str, source: str, strict_explicit: bool) -> None:
    message = f"Embedding provider '{provider}' from {source} is unavailable at runtime"
    if strict_explicit:
        raise RuntimeError(message)
    logger.warning("%s; falling back to auto-detect", message)


def _make_embedding_config(
    provider: str,
    model: str | None = None,
    dimension: int | None = None,
) -> Any:
    """Create an EmbeddingConfig using provider defaults when needed.

    Args:
        provider: Embedding provider name (local, gemini, etc.).
        model: Override model identifier.
        dimension: Override vector dimension.

    Returns:
        A houyi.rag.config.EmbeddingConfig instance.
    """
    from houyi.rag.config import EmbeddingConfig

    defaults = _PROVIDER_DEFAULTS.get(provider, ("text-embedding-3-small", 1536))
    return EmbeddingConfig(
        provider=provider,
        model=model or defaults[0],
        dimension=dimension or defaults[1],
    )


def resolve_embedding_config(
    *,
    preferred_provider: str | None = None,
    preferred_model: str | None = None,
    preferred_dimension: int | None = None,
    strict_explicit: bool = False,
) -> tuple[Any, str] | tuple[None, str]:
    """Resolve the embedding config using a clear priority chain.

    Priority (highest to lowest):
        1. **Explicit override** — preferred_provider (from library metadata / UI).
        2. **Environment variables** EMBEDDING_PROVIDER / EMBEDDING_MODEL.
        3. **Auto-detection** (Gemini → OpenAI → local fastembed).

    Args:
        preferred_provider: Explicit provider name from library settings.
        preferred_model: Explicit model override.
        preferred_dimension: Explicit dimension override.

    Returns:
        (EmbeddingConfig, provider_name) or (None, "no_provider").
    """
    if preferred_provider:
        if _is_provider_runtime_available(preferred_provider):
            cfg = _make_embedding_config(
                preferred_provider,
                preferred_model,
                preferred_dimension,
            )
            logger.debug("Embedding: using explicit provider '%s'", preferred_provider)
            return cfg, preferred_provider
        _raise_or_warn_provider_unavailable(
            preferred_provider, "preferred provider", strict_explicit
        )

    env_provider = os.environ.get(ENV_EMBEDDING_PROVIDER)
    env_model = os.environ.get(ENV_EMBEDDING_MODEL)
    if env_provider:
        if _is_provider_runtime_available(env_provider):
            cfg = _make_embedding_config(env_provider, env_model)
            logger.debug("Embedding: using env var provider '%s'", env_provider)
            return cfg, env_provider
        _raise_or_warn_provider_unavailable(env_provider, "env", strict_explicit)

    return _auto_detect_embedding()


def _auto_detect_embedding() -> tuple[Any, str] | tuple[None, str]:
    """Auto-detect the best available embedding provider.

    Detection order: Gemini (API-key / Vertex) → OpenAI → local fastembed.

    Returns:
        (EmbeddingConfig, provider_name) or (None, "no_provider").
    """
    if os.environ.get(ENV_GOOGLE_API_KEY):
        try:
            from google import genai

            cfg = _make_embedding_config("gemini")
            logger.debug("Embedding auto-detect: gemini (GOOGLE_API_KEY)")
            return cfg, "gemini"
        except ImportError:
            logger.warning(
                "Embedding auto-detect: GOOGLE_API_KEY is set but google-genai is unavailable; skipping Gemini"
            )

    google_project = os.environ.get(ENV_GOOGLE_CLOUD_PROJECT)
    if not google_project:
        creds_file = os.environ.get(ENV_GOOGLE_APPLICATION_CREDENTIALS, "")
        if creds_file:
            try:
                with open(creds_file) as f:
                    google_project = json.load(f).get("project_id", "")
            except Exception:
                pass
    if google_project:
        try:
            from google import genai  # noqa: F401

            cfg = _make_embedding_config("gemini")
            logger.debug("Embedding auto-detect: gemini (GOOGLE_CLOUD_PROJECT)")
            return cfg, "gemini"
        except ImportError:
            logger.warning(
                "Embedding auto-detect: GOOGLE_CLOUD_PROJECT is set but google-genai is unavailable; skipping Vertex/Gemini"
            )

    if os.environ.get(ENV_OPENAI_API_KEY):
        if _is_provider_runtime_available("openai"):
            cfg = _make_embedding_config("openai")
            logger.debug("Embedding auto-detect: openai")
            return cfg, "openai"
        logger.warning(
            "Embedding auto-detect: OPENAI_API_KEY is set but openai package is unavailable; skipping OpenAI"
        )

    try:
        import fastembed  # noqa: F401

        cfg = _make_embedding_config("local")
        logger.debug("Embedding auto-detect: local (fastembed)")
        return cfg, "local"
    except ImportError:
        logger.warning("Embedding auto-detect: fastembed is unavailable; local embedding disabled")

    return None, "no_provider"


def _detect_embedding_config() -> tuple[Any, str] | tuple[None, str]:
    """Backward-compatible alias — prefer resolve_embedding_config."""
    return resolve_embedding_config()
