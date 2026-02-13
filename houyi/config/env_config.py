"""Centralized environment variable configuration for Houyi.

Design principles:
- Single Responsibility: only reads env vars, provides typed defaults, logs warnings.
- Decoupled from models.py: uses model constants for defaults but owns all env logic.
- Singleton + snapshot: reads env once at construction; call reload() to re-read.
- Fail-soft: missing vars produce warnings, never raise (unless explicitly noted).

Industry references:
- LangChain get_from_dict_or_env() pattern
- litellm module-level config (litellm.api_key)
- pydantic-settings BaseSettings (we avoid the dependency but follow the spirit)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: env var name registry (single source of truth)
# ---------------------------------------------------------------------------
# These are intentionally private — consumers should use EnvConfig properties,
# not reference env var names directly.

_ENV_SILICONFLOW_API_KEY = "SILICONFLOW_API_KEY"
_ENV_SILICONFLOW_BASE_URL = "SILICONFLOW_BASE_URL"
_ENV_DEEPSEEK_MODEL = "DEEPSEEK_MODEL"
_ENV_DEFAULT_LLM_PROVIDER = "DEFAULT_LLM_PROVIDER"

_ENV_GEMINI_MODEL = "GEMINI_MODEL"
_ENV_GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"
_ENV_GOOGLE_PROJECT_ID = "GOOGLE_PROJECT_ID"
_ENV_GOOGLE_LOCATION = "GOOGLE_LOCATION"
# Vertex AI adapter also reads these alternate names
_ENV_VERTEX_PROJECT = "VERTEX_PROJECT"
_ENV_VERTEX_LOCATION = "VERTEX_LOCATION"
_ENV_VERTEX_GEMINI_MODEL = "VERTEX_GEMINI_MODEL"
_ENV_GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
_ENV_GOOGLE_CLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION"

_ENV_RAG_KNOWLEDGE_DIR = "RAG_KNOWLEDGE_DIR"
_ENV_RAG_EMBEDDING_PROVIDER = "RAG_EMBEDDING_PROVIDER"
_ENV_RAG_EMBEDDING_MODEL = "RAG_EMBEDDING_MODEL"

# ---------------------------------------------------------------------------
# Defaults (kept here, not in models.py — these are config concerns)
# ---------------------------------------------------------------------------
_DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_VERTEX_LOCATION = "us-central1"
_DEFAULT_RAG_KNOWLEDGE_DIR = "knowledge/"
_DEFAULT_RAG_EMBEDDING_PROVIDER = "openai"
_DEFAULT_RAG_EMBEDDING_MODEL = "text-embedding-3-small"


class EnvConfig:
    """Centralized environment variable configuration.

    Reads from ``os.environ`` with typed defaults.  Logs warnings when
    critical variables are missing.  Thread-safe singleton.

    Usage::

        from houyi.config import env

        key = env.siliconflow_api_key       # SILICONFLOW_API_KEY or None
        url = env.siliconflow_base_url      # SILICONFLOW_BASE_URL or default
        kdir = env.rag_knowledge_dir        # RAG_KNOWLEDGE_DIR or "knowledge/"
    """

    _instance: EnvConfig | None = None
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls) -> EnvConfig:
        # Allow direct construction for testing; singleton via get()
        return super().__new__(cls)

    def __init__(self) -> None:
        # Attributes are initialized here to satisfy static analyzers.
        # Actual values are loaded by _load().
        self._siliconflow_api_key: str | None = None
        self._siliconflow_base_url: str = _DEFAULT_SILICONFLOW_BASE_URL
        self._deepseek_model: str = ""
        self._default_llm_provider: str = ""

        self._gemini_model: str = ""
        self._google_credentials_path: str | None = None
        self._google_project_id: str | None = None
        self._google_location: str = _DEFAULT_VERTEX_LOCATION

        self._rag_knowledge_dir: str = _DEFAULT_RAG_KNOWLEDGE_DIR
        self._rag_embedding_provider: str = _DEFAULT_RAG_EMBEDDING_PROVIDER
        self._rag_embedding_model: str = _DEFAULT_RAG_EMBEDDING_MODEL

    @classmethod
    def get(cls) -> EnvConfig:
        """Get or create the global singleton instance.

        Thread-safe.  First call snapshots current ``os.environ``.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst._load()
                    cls._instance = inst
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton (for testing only)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Snapshot current os.environ into typed attributes."""
        from houyi.llm.models import DEFAULT_MODEL, GEMINI_25_PRO, PROVIDER_SILICONFLOW

        # --- LLM Provider ---
        self._siliconflow_api_key = os.getenv(_ENV_SILICONFLOW_API_KEY)
        self._siliconflow_base_url = os.getenv(
            _ENV_SILICONFLOW_BASE_URL, _DEFAULT_SILICONFLOW_BASE_URL
        )
        self._deepseek_model = os.getenv(_ENV_DEEPSEEK_MODEL, DEFAULT_MODEL)
        self._default_llm_provider = os.getenv(_ENV_DEFAULT_LLM_PROVIDER, PROVIDER_SILICONFLOW)

        # --- Google / Vertex AI ---
        self._gemini_model = (
            os.getenv(_ENV_VERTEX_GEMINI_MODEL) or os.getenv(_ENV_GEMINI_MODEL) or GEMINI_25_PRO
        )
        self._google_credentials_path = os.getenv(_ENV_GOOGLE_APPLICATION_CREDENTIALS)
        self._google_project_id = (
            os.getenv(_ENV_VERTEX_PROJECT)
            or os.getenv(_ENV_GOOGLE_PROJECT_ID)
            or os.getenv(_ENV_GOOGLE_CLOUD_PROJECT)
        )
        self._google_location = (
            os.getenv(_ENV_VERTEX_LOCATION)
            or os.getenv(_ENV_GOOGLE_LOCATION)
            or os.getenv(_ENV_GOOGLE_CLOUD_LOCATION)
            or _DEFAULT_VERTEX_LOCATION
        )

        # --- RAG ---
        self._rag_knowledge_dir = os.getenv(_ENV_RAG_KNOWLEDGE_DIR, _DEFAULT_RAG_KNOWLEDGE_DIR)
        self._rag_embedding_provider = os.getenv(
            _ENV_RAG_EMBEDDING_PROVIDER, _DEFAULT_RAG_EMBEDDING_PROVIDER
        )
        self._rag_embedding_model = os.getenv(
            _ENV_RAG_EMBEDDING_MODEL, _DEFAULT_RAG_EMBEDDING_MODEL
        )

        # --- Warnings for missing critical vars ---
        if not self._siliconflow_api_key:
            logger.warning(
                "%s not set — SiliconFlow adapter will use mock responses",
                _ENV_SILICONFLOW_API_KEY,
            )
        if not self._google_credentials_path:
            logger.debug(
                "%s not set — Vertex AI adapter will use mock responses",
                _ENV_GOOGLE_APPLICATION_CREDENTIALS,
            )

        logger.info(
            "EnvConfig loaded: provider=%s, rag_knowledge_dir=%s",
            self._default_llm_provider,
            self._rag_knowledge_dir,
        )

    def reload(self) -> None:
        """Re-read all environment variables.

        Useful in tests after patching ``os.environ``.
        """
        self._load()

    # ------------------------------------------------------------------
    # LLM Provider properties
    # ------------------------------------------------------------------

    @property
    def siliconflow_api_key(self) -> str | None:
        """SILICONFLOW_API_KEY or None."""
        return self._siliconflow_api_key

    @property
    def siliconflow_base_url(self) -> str:
        """SILICONFLOW_BASE_URL or default ``https://api.siliconflow.cn/v1``."""
        return self._siliconflow_base_url

    @property
    def deepseek_model(self) -> str:
        """DEEPSEEK_MODEL or default from models.DEFAULT_MODEL."""
        return self._deepseek_model

    @property
    def default_llm_provider(self) -> str:
        """DEFAULT_LLM_PROVIDER or ``"siliconflow"``."""
        return self._default_llm_provider

    # ------------------------------------------------------------------
    # Google / Vertex AI properties
    # ------------------------------------------------------------------

    @property
    def gemini_model(self) -> str:
        """VERTEX_GEMINI_MODEL / GEMINI_MODEL or default ``"gemini-2.5-pro"``."""
        return self._gemini_model

    @property
    def google_credentials_path(self) -> str | None:
        """GOOGLE_APPLICATION_CREDENTIALS or None."""
        return self._google_credentials_path

    @property
    def google_project_id(self) -> str | None:
        """VERTEX_PROJECT / GOOGLE_PROJECT_ID / GOOGLE_CLOUD_PROJECT or None."""
        return self._google_project_id

    @property
    def google_location(self) -> str:
        """VERTEX_LOCATION / GOOGLE_LOCATION / GOOGLE_CLOUD_LOCATION or ``"us-central1"``."""
        return self._google_location

    # ------------------------------------------------------------------
    # RAG properties
    # ------------------------------------------------------------------

    @property
    def rag_knowledge_dir(self) -> str:
        """RAG_KNOWLEDGE_DIR or default ``"knowledge/"``."""
        return self._rag_knowledge_dir

    @property
    def rag_embedding_provider(self) -> str:
        """RAG_EMBEDDING_PROVIDER or default ``"openai"``."""
        return self._rag_embedding_provider

    @property
    def rag_embedding_model(self) -> str:
        """RAG_EMBEDDING_MODEL or default ``"text-embedding-3-small"``."""
        return self._rag_embedding_model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a sanitized config summary for logging.

        API keys are masked to avoid leaking secrets.
        """

        def _mask(val: str | None) -> str:
            if not val:
                return "(not set)"
            if len(val) <= 8:
                return "****"
            return val[:4] + "****" + val[-4:]

        return {
            "siliconflow_api_key": _mask(self._siliconflow_api_key),
            "siliconflow_base_url": self._siliconflow_base_url,
            "deepseek_model": self._deepseek_model,
            "default_llm_provider": self._default_llm_provider,
            "gemini_model": self._gemini_model,
            "google_credentials_path": self._google_credentials_path or "(not set)",
            "google_project_id": self._google_project_id or "(not set)",
            "google_location": self._google_location,
            "rag_knowledge_dir": self._rag_knowledge_dir,
            "rag_embedding_provider": self._rag_embedding_provider,
            "rag_embedding_model": self._rag_embedding_model,
        }

    def __repr__(self) -> str:
        return f"<EnvConfig provider={self._default_llm_provider!r} rag_dir={self._rag_knowledge_dir!r}>"
