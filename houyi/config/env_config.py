"""Centralized environment variable configuration for Houyi.

Design principles:
- Single Responsibility: only reads env vars, provides typed defaults, logs warnings.
- Decoupled from models.py: uses model constants for defaults but owns all env logic.
- Singleton + snapshot: reads env once at construction; call reload() to re-read.
- Fail-soft: missing vars produce warnings, never raise (unless explicitly noted).

Env var naming follows upstream SDK conventions:
- google-genai SDK: GOOGLE_API_KEY, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
- openai SDK: OPENAI_API_KEY
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env var name registry (single source of truth)
#
# Public constants (ENV_*) are importable by other modules so that env var
# names never appear as raw strings elsewhere in the codebase.
# ---------------------------------------------------------------------------

# SiliconFlow / DeepSeek
ENV_SILICONFLOW_API_KEY = "SILICONFLOW_API_KEY"
ENV_SILICONFLOW_BASE_URL = "SILICONFLOW_BASE_URL"
ENV_DEEPSEEK_MODEL = "DEEPSEEK_MODEL"
ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
ENV_DEEPSEEK_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_DEFAULT_LLM_PROVIDER = "DEFAULT_LLM_PROVIDER"

# OpenAI
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_ORG = "OPENAI_ORG"

# Anthropic
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"

# Google (follows google-genai SDK naming)
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"
ENV_GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
ENV_GOOGLE_CLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION"
ENV_GEMINI_MODEL = "GEMINI_MODEL"

# Tool-call subsystem
ENV_TOOLCALL_ADAPTER = "HOUYI_TOOLCALL_ADAPTER"
ENV_TOOLCALL_MODEL = "HOUYI_TOOLCALL_MODEL"
ENV_TOOLCALL_MAX_TOKENS = "HOUYI_TOOLCALL_MAX_TOKENS"
ENV_TOOLCALL_TIMING = "HOUYI_TOOLCALL_TIMING"
ENV_TOOLCALL_FAST_PATH = "HOUYI_TOOLCALL_FAST_PATH"
ENV_TOOLCALL_TOOL_LATENCY_MS = "HOUYI_TOOLCALL_TOOL_LATENCY_MS"
ENV_TOOLCALL_MAX_RETRIES = "HOUYI_TOOLCALL_MAX_RETRIES"
ENV_TOOLCALL_TIMEOUT = "HOUYI_TOOLCALL_TIMEOUT"

# Replay / Cache
ENV_FRESH_REPLAY_USE_WEB_SEARCH_CACHE = "HOUYI_FRESH_REPLAY_USE_WEB_SEARCH_CACHE"
ENV_FRESH_REPLAY_USE_TOOL_CACHE = "HOUYI_FRESH_REPLAY_USE_TOOL_CACHE"

# Chat / Studio
ENV_CHAT_SYSTEM_PROMPT = "HOUYI_CHAT_SYSTEM_PROMPT"
ENV_CHAT_DATA_DIR = "HOUYI_CHAT_DATA_DIR"
ENV_CHAT_SETTINGS_PATH = "HOUYI_CHAT_SETTINGS_PATH"

# Server
ENV_HOUYI_PORT = "HOUYI_PORT"
ENV_KNOWLEDGE_STORAGE = "HOUYI_KNOWLEDGE_STORAGE"

# Knowledge / Embedding
ENV_RAG_KNOWLEDGE_DIR = "RAG_KNOWLEDGE_DIR"
ENV_EMBEDDING_PROVIDER = "EMBEDDING_PROVIDER"
ENV_EMBEDDING_MODEL = "EMBEDDING_MODEL"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_GOOGLE_LOCATION = "us-central1"
_DEFAULT_EMBEDDING_PROVIDER = "local"
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_RAG_KNOWLEDGE_DIR = "knowledge/"


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
        return super().__new__(cls)

    def __init__(self) -> None:
        self._siliconflow_api_key: str | None = None
        self._siliconflow_base_url: str = _DEFAULT_SILICONFLOW_BASE_URL
        self._deepseek_model: str = ""
        self._default_llm_provider: str = ""

        self._google_api_key: str | None = None
        self._google_credentials_path: str | None = None
        self._google_project: str | None = None
        self._google_location: str = _DEFAULT_GOOGLE_LOCATION
        self._gemini_model: str = ""

        self._rag_knowledge_dir: str = _DEFAULT_RAG_KNOWLEDGE_DIR
        self._embedding_provider: str = _DEFAULT_EMBEDDING_PROVIDER
        self._embedding_model: str = _DEFAULT_EMBEDDING_MODEL

    @classmethod
    def get(cls) -> EnvConfig:
        """Get or create the global singleton instance."""
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
        from houyi.llm.models import (
            DEFAULT_MODEL,
            GEMINI_25_PRO,
            PROVIDER_GOOGLE_AI,
            PROVIDER_SILICONFLOW,
            PROVIDER_VERTEX,
        )

        # --- LLM Provider ---
        self._siliconflow_api_key = os.getenv(ENV_SILICONFLOW_API_KEY)
        self._siliconflow_base_url = os.getenv(
            ENV_SILICONFLOW_BASE_URL, _DEFAULT_SILICONFLOW_BASE_URL
        )
        self._deepseek_model = os.getenv(ENV_DEEPSEEK_MODEL, DEFAULT_MODEL)
        self._default_llm_provider = os.getenv(ENV_DEFAULT_LLM_PROVIDER, PROVIDER_SILICONFLOW)

        # --- Google (aligned with google-genai SDK) ---
        self._google_api_key = os.getenv(ENV_GOOGLE_API_KEY)
        self._google_credentials_path = os.getenv(ENV_GOOGLE_APPLICATION_CREDENTIALS)
        self._google_project = os.getenv(ENV_GOOGLE_CLOUD_PROJECT)
        self._google_location = os.getenv(ENV_GOOGLE_CLOUD_LOCATION, _DEFAULT_GOOGLE_LOCATION)
        self._gemini_model = os.getenv(ENV_GEMINI_MODEL) or GEMINI_25_PRO

        # Auto-detect project from service account credentials
        if not self._google_project and self._google_credentials_path:
            import json
            import pathlib

            creds_path = pathlib.Path(self._google_credentials_path)
            if not creds_path.is_absolute():
                # Resolve relative path against workspace / .env directory
                for base in [pathlib.Path.cwd(), pathlib.Path(__file__).resolve().parents[3]]:
                    candidate = base / creds_path
                    if candidate.is_file():
                        creds_path = candidate
                        break

            if creds_path.is_file():
                try:
                    with open(creds_path) as f:
                        self._google_project = json.load(f).get("project_id")
                    if self._google_project:
                        logger.info(
                            "Auto-detected GOOGLE_CLOUD_PROJECT=%s from %s",
                            self._google_project,
                            creds_path,
                        )
                except Exception as exc:
                    logger.warning("Failed to read project_id from %s: %s", creds_path, exc)
            else:
                logger.warning(
                    "GOOGLE_APPLICATION_CREDENTIALS=%s — file not found "
                    "(checked cwd=%s). Vertex AI service account mode unavailable.",
                    self._google_credentials_path,
                    pathlib.Path.cwd(),
                )

        # --- Knowledge / Embedding ---
        self._rag_knowledge_dir = os.getenv(ENV_RAG_KNOWLEDGE_DIR, _DEFAULT_RAG_KNOWLEDGE_DIR)
        self._embedding_provider = os.getenv(ENV_EMBEDDING_PROVIDER, _DEFAULT_EMBEDDING_PROVIDER)
        self._embedding_model = os.getenv(ENV_EMBEDDING_MODEL, _DEFAULT_EMBEDDING_MODEL)

        # --- Warnings ---
        if not self._siliconflow_api_key:
            if self._default_llm_provider == PROVIDER_SILICONFLOW:
                logger.error(
                    "%s not set but provider is %s — will use mock responses",
                    ENV_SILICONFLOW_API_KEY,
                    PROVIDER_SILICONFLOW,
                )
            else:
                logger.debug(
                    "%s not set (provider=%s, not needed)",
                    ENV_SILICONFLOW_API_KEY,
                    self._default_llm_provider,
                )
        if not self._google_api_key and not self._google_credentials_path:
            if self._default_llm_provider in (PROVIDER_GOOGLE_AI, PROVIDER_VERTEX):
                logger.error(
                    "Neither %s nor %s set but provider is '%s' — Google adapters will fail",
                    ENV_GOOGLE_API_KEY,
                    ENV_GOOGLE_APPLICATION_CREDENTIALS,
                    self._default_llm_provider,
                )
            else:
                logger.debug(
                    "Neither %s nor %s set (provider=%s, not required)",
                    ENV_GOOGLE_API_KEY,
                    ENV_GOOGLE_APPLICATION_CREDENTIALS,
                    self._default_llm_provider,
                )

        logger.info(
            "EnvConfig loaded: provider=%s, embedding=%s, knowledge_dir=%s",
            self._default_llm_provider,
            self._embedding_provider,
            self._rag_knowledge_dir,
        )

    def reload(self) -> None:
        """Re-read all environment variables."""
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
    # Google properties (aligned with google-genai SDK)
    # ------------------------------------------------------------------

    @property
    def google_api_key(self) -> str | None:
        """GOOGLE_API_KEY or None."""
        return self._google_api_key

    @property
    def google_credentials_path(self) -> str | None:
        """GOOGLE_APPLICATION_CREDENTIALS or None."""
        return self._google_credentials_path

    @property
    def google_project(self) -> str | None:
        """GOOGLE_CLOUD_PROJECT (or auto-detected from credentials) or None."""
        return self._google_project

    @property
    def google_location(self) -> str:
        """GOOGLE_CLOUD_LOCATION or ``"us-central1"``."""
        return self._google_location

    @property
    def gemini_model(self) -> str:
        """GEMINI_MODEL or default ``"gemini-2.5-pro"``."""
        return self._gemini_model

    # Backward-compat alias used by VertexAIAdapter
    @property
    def google_project_id(self) -> str | None:
        """Alias for ``google_project``."""
        return self._google_project

    # ------------------------------------------------------------------
    # Knowledge / Embedding properties
    # ------------------------------------------------------------------

    @property
    def rag_knowledge_dir(self) -> str:
        """RAG_KNOWLEDGE_DIR or default ``"knowledge/"``."""
        return self._rag_knowledge_dir

    @property
    def embedding_provider(self) -> str:
        """EMBEDDING_PROVIDER or default ``"local"``."""
        return self._embedding_provider

    @property
    def embedding_model(self) -> str:
        """EMBEDDING_MODEL or default ``"BAAI/bge-small-en-v1.5"``."""
        return self._embedding_model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a sanitized config summary for logging."""

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
            "google_api_key": _mask(self._google_api_key),
            "google_credentials_path": self._google_credentials_path or "(not set)",
            "google_project": self._google_project or "(not set)",
            "google_location": self._google_location,
            "gemini_model": self._gemini_model,
            "rag_knowledge_dir": self._rag_knowledge_dir,
            "embedding_provider": self._embedding_provider,
            "embedding_model": self._embedding_model,
        }

    def __repr__(self) -> str:
        return f"<EnvConfig provider={self._default_llm_provider!r} embedding={self._embedding_provider!r}>"
