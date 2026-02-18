"""LLM adapter factory.

Central entry point for creating LLM adapters based on provider configuration.
Encapsulates the strategy selection logic (SDK vs httpx fallback for Vertex AI,
API key detection, etc.).

This factory is the single source of truth for adapter construction and is used
by both the chat service and the tool-call adapter hooks system.
"""

from __future__ import annotations

import logging
import os

from houyi.llm.base import LLMAdapter
from houyi.llm.models import (
    DEFAULT_MODEL,
    PROVIDER_DEEPSEEK,
    PROVIDER_GOOGLE_AI,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPAT,
    PROVIDER_SILICONFLOW,
    PROVIDER_VERTEX,
)

logger = logging.getLogger(__name__)

_VERTEX_ALIASES = frozenset({PROVIDER_VERTEX, PROVIDER_GOOGLE_AI})


class LLMAdapterFactory:
    """Factory for creating LLM adapters.

    Usage::

        adapter = LLMAdapterFactory.create()                  # env default
        adapter = LLMAdapterFactory.create("vertex")           # explicit
        adapter = LLMAdapterFactory.create("deepseek")         # DeepSeek
        adapter = LLMAdapterFactory.create("openai_compat")    # generic
    """

    @staticmethod
    def create(provider: str | None = None) -> LLMAdapter:
        """Create an LLM adapter.

        Args:
            provider: Provider name.  Supported values:

                * ``siliconflow`` — SiliconFlowAdapter (default)
                * ``vertex`` / ``google_ai`` — Gemini
                * ``openai`` — OpenAI via the SDK
                * ``openai_compat`` — generic OpenAI-compatible endpoint
                * ``deepseek`` — DeepSeek via OpenAI-compatible endpoint
                * ``None`` — reads ``DEFAULT_LLM_PROVIDER`` from env

        Returns:
            Configured LLM adapter instance.
        """
        from houyi.config.env_config import EnvConfig

        provider = provider or EnvConfig.get().default_llm_provider

        if provider == PROVIDER_SILICONFLOW:
            from houyi.llm.siliconflow_adapter import SiliconFlowAdapter

            return SiliconFlowAdapter()

        if provider in _VERTEX_ALIASES:
            return _create_vertex_adapter()

        if provider == PROVIDER_OPENAI:
            return _create_openai_adapter()

        if provider == PROVIDER_OPENAI_COMPAT:
            return _create_openai_compat_adapter()

        if provider == PROVIDER_DEEPSEEK:
            return _create_deepseek_adapter()

        logger.warning("Unknown provider %s, falling back to SiliconFlow", provider)
        from houyi.llm.siliconflow_adapter import SiliconFlowAdapter

        return SiliconFlowAdapter()


def _create_vertex_adapter() -> LLMAdapter:
    """Create the best available Gemini/Vertex AI adapter.

    Priority:
    1. google-genai SDK (Vertex AI if project available, else Developer API)
    2. httpx-based VertexAIAdapter (service account JWT fallback)
    """
    from houyi.config.env_config import EnvConfig

    env = EnvConfig.get()

    if env.google_project or env.google_api_key:
        try:
            from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

            adapter = GoogleVertexGeminiAdapter.from_env()
            return adapter
        except ImportError:
            logger.info("google-genai SDK not installed — falling back to httpx Vertex AI adapter")
        except ValueError as exc:
            logger.warning("GoogleVertexGeminiAdapter: %s — falling back", exc)

    logger.info("Using VertexAIAdapter (httpx JWT mode)")
    from houyi.llm.vertex_httpx_adapter import VertexAIAdapter

    return VertexAIAdapter()


def _create_openai_adapter() -> LLMAdapter:
    """Create an OpenAI adapter (requires ``openai`` SDK)."""
    from houyi.llm.openai_adapter import OpenAIAdapter

    return OpenAIAdapter()


def _create_openai_compat_adapter() -> LLMAdapter:
    """Create a generic OpenAI-compatible adapter."""
    from houyi.llm.openai_compat_adapter import OpenAICompatibleAdapter

    return OpenAICompatibleAdapter()


def _create_deepseek_adapter() -> LLMAdapter:
    """Create a DeepSeek adapter via the OpenAI-compatible endpoint."""
    from houyi.config.env_config import (
        ENV_DEEPSEEK_API_KEY,
        ENV_DEEPSEEK_BASE_URL,
        ENV_DEEPSEEK_MODEL,
        ENV_OPENAI_API_KEY,
        ENV_OPENAI_BASE_URL,
        ENV_TOOLCALL_MODEL,
    )
    from houyi.llm.openai_compat_adapter import OpenAICompatibleAdapter

    api_key = os.getenv(ENV_DEEPSEEK_API_KEY) or os.getenv(ENV_OPENAI_API_KEY)
    base_url = os.getenv(ENV_DEEPSEEK_BASE_URL) or os.getenv(ENV_OPENAI_BASE_URL)
    model = os.getenv(ENV_DEEPSEEK_MODEL) or os.getenv(ENV_TOOLCALL_MODEL) or DEFAULT_MODEL
    return OpenAICompatibleAdapter(api_key=api_key, base_url=base_url, model=model)
