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

from houyi.adapters.llm.base import LLMAdapter
from houyi.adapters.llm.models import (
    DEFAULT_MODEL,
    PROVIDER_DASHSCOPE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GOOGLE_AI,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPAT,
    PROVIDER_SILICONFLOW,
    PROVIDER_VERTEX,
)

logger = logging.getLogger(__name__)

_VERTEX_ALIASES = frozenset({PROVIDER_VERTEX, PROVIDER_GOOGLE_AI})
_ENV_VERTEX_ADAPTER = "HOUYI_VERTEX_ADAPTER"


class LLMAdapterFactory:
    """Factory for creating LLM adapters.

    Usage::

        adapter = LLMAdapterFactory.create()                  # env default
        adapter = LLMAdapterFactory.create("vertex")           # explicit
        adapter = LLMAdapterFactory.create("deepseek")         # DeepSeek
        adapter = LLMAdapterFactory.create("openai_compat")    # generic
    """

    @staticmethod
    def create(
        provider: str | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMAdapter:
        """Create an LLM adapter.

        Args:
            provider: Provider name.  Supported values:

                * siliconflow — SiliconFlowAdapter (default)
                * vertex / google_ai — Gemini
                * openai — OpenAI via the SDK
                * openai_compat — generic OpenAI-compatible endpoint
                * deepseek — DeepSeek via OpenAI-compatible endpoint
                * dashscope — Alibaba Cloud Bailian via OpenAI-compatible endpoint
                * None — reads DEFAULT_LLM_PROVIDER from env
            model: Optional per-call model override. When omitted each provider
                resolves its own configured/default model.
            api_key: Optional credential override. When omitted the provider
                resolves its own keyed credential (never another provider's key).
            base_url: Optional endpoint override.

        Returns:
            Configured LLM adapter instance.
        """
        from houyi.infrastructure.config.env_config import EnvConfig

        provider = provider or EnvConfig.get().default_llm_provider

        if provider == PROVIDER_SILICONFLOW:
            from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter

            return SiliconFlowAdapter(api_key=api_key, base_url=base_url, default_model=model)

        if provider in _VERTEX_ALIASES:
            return _create_vertex_adapter()

        if provider == PROVIDER_OPENAI:
            return _create_openai_adapter()

        if provider == PROVIDER_OPENAI_COMPAT:
            return _create_openai_compat_adapter()

        if provider == PROVIDER_DEEPSEEK:
            return _create_deepseek_adapter()

        if provider == PROVIDER_DASHSCOPE:
            return _create_dashscope_adapter(model=model, api_key=api_key, base_url=base_url)

        logger.warning("Unknown provider %s, falling back to SiliconFlow", provider)
        from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter

        return SiliconFlowAdapter()


def _create_vertex_adapter() -> LLMAdapter:
    """Create the best available Gemini/Vertex AI adapter.

    Priority:
    1. google-genai SDK (Vertex AI if project available, else Developer API)
    2. httpx-based VertexAIAdapter (service account JWT fallback)
    """
    from houyi.infrastructure.config.env_config import EnvConfig

    env = EnvConfig.get()
    route_mode = str(os.getenv(_ENV_VERTEX_ADAPTER, "auto") or "auto").strip().lower()

    if route_mode == "httpx":
        logger.info("Using VertexAIAdapter (httpx JWT mode, forced by %s)", _ENV_VERTEX_ADAPTER)
        from houyi.adapters.llm.vertex_httpx_adapter import VertexAIAdapter

        return VertexAIAdapter()

    if route_mode == "genai":
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

        return GoogleVertexGeminiAdapter.from_env()

    if env.google_project or env.google_api_key:
        try:
            from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

            adapter = GoogleVertexGeminiAdapter.from_env()
            return adapter
        except ImportError:
            logger.info("google-genai SDK not installed — falling back to httpx Vertex AI adapter")
        except ValueError as exc:
            logger.warning("GoogleVertexGeminiAdapter: %s — falling back", exc)

    logger.info("Using VertexAIAdapter (httpx JWT mode)")
    from houyi.adapters.llm.vertex_httpx_adapter import VertexAIAdapter

    return VertexAIAdapter()


def _create_openai_adapter() -> LLMAdapter:
    """Create an OpenAI adapter (requires openai SDK)."""
    from houyi.adapters.llm.openai_adapter import OpenAIAdapter

    return OpenAIAdapter()


def _create_openai_compat_adapter() -> LLMAdapter:
    """Create a generic OpenAI-compatible adapter."""
    from houyi.adapters.llm.openai_compat_adapter import OpenAICompatibleAdapter

    return OpenAICompatibleAdapter()


def _create_deepseek_adapter() -> LLMAdapter:
    """Create a DeepSeek adapter via the OpenAI-compatible endpoint."""
    from houyi.adapters.llm.openai_compat_adapter import OpenAICompatibleAdapter
    from houyi.infrastructure.config.env_config import (
        ENV_DEEPSEEK_API_KEY,
        ENV_DEEPSEEK_BASE_URL,
        ENV_OPENAI_API_KEY,
        ENV_OPENAI_BASE_URL,
        ENV_TOOLCALL_MODEL,
        EnvConfig,
    )

    api_key = os.getenv(ENV_DEEPSEEK_API_KEY) or os.getenv(ENV_OPENAI_API_KEY)
    base_url = os.getenv(ENV_DEEPSEEK_BASE_URL) or os.getenv(ENV_OPENAI_BASE_URL)
    model = os.getenv(ENV_TOOLCALL_MODEL) or EnvConfig.get().siliconflow_model or DEFAULT_MODEL
    return OpenAICompatibleAdapter(api_key=api_key, base_url=base_url, model=model)


def _create_dashscope_adapter(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMAdapter:
    """Create a Bailian (DashScope) adapter via the OpenAI-compatible endpoint.

    DashScope exposes an OpenAI-compatible surface at compatible-mode/v1, so it
    reuses OpenAICompatibleAdapter with DashScope-keyed credentials. The adapter
    raises if no DashScope key is available rather than silently borrowing
    OPENAI_API_KEY, so a misconfigured run fails fast instead of routing to the
    wrong endpoint.
    """
    from houyi.adapters.llm.openai_compat_adapter import OpenAICompatibleAdapter
    from houyi.infrastructure.config.env_config import ENV_DASHSCOPE_API_KEY, EnvConfig

    env = EnvConfig.get()
    resolved_key = api_key or env.dashscope_api_key
    if not resolved_key:
        raise ValueError(
            f"{ENV_DASHSCOPE_API_KEY} is not set but the LLM provider is dashscope. "
            "DashScope must use its own key; falling back to OPENAI_API_KEY would "
            "silently route requests to the wrong endpoint. Set "
            f"{ENV_DASHSCOPE_API_KEY} or choose a different --llm-provider."
        )
    return OpenAICompatibleAdapter(
        api_key=resolved_key,
        base_url=base_url or env.dashscope_base_url,
        model=model or env.dashscope_model,
    )


create_vertex_adapter = _create_vertex_adapter
