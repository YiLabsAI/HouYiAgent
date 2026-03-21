from __future__ import annotations

import logging
import os
from typing import Any

from houyi.adapters.llm import LLMAdapter
from houyi.adapters.llm.models import PROVIDER_SILICONFLOW

logger = logging.getLogger(__name__)

_SUMMARY_MODEL_ENV = "HOUYI_CHAT_SUMMARY_MODEL"


class ModelAdapterResolver:
    def __init__(
        self,
        *,
        get_settings_store: Any,
        get_default_adapter: Any,
        is_vertex_provider: Any,
        create_vertex_adapter: Any,
        openai_compat_adapter_cls: Any,
        siliconflow_adapter_cls: Any,
    ) -> None:
        self._get_settings_store = get_settings_store
        self._get_default_adapter = get_default_adapter
        self._is_vertex_provider = is_vertex_provider
        self._create_vertex_adapter = create_vertex_adapter
        self._openai_compat_adapter_cls = openai_compat_adapter_cls
        self._siliconflow_adapter_cls = siliconflow_adapter_cls
        self._adapter_cache: dict[str, LLMAdapter] = {}

    @property
    def adapter_cache(self) -> dict[str, LLMAdapter]:
        return self._adapter_cache

    def invalidate_adapter_cache(self) -> None:
        self._adapter_cache.clear()
        logger.info("LLM adapter cache invalidated")

    def resolve_summary_model(self, model: str) -> str | None:
        _ = model
        raw = str(os.getenv(_SUMMARY_MODEL_ENV, "") or "").strip()
        return raw or None

    def get_adapter_for_model(self, model: str) -> LLMAdapter:
        settings_store = self._get_settings_store()
        default_adapter = self._get_default_adapter()
        if not settings_store:
            return default_adapter
        settings = settings_store.get()
        for provider in settings.providers:
            if not provider.enabled:
                continue
            if model not in provider.models:
                continue
            cached = self._adapter_cache.get(provider.id)
            if cached is not None:
                return cached
            adapter = self._build_provider_adapter(
                provider=provider,
                model=model,
                default_adapter=default_adapter,
            )
            if adapter is default_adapter:
                return adapter
            self._adapter_cache[provider.id] = adapter
            logger.info(
                "Model '%s' routed to provider '%s' (%s)",
                model,
                provider.name,
                provider.base_url or "default",
            )
            return adapter
        return default_adapter

    def _build_provider_adapter(
        self,
        *,
        provider: Any,
        model: str,
        default_adapter: LLMAdapter,
    ) -> LLMAdapter:
        provider_id = str(provider.id or "").strip().lower()
        provider_name = str(provider.name or "").strip().lower()
        provider_url = (provider.base_url or "").rstrip("/")
        provider_url_lower = provider_url.lower()
        if self._is_vertex_provider(provider.id, provider_url) or self._is_vertex_provider(
            provider_name, provider_url
        ):
            return self._create_vertex_adapter()
        # Keep the base_url fallback for SiliconFlow so custom provider ids or
        # names that still point at a SiliconFlow endpoint continue to use the
        # provider-specific adapter instead of drifting into the generic
        # OpenAI-compatible path.
        if (
            provider_id == PROVIDER_SILICONFLOW
            or provider_name == PROVIDER_SILICONFLOW
            or PROVIDER_SILICONFLOW in provider_url_lower
        ):
            return self._siliconflow_adapter_cls(
                api_key=provider.api_key or None,
                base_url=provider_url,
                default_model=model,
            )
        # Any other `/v1` endpoint should use the generic OpenAI-compatible
        # adapter. Routing all `/v1` providers into SiliconFlowAdapter causes
        # cross-provider behavior drift (e.g. placeholder/tool-call shaping).
        if provider_url and "/v1" in provider_url:
            return self._openai_compat_adapter_cls(
                api_key=provider.api_key or None,
                base_url=provider_url,
                model=model,
            )
        logger.info(
            "Model '%s' provider '%s' has no OpenAI-compatible base_url, using default adapter",
            model,
            provider.name,
        )
        return default_adapter
