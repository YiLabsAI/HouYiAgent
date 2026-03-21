from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from houyi.skills.web_search.provider_resolution import (
    normalize_web_search_provider_name,
    resolve_supported_web_search_provider,
)

_WEB_SEARCH_TOOL_NAMES = frozenset({"web_search", "houyi_web_search"})

logger = logging.getLogger(__name__)


def _normalize_provider_name(provider: Any) -> str:
    return normalize_web_search_provider_name(str(provider or ""))


def _resolve_supported_provider(requested_provider: Any, configured_provider: Any) -> str:
    return resolve_supported_web_search_provider(
        str(requested_provider or ""),
        configured_provider=str(configured_provider or ""),
    )


@dataclass(frozen=True)
class WebSearchProviderHook:
    provider: str

    async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        if tool_call.get("tool_name") not in _WEB_SEARCH_TOOL_NAMES:
            return None
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return None
        requested_provider = args.get("provider")
        resolved_provider = _resolve_supported_provider(requested_provider, self.provider)
        if not resolved_provider:
            return None
        if _normalize_provider_name(requested_provider) != resolved_provider:
            logger.info(
                "web_search provider normalized: requested=%s resolved=%s configured=%s",
                _normalize_provider_name(requested_provider),
                resolved_provider,
                _normalize_provider_name(self.provider),
            )
        if _normalize_provider_name(args.get("provider")) == resolved_provider:
            return None
        return {"args": {**args, "provider": resolved_provider}}


@dataclass(frozen=True)
class WebSearchCachePolicyHook:
    use_cache: bool

    async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        if tool_call.get("tool_name") not in _WEB_SEARCH_TOOL_NAMES:
            return None
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return None
        if "use_cache" in args:
            return None
        return {"args": {**args, "use_cache": self.use_cache}}


def build_web_search_tool_hooks(
    *,
    web_search_provider: str | None,
    replay_mode: str | None,
    allow_fresh_web_search_cache: bool,
) -> list[Any] | None:
    if not web_search_provider:
        return None

    hooks: list[Any] = [WebSearchProviderHook(provider=web_search_provider)]

    if replay_mode == "fresh" and not allow_fresh_web_search_cache:
        hooks.append(WebSearchCachePolicyHook(use_cache=False))

    return hooks


__all__ = [
    "WebSearchCachePolicyHook",
    "WebSearchProviderHook",
    "build_web_search_tool_hooks",
]
