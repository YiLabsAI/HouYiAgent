from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_ALLOWED_WEB_SEARCH_PROVIDERS = {"ddg", "serper", "tavily", "bocha"}
_WEB_SEARCH_TOOL_NAMES = frozenset({"web_search", "houyi_web_search"})


@dataclass(frozen=True)
class WebSearchProviderHook:
    provider: str

    async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        if tool_call.get("tool_name") not in _WEB_SEARCH_TOOL_NAMES:
            return None
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return None
        requested_provider = args.get("provider", self.provider)
        normalized_provider = str(requested_provider or "").strip().lower()
        if not normalized_provider:
            return None
        if normalized_provider not in _ALLOWED_WEB_SEARCH_PROVIDERS:
            raise ValueError(
                "Unsupported web_search provider "
                f"'{normalized_provider}'. Allowed providers: ddg, serper, tavily, bocha"
            )
        if args.get("provider") == normalized_provider:
            return None
        return {"args": {**args, "provider": normalized_provider}}


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
