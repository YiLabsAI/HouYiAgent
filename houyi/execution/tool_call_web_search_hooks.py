from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WebSearchProviderHook:
    provider: str

    async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        if tool_call.get("tool_name") != "web_search":
            return None
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return None
        if "provider" in args:
            return None
        return {"args": {**args, "provider": self.provider}}


@dataclass(frozen=True)
class WebSearchCachePolicyHook:
    use_cache: bool

    async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        if tool_call.get("tool_name") != "web_search":
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
