from __future__ import annotations

import pytest

from houyi.execution.tool_call_web_search_hooks import (
    WebSearchCachePolicyHook,
    WebSearchProviderHook,
    build_web_search_tool_hooks,
)


class TestToolCallWebSearchHooks:
    @pytest.mark.asyncio
    async def test_provider_hook_injects_when_missing(self) -> None:
        hook = WebSearchProviderHook(provider="tavily")
        patch = await hook.before_tool_call({"tool_name": "web_search", "args": {"query": "q"}})
        assert patch == {"args": {"query": "q", "provider": "tavily"}}

    @pytest.mark.asyncio
    async def test_provider_hook_noop_when_provider_present(self) -> None:
        hook = WebSearchProviderHook(provider="tavily")
        patch = await hook.before_tool_call(
            {"tool_name": "web_search", "args": {"query": "q", "provider": "x"}}
        )
        assert patch is None

    @pytest.mark.asyncio
    async def test_cache_policy_hook_injects_when_missing(self) -> None:
        hook = WebSearchCachePolicyHook(use_cache=False)
        patch = await hook.before_tool_call({"tool_name": "web_search", "args": {"query": "q"}})
        assert patch == {"args": {"query": "q", "use_cache": False}}

    @pytest.mark.asyncio
    async def test_cache_policy_hook_noop_when_use_cache_present(self) -> None:
        hook = WebSearchCachePolicyHook(use_cache=False)
        patch = await hook.before_tool_call(
            {"tool_name": "web_search", "args": {"query": "q", "use_cache": True}}
        )
        assert patch is None

    def test_build_web_search_tool_hooks_none(self) -> None:
        assert (
            build_web_search_tool_hooks(
                web_search_provider=None,
                replay_mode=None,
                allow_fresh_web_search_cache=False,
            )
            is None
        )

    def test_build_web_search_tool_hooks_provider_only(self) -> None:
        hooks = build_web_search_tool_hooks(
            web_search_provider="tavily",
            replay_mode=None,
            allow_fresh_web_search_cache=False,
        )
        assert hooks
        assert isinstance(hooks[0], WebSearchProviderHook)

    def test_build_web_search_tool_hooks_fresh_adds_cache_policy(self) -> None:
        hooks = build_web_search_tool_hooks(
            web_search_provider="tavily",
            replay_mode="fresh",
            allow_fresh_web_search_cache=False,
        )
        assert hooks
        assert any(isinstance(h, WebSearchProviderHook) for h in hooks)
        assert any(isinstance(h, WebSearchCachePolicyHook) for h in hooks)

    def test_build_web_search_tool_hooks_fresh_allowed_skips_cache_policy(self) -> None:
        hooks = build_web_search_tool_hooks(
            web_search_provider="tavily",
            replay_mode="fresh",
            allow_fresh_web_search_cache=True,
        )
        assert hooks
        assert any(isinstance(h, WebSearchProviderHook) for h in hooks)
        assert not any(isinstance(h, WebSearchCachePolicyHook) for h in hooks)
