from __future__ import annotations

from unittest.mock import patch

import pytest

from houyi.application.tool_calling.tool_hooks.web_search import (
    WebSearchCachePolicyHook,
    WebSearchProviderHook,
    build_web_search_tool_hooks,
)


class TestToolCallWebSearchHooks:
    @pytest.mark.asyncio
    async def test_provider_injects(self) -> None:
        hook = WebSearchProviderHook(provider="tavily")
        patch = await hook.before_tool_call({"tool_name": "web_search", "args": {"query": "q"}})
        assert patch == {"args": {"query": "q", "provider": "tavily"}}

    @pytest.mark.asyncio
    async def test_injects_chat_alias(self) -> None:
        hook = WebSearchProviderHook(provider="tavily")
        patch = await hook.before_tool_call(
            {"tool_name": "houyi_web_search", "args": {"query": "q"}}
        )
        assert patch == {"args": {"query": "q", "provider": "tavily"}}

    @pytest.mark.asyncio
    async def test_provider_keeps(self) -> None:
        hook = WebSearchProviderHook(provider="tavily")
        patch = await hook.before_tool_call(
            {"tool_name": "web_search", "args": {"query": "q", "provider": "tavily"}}
        )
        assert patch is None

    @pytest.mark.asyncio
    async def test_provider_falls_back(self) -> None:
        hook = WebSearchProviderHook(provider="serper")
        result = await hook.before_tool_call(
            {"tool_name": "web_search", "args": {"query": "q", "provider": "google"}}
        )
        assert result == {"args": {"query": "q", "provider": "serper"}}

    @pytest.mark.asyncio
    async def test_provider_unknown(self) -> None:
        hook = WebSearchProviderHook(provider="serper")
        result = await hook.before_tool_call(
            {"tool_name": "web_search", "args": {"query": "q", "provider": "bing"}}
        )
        assert result == {"args": {"query": "q", "provider": "serper"}}

    @pytest.mark.asyncio
    async def test_provider_logs_normalization(self) -> None:
        hook = WebSearchProviderHook(provider="serper")
        with patch("houyi.application.tool_calling.tool_hooks.web_search.logger.info") as mock_info:
            result = await hook.before_tool_call(
                {
                    "tool_name": "web_search",
                    "args": {"query": "q", "provider": "google_scholar"},
                }
            )

        assert result == {"args": {"query": "q", "provider": "serper"}}
        mock_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_injects(self) -> None:
        hook = WebSearchCachePolicyHook(use_cache=False)
        patch = await hook.before_tool_call({"tool_name": "web_search", "args": {"query": "q"}})
        assert patch == {"args": {"query": "q", "use_cache": False}}

    @pytest.mark.asyncio
    async def test_cache_injects_chat_alias(self) -> None:
        hook = WebSearchCachePolicyHook(use_cache=False)
        patch = await hook.before_tool_call(
            {"tool_name": "houyi_web_search", "args": {"query": "q"}}
        )
        assert patch == {"args": {"query": "q", "use_cache": False}}

    @pytest.mark.asyncio
    async def test_cache_keeps(self) -> None:
        hook = WebSearchCachePolicyHook(use_cache=False)
        patch = await hook.before_tool_call(
            {"tool_name": "web_search", "args": {"query": "q", "use_cache": True}}
        )
        assert patch is None

    def test_hooks_none(self) -> None:
        assert (
            build_web_search_tool_hooks(
                web_search_provider=None,
                replay_mode=None,
                allow_fresh_web_search_cache=False,
            )
            is None
        )

    def test_hooks_provider(self) -> None:
        hooks = build_web_search_tool_hooks(
            web_search_provider="tavily",
            replay_mode=None,
            allow_fresh_web_search_cache=False,
        )
        assert hooks
        assert isinstance(hooks[0], WebSearchProviderHook)

    def test_hooks_fresh_cache(self) -> None:
        hooks = build_web_search_tool_hooks(
            web_search_provider="tavily",
            replay_mode="fresh",
            allow_fresh_web_search_cache=False,
        )
        assert hooks
        assert any(isinstance(h, WebSearchProviderHook) for h in hooks)
        assert any(isinstance(h, WebSearchCachePolicyHook) for h in hooks)

    def test_hooks_fresh_skipcache(self) -> None:
        hooks = build_web_search_tool_hooks(
            web_search_provider="tavily",
            replay_mode="fresh",
            allow_fresh_web_search_cache=True,
        )
        assert hooks
        assert any(isinstance(h, WebSearchProviderHook) for h in hooks)
        assert not any(isinstance(h, WebSearchCachePolicyHook) for h in hooks)
