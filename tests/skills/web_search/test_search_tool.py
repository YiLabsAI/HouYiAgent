"""Unit tests for WebSearchTool — runtime tool wrapping WebSearchService."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from houyi.application.research.tools import WebSearchTool
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import WebSearchMetadata, WebSearchResponse, WebSearchResult


def _mock_web_search() -> WebSearchService:
    svc = AsyncMock(spec=WebSearchService)
    svc.search = AsyncMock(
        return_value=WebSearchResponse(
            query="test",
            provider="mock",
            results=[
                WebSearchResult(
                    title="Mock Result 1",
                    url="https://example.com/1",
                    snippet="snippet 1",
                    content="Full content 1",
                ),
                WebSearchResult(
                    title="Mock Result 2",
                    url="https://example.com/2",
                    snippet="snippet 2",
                    content="Full content 2",
                ),
            ],
            metadata=WebSearchMetadata(
                cached=False, cache_hit=False, latency_ms=10, provider="mock"
            ),
        ),
    )
    return svc


class TestWebSearchToolCall:
    async def test_returns_json(self):
        tool = WebSearchTool(_mock_web_search())
        raw = await tool("AI frameworks", max_results=5, include_content=True)
        data = json.loads(raw)
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["provider"] == "mock"
        assert data["results"][0]["url"] == "https://example.com/1"
        assert data["results"][0]["content"] != ""

    async def test_without_content(self):
        tool = WebSearchTool(_mock_web_search())
        raw = await tool("AI frameworks", include_content=False)
        data = json.loads(raw)
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert r["content"] == ""

    async def test_error_returns_json(self):
        ws = _mock_web_search()
        ws.search.side_effect = RuntimeError("network failure")
        tool = WebSearchTool(ws)
        raw = await tool("failing query")
        data = json.loads(raw)
        assert "error" in data
        assert data["results"] == []
        assert "network failure" in data["error"]
