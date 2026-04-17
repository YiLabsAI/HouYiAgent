"""Shared fixtures for research engine tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import WebSearchMetadata, WebSearchResponse, WebSearchResult


class MockLLM(LLMAdapter):
    """LLM adapter that returns pre-configured responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    def _next_content(self) -> str:
        content = (
            self._responses[self._call_count] if self._call_count < len(self._responses) else "{}"
        )
        self._call_count += 1
        return content

    async def chat(self, messages: list, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._next_content(), finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content_delta=self._next_content())


def make_mock_web_search(results: list[WebSearchResult] | None = None) -> WebSearchService:
    """Create a mock WebSearchService that returns pre-configured results."""
    svc = AsyncMock(spec=WebSearchService)
    svc.search = AsyncMock(
        return_value=WebSearchResponse(
            query="test",
            provider="mock",
            results=results
            or [
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


def make_unique_url_web_search() -> WebSearchService:
    """Mock WebSearchService returning unique URLs on every call.

    Useful when cross-sub-question URL dedup (claim_url) is active and each
    sub-question must receive its own distinct sources.
    """
    _counter = {"n": 0}

    async def _search(*args: Any, **kwargs: Any) -> WebSearchResponse:
        idx = _counter["n"]
        _counter["n"] += 1
        return WebSearchResponse(
            query="test",
            provider="mock",
            results=[
                WebSearchResult(
                    title=f"Mock Result {idx}a",
                    url=f"https://example.com/{idx}a",
                    snippet=f"snippet {idx}a",
                    content=f"Full content {idx}a",
                ),
                WebSearchResult(
                    title=f"Mock Result {idx}b",
                    url=f"https://example.com/{idx}b",
                    snippet=f"snippet {idx}b",
                    content=f"Full content {idx}b",
                ),
            ],
            metadata=WebSearchMetadata(
                cached=False, cache_hit=False, latency_ms=10, provider="mock"
            ),
        )

    svc = AsyncMock(spec=WebSearchService)
    svc.search = _search
    return svc


@pytest.fixture
def mock_llm():
    """Default mock LLM returning valid JSON for planner."""
    plan_json = json.dumps(
        {
            "sub_questions": [
                {
                    "question": "What are current frameworks?",
                    "priority": 5,
                    "search_strategy": "web",
                    "expected_sources": 5,
                },
                {
                    "question": "How do they compare?",
                    "priority": 4,
                    "search_strategy": "web",
                    "expected_sources": 5,
                },
                {
                    "question": "Future trends?",
                    "priority": 3,
                    "search_strategy": "web",
                    "expected_sources": 3,
                },
                {
                    "question": "What evidence compares practical tradeoffs?",
                    "priority": 2,
                    "search_strategy": "web",
                    "expected_sources": 3,
                },
                {
                    "question": "What limitations or risks are reported?",
                    "priority": 1,
                    "search_strategy": "web",
                    "expected_sources": 3,
                },
            ],
            "outline": [
                {
                    "title": "Overview",
                    "objective": "Landscape overview",
                    "related_question_ids": [0],
                },
                {
                    "title": "Comparison",
                    "objective": "Feature comparison",
                    "related_question_ids": [1],
                },
                {"title": "Outlook", "objective": "Future directions", "related_question_ids": [2]},
                {
                    "title": "Tradeoffs",
                    "objective": "Practical tradeoff analysis",
                    "related_question_ids": [3],
                },
                {
                    "title": "Limitations",
                    "objective": "Known limitations and risks",
                    "related_question_ids": [4],
                },
            ],
            "estimated_duration_min": 8,
        }
    )
    return MockLLM(responses=[plan_json])


@pytest.fixture
def mock_web_search():
    return make_mock_web_search()
