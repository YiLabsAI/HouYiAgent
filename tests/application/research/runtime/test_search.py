"""Unit tests for SearchCoordinator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from houyi.application.research.runtime.search import (
    SearchCoordinator,
    _parse_query_list,
    _parse_sufficiency,
)
from houyi.application.research.types import SearchContext, SubQuestion
from houyi.skills.web_search.types import WebSearchMetadata, WebSearchResponse

from ..conftest import MockLLM, make_mock_web_search


def _context() -> SearchContext:
    return SearchContext(run_id="r1", plan_id="p1", user_query="AI frameworks")


class TestSearch:
    async def test_single_round_sufficient(self):
        llm = MockLLM(
            responses=[
                '["ai agent frameworks 2026"]',
                json.dumps({"sufficient": True, "rationale": "enough info"}),
            ]
        )
        ws = make_mock_web_search()
        coord = SearchCoordinator(llm, ws, max_search_rounds=3)
        result = await coord.search(SubQuestion(question="What frameworks?"), _context())
        assert len(result.rounds) == 1
        assert result.rounds[0].sufficient is True
        assert len(result.sources) >= 1

    async def test_multi_round_exhaustion(self):
        responses = []
        for _ in range(3):
            responses.append('["query"]')
            responses.append(json.dumps({"sufficient": False, "rationale": "need more"}))
        llm = MockLLM(responses=responses)
        ws = make_mock_web_search()
        coord = SearchCoordinator(llm, ws, max_search_rounds=3)
        result = await coord.search(SubQuestion(question="Deep dive?"), _context())
        assert result.exhausted is True
        assert len(result.rounds) == 3

    async def test_coverage_score(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        coord = SearchCoordinator(llm, ws)
        result = await coord.search(SubQuestion(question="Q?", expected_sources=10), _context())
        assert 0 <= result.coverage_score <= 1.0

    async def test_excluded_urls_skipped(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        ctx = SearchContext(
            run_id="r1",
            plan_id="p1",
            user_query="test",
            excluded_urls=["https://example.com/1"],
        )
        coord = SearchCoordinator(llm, ws)
        result = await coord.search(SubQuestion(question="Q?"), ctx)
        urls = [s.url for s in result.sources]
        assert "https://example.com/1" not in urls


class TestParseHelpers:
    def test_parse_query_list_valid(self):
        assert _parse_query_list('["a", "b"]') == ["a", "b"]

    def test_parse_query_list_fenced(self):
        assert _parse_query_list('```json\n["a"]\n```') == ["a"]

    def test_parse_query_list_fallback(self):
        assert _parse_query_list("just a query") == ["just a query"]

    def test_parse_query_list(self):
        payload = (
            "Thoughts...\n"
            "**Query 1:** first focused query\n"
            "**Query 2:** second focused query\n"
            "**Query 3:** third focused query"
        )
        assert _parse_query_list(payload) == [
            "first focused query",
            "second focused query",
            "third focused query",
        ]

    def test_parse_query_list_truncates(self):
        payload = ["x" * 500]
        parsed = _parse_query_list(json.dumps(payload))
        assert len(parsed) == 1
        assert len(parsed[0]) == 380

    def test_parse_sufficiency_true(self):
        ok, _ = _parse_sufficiency('{"sufficient": true, "rationale": "ok"}')
        assert ok is True

    def test_parse_sufficiency_false(self):
        ok, _ = _parse_sufficiency('{"sufficient": false, "rationale": "need more"}')
        assert ok is False

    def test_parse_sufficiency_malformed(self):
        ok, _ = _parse_sufficiency("garbage")
        assert ok is False


class TestBoundaryAndInteraction:
    async def test_zero_results_from_search(self):
        llm = MockLLM(
            responses=[
                '["empty topic query"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        ws.search = AsyncMock(
            return_value=WebSearchResponse(
                query="test",
                provider="mock",
                results=[],
                metadata=WebSearchMetadata(
                    cached=False, cache_hit=False, latency_ms=10, provider="mock"
                ),
            )
        )
        coord = SearchCoordinator(llm, ws)
        result = await coord.search(SubQuestion(question="Nothing?"), _context())
        assert result.sources == []
        assert result.coverage_score == 0.0

    async def test_search_exception_handled(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        ws.search = AsyncMock(side_effect=RuntimeError("network down"))
        coord = SearchCoordinator(llm, ws)
        result = await coord.search(SubQuestion(question="Q?"), _context())
        assert result.sources == []
        assert len(result.rounds) == 1

    async def test_search_call_count_matches(self):
        llm = MockLLM(
            responses=[
                '["q1", "q2"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        coord = SearchCoordinator(llm, ws)
        await coord.search(SubQuestion(question="Q?"), _context())
        assert ws.search.call_count == 2
