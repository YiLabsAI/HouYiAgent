from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

from houyi.application.research.runtime.search_executor import SearchExecutor
from houyi.application.research.types import SearchContext, SubQuestion
from houyi.skills.web_search.types import WebSearchMetadata, WebSearchResponse, WebSearchResult

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
        coord = SearchExecutor(llm, ws, max_search_rounds=3)
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
        coord = SearchExecutor(llm, ws, max_search_rounds=3)
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
        coord = SearchExecutor(llm, ws)
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
        coord = SearchExecutor(llm, ws)
        result = await coord.search(SubQuestion(question="Q?"), ctx)
        urls = [s.url for s in result.sources]
        assert "https://example.com/1" not in urls

    async def test_query_parallelism(self):
        llm = MockLLM(
            responses=[
                '["q1", "q2"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        started: set[str] = set()
        release = asyncio.Event()

        async def _search(query: str, *, max_results: int, include_content: bool):
            started.add(query)
            if len(started) >= 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=1)
            return await make_mock_web_search().search(
                query,
                max_results=max_results,
                include_content=include_content,
            )

        ws.search = AsyncMock(side_effect=_search)
        executor = SearchExecutor(llm, ws, max_query_parallelism=2)
        await executor.search(SubQuestion(question="Q?"), _context())
        assert started == {"q1", "q2"}

    async def test_query_dedupe(self):
        llm = MockLLM(
            responses=[
                '["q1", "q1", "q2"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        executor = SearchExecutor(llm, ws)
        result = await executor.search(SubQuestion(question="Q?"), _context())
        assert ws.search.await_count == 2
        assert result.rounds[0].skipped_queries == 1

    async def test_query_cancel(self):
        llm = MockLLM(
            responses=[
                '["q1", "q2"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        cancelled = asyncio.Event()
        events: list[tuple[str, dict]] = []

        async def _search(query: str, *, max_results: int, include_content: bool):
            if query == "q1":
                return WebSearchResponse(
                    query=query,
                    provider="mock",
                    results=[make_mock_web_search().search.return_value.results[0]],
                    metadata=WebSearchMetadata(
                        cached=False,
                        cache_hit=False,
                        latency_ms=10,
                        provider="mock",
                    ),
                )
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("unreachable")

        async def _on_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        ws.search = AsyncMock(side_effect=_search)
        executor = SearchExecutor(llm, ws, max_query_parallelism=2, on_event=_on_event)
        result = await executor.search(SubQuestion(question="Q?", expected_sources=1), _context())
        assert result.rounds[0].cancelled_queries == 1
        assert cancelled.is_set()
        assert any(name == "search.query_cancelled" for name, _ in events)

    async def test_timing_events(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        events: list[tuple[str, dict]] = []

        async def _on_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        executor = SearchExecutor(llm, ws, on_event=_on_event)
        await executor.search(SubQuestion(question="Q?"), _context())
        event_names = [name for name, _ in events]
        assert "search.query_timing" in event_names
        assert "search.round_timing" in event_names
        assert "search.sufficiency_features" in event_names
        assert "search.sufficiency_decision" in event_names

    async def test_guardrail_sufficient(self):
        llm = MockLLM(responses=['["q1"]'])
        ws = make_mock_web_search(
            results=[
                WebSearchResult(
                    title="AI frameworks official documentation 2026",
                    url="https://docs.example.edu/frameworks",
                    snippet="official AI frameworks documentation 2026",
                    content="official AI frameworks documentation",
                ),
                WebSearchResult(
                    title="AI frameworks benchmark paper 2026",
                    url="https://arxiv.org/abs/1234.5678",
                    snippet="AI frameworks benchmark paper 2026",
                    content="research paper",
                ),
                WebSearchResult(
                    title="AI frameworks comparison 2026",
                    url="https://example.com/compare",
                    snippet="AI frameworks comparison and tradeoffs 2026",
                    content="comparison",
                ),
                WebSearchResult(
                    title="AI frameworks production guide 2026",
                    url="https://vendor.dev/frameworks-guide",
                    snippet="AI frameworks production guide 2026",
                    content="guide",
                ),
            ]
        )
        executor = SearchExecutor(llm, ws)
        result = await executor.search(
            SubQuestion(question="AI frameworks", expected_sources=4),
            _context(),
        )
        assert result.rounds[0].sufficient is True
        assert result.rounds[0].decision_by == "guardrail"
        assert result.rounds[0].reason_code == "guardrail_sufficient"
        assert llm._call_count == 1

    async def test_guardrail_low_diversity(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                '["q2"]',
                '["q3"]',
            ]
        )
        ws = make_mock_web_search(
            results=[
                WebSearchResult(
                    title="AI frameworks official guide",
                    url="https://docs.example.com/frameworks-1",
                    snippet="official AI frameworks guide",
                    content="guide",
                ),
                WebSearchResult(
                    title="AI frameworks architecture",
                    url="https://docs.example.com/frameworks-2",
                    snippet="AI frameworks architecture details",
                    content="architecture",
                ),
                WebSearchResult(
                    title="AI frameworks deployment",
                    url="https://docs.example.com/frameworks-3",
                    snippet="AI frameworks deployment details",
                    content="deployment",
                ),
            ]
        )
        executor = SearchExecutor(llm, ws, max_search_rounds=3)
        result = await executor.search(
            SubQuestion(question="AI frameworks", expected_sources=3), _context()
        )
        assert result.rounds[0].reason_code == "low_diversity"

    async def test_guardrail_low_authority(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                '["q2"]',
                '["q3"]',
            ]
        )
        ws = make_mock_web_search(
            results=[
                WebSearchResult(
                    title="AI frameworks blog comparison",
                    url="https://blog-one.example.com/frameworks",
                    snippet="AI frameworks comparison blog",
                    content="blog",
                ),
                WebSearchResult(
                    title="AI frameworks community notes",
                    url="https://notes.example.org/frameworks",
                    snippet="AI frameworks community notes",
                    content="notes",
                ),
            ]
        )
        executor = SearchExecutor(llm, ws, max_search_rounds=3)
        result = await executor.search(
            SubQuestion(question="AI frameworks", expected_sources=2), _context()
        )
        assert result.rounds[0].reason_code == "low_authority"

    async def test_guardrail_low_recency(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                '["q2"]',
                '["q3"]',
            ]
        )
        ws = make_mock_web_search(
            results=[
                WebSearchResult(
                    title="Current AI frameworks official documentation",
                    url="https://docs.example.edu/frameworks",
                    snippet="official AI frameworks documentation",
                    content="official documentation",
                ),
                WebSearchResult(
                    title="Current AI frameworks benchmark paper",
                    url="https://arxiv.org/abs/1111.2222",
                    snippet="AI frameworks benchmark paper without explicit year",
                    content="paper",
                ),
            ]
        )
        ctx = SearchContext(run_id="r1", plan_id="p1", user_query="latest AI frameworks")
        executor = SearchExecutor(llm, ws, max_search_rounds=3)
        result = await executor.search(
            SubQuestion(question="current AI frameworks", expected_sources=2), ctx
        )
        assert result.rounds[0].reason_code == "low_recency"

    async def test_query_budget_timeout(self):
        llm = MockLLM(
            responses=[
                '["q1"]',
                json.dumps({"sufficient": False, "rationale": "need more"}),
            ]
        )
        ws = make_mock_web_search()
        events: list[tuple[str, dict]] = []

        async def _search(query: str, *, max_results: int, include_content: bool):
            await asyncio.sleep(0.3)
            return await make_mock_web_search().search(
                query,
                max_results=max_results,
                include_content=include_content,
            )

        async def _on_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        ws.search = AsyncMock(side_effect=_search)
        ctx = SearchContext(
            run_id="r1",
            plan_id="p1",
            user_query="AI frameworks",
            max_query_budget_ms=50,
        )
        executor = SearchExecutor(llm, ws, on_event=_on_event)
        result = await executor.search(SubQuestion(question="Q?"), ctx)
        assert result.rounds[0].sufficient is False
        query_events = [data for name, data in events if name == "search.query_timing"]
        assert query_events[0]["reason_code"] == "query_budget_exhausted"

    async def test_round_budget_stop(self):
        llm = MockLLM(responses=['["q1", "q2"]'])
        ws = make_mock_web_search()
        events: list[tuple[str, dict]] = []

        async def _search(query: str, *, max_results: int, include_content: bool):
            await asyncio.sleep(0.3)
            return await make_mock_web_search().search(
                query,
                max_results=max_results,
                include_content=include_content,
            )

        async def _on_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        ws.search = AsyncMock(side_effect=_search)
        ctx = SearchContext(
            run_id="r1",
            plan_id="p1",
            user_query="AI frameworks",
            max_round_budget_ms=50,
            max_query_parallelism=2,
        )
        executor = SearchExecutor(llm, ws, max_query_parallelism=2, on_event=_on_event)
        result = await executor.search(SubQuestion(question="Q?"), ctx)
        assert result.rounds[0].stop_layer == "round"
        assert result.rounds[0].reason_code == "round_budget_exhausted"
        assert any(name == "search.budget_consumed" for name, _ in events)

    async def test_collaboration_prompt(self):
        class _CapturingLLM(MockLLM):
            def __init__(self, responses: list[str]) -> None:
                super().__init__(responses=responses)
                self.prompts: list[str] = []

            async def chat(self, messages: list, **kwargs):
                self.prompts.append(str(messages[0]["content"]))
                return await super().chat(messages, **kwargs)

        llm = _CapturingLLM(
            responses=[
                '["q1"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        snapshots = [
            {
                "peer_findings": ["peer finding"],
                "peer_queries": ["peer query"],
                "peer_gaps": ["missing benchmark"],
                "preferred_providers": ["mock"],
                "shared_source_count": 2,
            }
        ]

        async def _snapshot(round_number: int) -> dict:
            return snapshots[round_number - 1]

        executor = SearchExecutor(llm, ws, get_collaboration_snapshot=_snapshot)
        await executor.search(SubQuestion(question="Q?"), _context())
        assert "Peer findings: peer finding" in llm.prompts[0]
        assert "Peer queries already attempted: peer query" in llm.prompts[0]
        assert "Preferred providers from collaboration: mock" in llm.prompts[0]
        assert "Collaboration summary: shared_sources=2" in llm.prompts[1]

    async def test_collaboration_stop(self):
        llm = MockLLM(responses=[])
        ws = make_mock_web_search()

        async def _snapshot(round_number: int) -> dict:
            return {"stop_reason": "peer already covered it"}

        executor = SearchExecutor(llm, ws, get_collaboration_snapshot=_snapshot)
        result = await executor.search(SubQuestion(question="Q?"), _context())
        assert len(result.rounds) == 1
        assert result.rounds[0].sufficient is True
        assert result.rounds[0].rationale == "peer already covered it"
        assert ws.search.await_count == 0


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
        coord = SearchExecutor(llm, ws)
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
        coord = SearchExecutor(llm, ws)
        result = await coord.search(SubQuestion(question="Q?"), _context())
        assert result.sources == []
        assert len(result.rounds) == 3
        assert result.exhausted is True
        assert result.rounds[0].reason_code == "no_sources"

    async def test_search_call_count_matches(self):
        llm = MockLLM(
            responses=[
                '["q1", "q2"]',
                json.dumps({"sufficient": True, "rationale": "ok"}),
            ]
        )
        ws = make_mock_web_search()
        coord = SearchExecutor(llm, ws)
        await coord.search(SubQuestion(question="Q?"), _context())
        assert ws.search.call_count == 2
