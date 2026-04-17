from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from houyi.application.research.runtime.search_round_runner import (
    _MAX_SOURCES_PER_DOMAIN_PER_ROUND,
    RoundRequest,
    RoundRunner,
    _extract_domain,
)
from houyi.application.research.runtime.search_telemetry import TelemetryEmitter
from houyi.skills.web_search.types import WebSearchMetadata, WebSearchResponse

from ..conftest import make_mock_web_search


def _noop_telemetry() -> TelemetryEmitter:
    async def _notify(event_type: str, data: dict) -> None:
        pass

    return TelemetryEmitter(notify=_notify)


def _request(
    queries: list[str],
    *,
    parallelism: int = 1,
    round_budget_ms: int = 60_000,
    query_budget_ms: int = 45_000,
    target: int = 10,
) -> RoundRequest:
    return RoundRequest(
        question_id="q1",
        round_index=1,
        queries=queries,
        seen_urls=set(),
        all_sources={},
        max_results_per_query=4,
        query_parallelism=parallelism,
        target_total_sources=target,
        query_budget_ms=query_budget_ms,
        round_budget_ms=round_budget_ms,
    )


class TestRoundRunner:
    async def test_empty_queries_returns_empty(self):
        runner = RoundRunner(web_search=make_mock_web_search(), telemetry=_noop_telemetry())
        result = await runner.run(_request([]))
        assert result.hits == []
        assert result.cancelled_queries == 0

    async def test_serial_collects_hits(self):
        runner = RoundRunner(web_search=make_mock_web_search(), telemetry=_noop_telemetry())
        result = await runner.run(_request(["q1"]))
        assert len(result.hits) >= 1

    async def test_parallel_runs_concurrently(self):
        started: set[str] = set()
        release = asyncio.Event()

        async def _search(query, *, max_results, include_content):
            started.add(query)
            if len(started) >= 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=1)
            return await make_mock_web_search().search(
                query, max_results=max_results, include_content=include_content
            )

        ws = make_mock_web_search()
        ws.search = AsyncMock(side_effect=_search)
        runner = RoundRunner(web_search=ws, telemetry=_noop_telemetry())
        await runner.run(_request(["q1", "q2"], parallelism=2))
        assert started == {"q1", "q2"}

    async def test_round_budget_stops_serial(self):
        events: list[str] = []

        async def _slow(query, *, max_results, include_content):
            await asyncio.sleep(0.3)
            return await make_mock_web_search().search(
                query, max_results=max_results, include_content=include_content
            )

        captured: list[tuple[str, dict]] = []

        async def _notify(event_type: str, data: dict) -> None:
            captured.append((event_type, data))

        ws = make_mock_web_search()
        ws.search = AsyncMock(side_effect=_slow)
        runner = RoundRunner(web_search=ws, telemetry=TelemetryEmitter(notify=_notify))
        result = await runner.run(_request(["q1", "q2"], round_budget_ms=50))
        assert result.stop_layer == "round"
        assert result.reason_code == "round_budget_exhausted"

    async def test_source_target_cancels_remaining(self):
        captured: list[tuple[str, dict]] = []

        async def _notify(event_type: str, data: dict) -> None:
            captured.append((event_type, data))

        runner = RoundRunner(
            web_search=make_mock_web_search(),
            telemetry=TelemetryEmitter(notify=_notify),
        )
        result = await runner.run(_request(["q1", "q2"], target=1))
        assert result.cancelled_queries >= 1
        assert any(e == "search.query_cancelled" for e, _ in captured)

    async def test_timeout_error(self):
        ws = make_mock_web_search(results=[])
        ws.search = AsyncMock(
            return_value=WebSearchResponse(
                query="q1",
                provider="serper",
                results=[],
                metadata=WebSearchMetadata(
                    cached=False,
                    cache_hit=False,
                    latency_ms=10,
                    provider="serper",
                    errors=[
                        {
                            "type": "ProviderTimeoutError",
                            "message": "timeout",
                            "provider": "serper",
                        }
                    ],
                ),
            )
        )
        runner = RoundRunner(web_search=ws, telemetry=_noop_telemetry())
        result = await runner.run(_request(["q1"]))
        assert len(result.executions) == 1
        assert result.executions[0].reason_code == "empty_with_errors"

    def test_extract_domain_strips_www(self):
        assert _extract_domain("https://www.example.com/path") == "example.com"

    def test_extract_domain_keeps_subdomain(self):
        assert _extract_domain("https://blog.example.com") == "blog.example.com"

    def test_extract_domain_handles_empty(self):
        assert _extract_domain("") == ""

    def test_domain_limit_is_positive(self):
        assert _MAX_SOURCES_PER_DOMAIN_PER_ROUND >= 2
