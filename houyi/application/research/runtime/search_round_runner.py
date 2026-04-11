from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from houyi.application.research.runtime.search_budget import _MIN_BUDGET_MS
from houyi.application.research.runtime.search_telemetry import TelemetryEmitter
from houyi.application.research.types import SearchHit, SourceReference
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import WebSearchResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueryExecution:
    query: str
    hits: list[SearchHit]
    elapsed_ms: float
    provider: str
    reason_code: str = ""


@dataclass(slots=True)
class RoundResult:
    hits: list[SearchHit]
    skipped_queries: int
    cancelled_queries: int
    reason_code: str = ""
    stop_layer: str = ""
    stop_reason: str = ""
    executions: list[QueryExecution] = field(default_factory=list)


@dataclass(slots=True)
class RoundRequest:
    question_id: str
    round_index: int
    queries: list[str]
    seen_urls: set[str]
    all_sources: dict[str, SourceReference]
    max_results_per_query: int
    query_parallelism: int
    target_total_sources: int
    query_budget_ms: int
    round_budget_ms: int


@dataclass(slots=True)
class RoundRunner:
    web_search: WebSearchService
    telemetry: TelemetryEmitter
    claim_url: Callable[[str], Awaitable[bool]] | None = None
    check_cancelled: Callable[[], None] | None = None

    async def run(self, request: RoundRequest) -> RoundResult:
        if not request.queries:
            return RoundResult(hits=[], skipped_queries=0, cancelled_queries=0)
        if request.query_parallelism <= 1 or len(request.queries) <= 1:
            return await self._run_serial(request)
        return await self._run_parallel(request)

    async def _run_serial(self, request: RoundRequest) -> RoundResult:
        hits: list[SearchHit] = []
        executions: list[QueryExecution] = []
        cancelled_queries = 0
        round_started = time.perf_counter()
        for index, query in enumerate(request.queries):
            self._run_cancel_check()
            if _remaining_budget_ms(round_started, request.round_budget_ms) <= 0:
                remaining = request.queries[index:]
                cancelled_queries = len(remaining)
                await self.telemetry.query_cancelled(
                    request.question_id,
                    request.round_index,
                    remaining,
                    reason="round_budget_exhausted",
                )
                await self.telemetry.budget_consumed(
                    question_id=request.question_id,
                    round_index=request.round_index,
                    layer="round",
                    reason_code="round_budget_exhausted",
                    budget_ms=request.round_budget_ms,
                    remaining_ms=0,
                )
                return RoundResult(
                    hits=hits,
                    skipped_queries=0,
                    cancelled_queries=cancelled_queries,
                    reason_code="round_budget_exhausted",
                    stop_layer="round",
                    stop_reason="Round budget exhausted before finishing pending queries",
                    executions=executions,
                )
            execution = await self._search_query(
                query,
                request.max_results_per_query,
                request.query_budget_ms,
            )
            executions.append(execution)
            hits.extend(await self._merge_query_hits(request, execution))
            await self.telemetry.query_timing(
                question_id=request.question_id,
                round_index=request.round_index,
                query=execution.query,
                elapsed_ms=execution.elapsed_ms,
                hit_count=len(execution.hits),
                provider=execution.provider,
                reason_code=execution.reason_code,
            )
            if len(request.all_sources) < request.target_total_sources:
                continue
            remaining = request.queries[index + 1 :]
            cancelled_queries = len(remaining)
            await self.telemetry.query_cancelled(
                request.question_id,
                request.round_index,
                remaining,
                reason="source_target_reached",
            )
            break
        return RoundResult(
            hits=hits,
            skipped_queries=0,
            cancelled_queries=cancelled_queries,
            executions=executions,
        )

    async def _run_parallel(self, request: RoundRequest) -> RoundResult:
        semaphore = asyncio.Semaphore(request.query_parallelism)
        round_started = time.perf_counter()

        async def _run_one(query: str) -> QueryExecution:
            async with semaphore:
                return await self._search_query(
                    query,
                    request.max_results_per_query,
                    request.query_budget_ms,
                )

        tasks = {asyncio.create_task(_run_one(query)): query for query in request.queries}
        hits: list[SearchHit] = []
        executions: list[QueryExecution] = []
        cancelled_queries = 0
        pending: set[asyncio.Task[QueryExecution]] = set(tasks)

        while pending:
            self._run_cancel_check()
            remaining_round_ms = _remaining_budget_ms(round_started, request.round_budget_ms)
            if remaining_round_ms <= 0:
                cancelled_queries = await self._cancel_pending_query_tasks(
                    request.question_id,
                    request.round_index,
                    tasks,
                    reason="round_budget_exhausted",
                )
                await self.telemetry.budget_consumed(
                    question_id=request.question_id,
                    round_index=request.round_index,
                    layer="round",
                    reason_code="round_budget_exhausted",
                    budget_ms=request.round_budget_ms,
                    remaining_ms=0,
                )
                await asyncio.gather(*tasks, return_exceptions=True)
                return RoundResult(
                    hits=hits,
                    skipped_queries=0,
                    cancelled_queries=cancelled_queries,
                    reason_code="round_budget_exhausted",
                    stop_layer="round",
                    stop_reason="Round budget exhausted before finishing pending queries",
                    executions=executions,
                )
            done, pending = await asyncio.wait(
                pending,
                timeout=remaining_round_ms / 1000.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue
            for task in done:
                execution = await task
                executions.append(execution)
                hits.extend(await self._merge_query_hits(request, execution))
                await self.telemetry.query_timing(
                    question_id=request.question_id,
                    round_index=request.round_index,
                    query=execution.query,
                    elapsed_ms=execution.elapsed_ms,
                    hit_count=len(execution.hits),
                    provider=execution.provider,
                    reason_code=execution.reason_code,
                )
                if len(request.all_sources) < request.target_total_sources:
                    continue
                cancelled_queries = await self._cancel_pending_query_tasks(
                    request.question_id,
                    request.round_index,
                    tasks,
                    reason="source_target_reached",
                )
                pending = set()
                break

        if cancelled_queries:
            await asyncio.gather(*tasks, return_exceptions=True)
        return RoundResult(
            hits=hits,
            skipped_queries=0,
            cancelled_queries=cancelled_queries,
            executions=executions,
        )

    def _run_cancel_check(self) -> None:
        if self.check_cancelled is not None:
            self.check_cancelled()

    async def _cancel_pending_query_tasks(
        self,
        question_id: str,
        round_index: int,
        tasks: dict[asyncio.Task[QueryExecution], str],
        *,
        reason: str,
    ) -> int:
        pending_queries: list[str] = []
        for pending, pending_query in tasks.items():
            if pending.done():
                continue
            pending.cancel()
            pending_queries.append(pending_query)
        await self.telemetry.query_cancelled(
            question_id,
            round_index,
            pending_queries,
            reason=reason,
        )
        return len(pending_queries)

    async def _merge_query_hits(
        self,
        request: RoundRequest,
        execution: QueryExecution,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for hit in execution.hits:
            if hit.url in request.seen_urls:
                continue
            if self.claim_url is not None and not await self.claim_url(hit.url):
                request.seen_urls.add(hit.url)
                continue
            request.seen_urls.add(hit.url)
            hits.append(hit)
            src = _to_source_ref(hit)
            request.all_sources[src.reference_id] = src
            await self.telemetry.notify(
                "search.source_discovered",
                {
                    "question_id": request.question_id,
                    "title": hit.title,
                    "url": hit.url,
                    "snippet": (hit.snippet or "")[:200],
                    "query": execution.query,
                },
            )
        return hits

    async def _search_query(
        self,
        query: str,
        max_results_per_query: int,
        query_budget_ms: int,
    ) -> QueryExecution:
        started = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                self.web_search.search(
                    query,
                    max_results=max_results_per_query,
                    include_content=True,
                ),
                timeout=max(query_budget_ms, _MIN_BUDGET_MS) / 1000.0,
            )
        except TimeoutError:
            return QueryExecution(
                query=query,
                hits=[],
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                provider="",
                reason_code="query_budget_exhausted",
            )
        except Exception:
            logger.warning("Search query failed: %s", query, exc_info=True)
            return QueryExecution(
                query=query,
                hits=[],
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                provider="",
                reason_code="query_error",
            )
        return QueryExecution(
            query=query,
            hits=[_to_search_hit(result, resp.provider) for result in resp.results],
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            provider=resp.provider,
        )


def _remaining_budget_ms(started: float, budget_ms: int) -> int:
    if budget_ms <= 0:
        return 0
    elapsed_ms = int((time.perf_counter() - started) * 1000.0)
    return max(0, budget_ms - elapsed_ms)


def _to_search_hit(result: WebSearchResult, provider: str) -> SearchHit:
    return SearchHit(
        url=result.url,
        title=result.title,
        snippet=result.snippet,
        content=result.content,
        provider=provider,
        published_at=None,
        rank=0,
    )


def _to_source_ref(hit: SearchHit) -> SourceReference:
    return SourceReference(
        url=hit.url,
        title=hit.title,
        snippet=hit.snippet,
        source_type="web",
        provider=hit.provider,
        reliability_score=0.5,
    )
