"""Research search orchestration core.

This module owns cross-sub-question execution for Deep Research. The
coordinator selects the orchestration strategy for `direct`, `delegate`, and
`autonomous` modes, controls concurrency, emits lifecycle events, and routes
each sub-question into the shared `SearchExecutor` kernel or the agent-based
fallback path when needed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.aggregator import SourceAggregator
from houyi.application.research.runtime.processing import (
    process_agent_search_output,
    process_sub_question_execution_order,
)
from houyi.application.research.runtime.report_pipeline import ReportPipeline
from houyi.application.research.runtime.search_executor import SearchExecutor
from houyi.application.research.runtime.tools import WebSearchTool
from houyi.application.research.types import (
    AggregatedSources,
    OrchestrationMode,
    ResearchPlan,
    ResearchSettings,
    SearchContext,
    SearchResult,
)
from houyi.application.runtime.agent import Agent
from houyi.application.runtime.events import EventEmitter
from houyi.application.runtime.shared_state import SharedStateBackend
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)

_SEARCHER_SYSTEM_PROMPT = """\
You are a research search agent. Your job is to search the web for a specific sub-question and return structured, well-analyzed results.

Search strategy:
1. First search: broad query covering the main topic.
2. Review results. If key information is missing, refine your query.
3. Continue searching with different angles until you have comprehensive coverage.
4. Use up to 3-5 searches maximum. Stop when you have enough quality sources.
5. Focus on authoritative, recent sources. Prefer primary sources over summaries.

IMPORTANT:
- Analyze and synthesize across sources — don't just list them.
- Cross-reference facts between sources for accuracy.
- Note contradictions or uncertainties in your summary.

When finished, respond with ONLY a JSON object (no markdown fences):
{
  "sources": [
    {"url": "...", "title": "...", "snippet": "...", "content_summary": "..."}
  ],
  "summary": "Comprehensive analysis synthesizing findings for this sub-question",
  "queries_used": ["query1", "query2"]
}
"""


@dataclass(slots=True)
class CoordinatorServices:
    """Immutable runtime dependencies injected into `ResearchCoordinator`.

    This keeps the coordinator itself focused on orchestration logic instead
    of construction logic, and makes the execution contract explicit at the
    top-level runtime composition boundary.
    """

    run_id: str
    llm_adapter: LLMAdapter
    web_search: WebSearchService
    web_search_tool: WebSearchTool
    settings: ResearchSettings
    shared_state: SharedStateBackend
    aggregator: SourceAggregator
    report_pipeline: ReportPipeline
    search_executor: SearchExecutor
    search_event_handler: Callable[[str, dict[str, Any]], Awaitable[None]]
    event_emitter: EventEmitter | None
    emit: Callable[..., Awaitable[None]]
    emit_restored_search_events: Callable[[str, SearchResult], Awaitable[None]]
    check_cancelled: Callable[[], None]
    elapsed_seconds: Callable[[], float]
    agent_timeout_seconds: Callable[[], int]
    llm_kwargs: dict[str, Any]


@dataclass(slots=True)
class SearchPhaseResult:
    """Search-phase outputs produced by `ResearchCoordinator.run()`."""

    search_results: list[SearchResult]
    aggregated_sources: AggregatedSources
    intermediate_reports: list[Any]
    search_elapsed_ms: float
    aggregate_ms: float = 0.0
    intermediate_ms: float = 0.0


@dataclass(slots=True)
class _AutonomousCollaborationState:
    lock: asyncio.Lock
    seen_queries: set[str]
    seen_urls: set[str]
    findings_by_question: dict[str, list[dict[str, str]]]
    provider_successes: dict[str, int]
    provider_failures: dict[str, int]
    gaps_by_question: dict[str, list[str]]

    async def claim_query(self, query: str) -> bool:
        async with self.lock:
            if query in self.seen_queries:
                return False
            self.seen_queries.add(query)
            return True

    async def claim_url(self, url: str) -> bool:
        async with self.lock:
            if url in self.seen_urls:
                return False
            self.seen_urls.add(url)
            return True

    async def record_query_outcome(self, provider: str, hit_count: int) -> None:
        provider_name = (provider or "").strip().lower()
        if not provider_name:
            return
        async with self.lock:
            target = self.provider_successes if hit_count > 0 else self.provider_failures
            target[provider_name] = target.get(provider_name, 0) + 1

    async def record_findings(self, question_id: str, findings: list[dict[str, str]]) -> None:
        if not findings:
            return
        async with self.lock:
            existing = self.findings_by_question.setdefault(question_id, [])
            seen = {
                (item.get("url", ""), item.get("snippet", ""), item.get("title", ""))
                for item in existing
            }
            for finding in findings:
                key = (
                    finding.get("url", ""),
                    finding.get("snippet", ""),
                    finding.get("title", ""),
                )
                if key in seen:
                    continue
                existing.append(finding)
                seen.add(key)

    async def record_gaps(self, question_id: str, gaps: list[str]) -> None:
        async with self.lock:
            cleaned = [gap.strip() for gap in gaps if gap and gap.strip()]
            self.gaps_by_question[question_id] = cleaned

    async def snapshot(
        self, *, question_id: str, expected_sources: int, round_number: int
    ) -> dict[str, Any]:
        async with self.lock:
            peer_findings = [
                finding.get("snippet", "")
                for owner, findings in self.findings_by_question.items()
                if owner != question_id
                for finding in findings
                if finding.get("snippet")
            ]
            peer_queries = sorted(self.seen_queries)
            peer_gaps = [
                gap
                for owner, gaps in self.gaps_by_question.items()
                if owner != question_id
                for gap in gaps
            ]
            preferred_providers = _rank_provider_preferences(
                self.provider_successes,
                self.provider_failures,
            )
            stop_reason = None
            if (
                round_number > 1
                and len(self.seen_urls) >= max(expected_sources, 1)
                and not peer_gaps
            ):
                stop_reason = (
                    "Peer collaboration already covers enough sources; stop duplicate search"
                )
            return {
                "peer_findings": peer_findings,
                "peer_queries": peer_queries,
                "peer_gaps": peer_gaps,
                "preferred_providers": preferred_providers,
                "shared_source_count": len(self.seen_urls),
                "stop_reason": stop_reason,
                "provider_successes": dict(self.provider_successes),
                "provider_failures": dict(self.provider_failures),
            }


class ResearchCoordinator:
    """Top-level orchestrator for the research search phase.

    Responsibilities:
    - choose mode-specific sub-question execution strategy
    - enforce cross-question concurrency limits
    - preserve retry / checkpoint semantics
    - bridge shared-state collaboration for `autonomous`
    - degrade to the agent fallback path when the core search executor fails
    """

    def __init__(self, deps: CoordinatorServices) -> None:
        self._deps = deps
        self._plan: ResearchPlan | None = None
        self._retry_checkpoint: dict[str, SearchResult] = {}
        self._intermediate_reuse_by_qid: dict[str, Any] = {}
        self._search_results: list[SearchResult] = []

    async def run(
        self,
        *,
        plan: ResearchPlan,
        retry_checkpoint: dict[str, SearchResult],
        intermediate_reuse_by_qid: dict[str, Any],
    ) -> SearchPhaseResult:
        self._plan = plan
        self._retry_checkpoint = retry_checkpoint
        self._intermediate_reuse_by_qid = intermediate_reuse_by_qid
        self._search_results = []

        t_search = time.perf_counter()
        questions = process_sub_question_execution_order(plan.sub_questions)
        mode = self._deps.settings.orchestration_mode

        search_coro = self._dispatch_search(mode, questions)
        global_cap = self._deps.settings.max_search_phase_seconds
        if global_cap > 0:
            try:
                await asyncio.wait_for(search_coro, timeout=global_cap)
            except TimeoutError:
                # Salvage: self._search_results already contains results
                # for sub-questions that finished before the deadline.
                logger.warning(
                    "Global search phase cap (%ds) exceeded; salvaging %d/%d results",
                    global_cap,
                    len(self._search_results),
                    len(questions),
                )
        else:
            await search_coro

        t_aggregate = time.perf_counter()
        aggregated = await self._deps.aggregator.aggregate(
            self._search_results,
            user_query=self._plan.query if self._plan else "",
        )
        aggregate_ms = (time.perf_counter() - t_aggregate) * 1000.0

        intermediate_reports: list[Any] = []
        intermediate_ms = 0.0
        if self._deps.settings.depth in ("standard", "deep"):
            t_intermediate = time.perf_counter()
            intermediate_reports = await self._deps.report_pipeline.generate_intermediates(
                self._search_results,
                questions,
                plan.query,
                depth=(
                    self._deps.settings.depth.value
                    if hasattr(self._deps.settings.depth, "value")
                    else str(self._deps.settings.depth)
                ),
                reuse_by_question_id=self._intermediate_reuse_by_qid,
                checkpoint_question_ids=frozenset(self._retry_checkpoint.keys()),
            )
            intermediate_ms = (time.perf_counter() - t_intermediate) * 1000.0

        search_elapsed_ms = (time.perf_counter() - t_search) * 1000.0
        per_q_ms = [sr.elapsed_ms for sr in self._search_results if sr.elapsed_ms > 0]
        per_q_summary = ""
        if per_q_ms:
            per_q_summary = (
                f" per_question_ms: avg={sum(per_q_ms) / len(per_q_ms):.0f}"
                f" min={min(per_q_ms):.0f} max={max(per_q_ms):.0f}"
                f" p50={sorted(per_q_ms)[len(per_q_ms) // 2]:.0f}"
            )
        logger.info(
            "Research search completed in %.2fs (aggregate=%.2fms, intermediate=%.2fms) for %d questions.%s",
            search_elapsed_ms / 1000.0,
            aggregate_ms,
            intermediate_ms,
            len(plan.sub_questions),
            per_q_summary,
        )
        return SearchPhaseResult(
            search_results=list(self._search_results),
            aggregated_sources=aggregated,
            intermediate_reports=intermediate_reports,
            search_elapsed_ms=round(search_elapsed_ms, 1),
            aggregate_ms=round(aggregate_ms, 1),
            intermediate_ms=round(intermediate_ms, 1),
        )

    async def _dispatch_search(self, mode: OrchestrationMode, questions: list[Any]) -> None:
        """Route to the mode-specific sub-question execution strategy."""
        if mode == OrchestrationMode.DELEGATE:
            await self._search_delegate(questions)
        elif mode == OrchestrationMode.AUTONOMOUS:
            await self._search_autonomous(questions)
        else:
            await self._search_direct(questions)

    async def _search_direct(self, questions: list[Any]) -> None:
        total = len(questions)
        pending = [
            question for question in questions if question.question_id not in self._retry_checkpoint
        ]

        for sub_question in questions:
            cached = self._retry_checkpoint.get(sub_question.question_id)
            if cached:
                await self._reuse_checkpoint(sub_question, cached, total)

        semaphore = asyncio.Semaphore(self._deps.settings.max_agents)

        async def _run_one(sub_question: Any) -> SearchResult:
            async with semaphore:
                self._deps.check_cancelled()
                await self._deps.emit(
                    "research.step_started",
                    step_id=sub_question.question_id,
                    step=sub_question.question[:60],
                    total_steps=total,
                    completed_steps=len(self._search_results),
                    elapsed_seconds=self._deps.elapsed_seconds(),
                )
                result = await self._search_question_isolated(sub_question)
                await self._deps.emit(
                    "research.step_completed",
                    step_id=sub_question.question_id,
                    step=sub_question.question[:60],
                    total_steps=total,
                    sources=len(result.sources),
                    elapsed_seconds=self._deps.elapsed_seconds(),
                )
                return result

        results = await asyncio.gather(
            *[_run_one(item) for item in pending], return_exceptions=True
        )
        for sub_question, result in zip(pending, results, strict=True):
            self._search_results.append(self._normalize_result(sub_question, result, "DIRECT"))

    async def _seed_dedup_from_checkpoints(
        self, questions: list[Any], dedup: _DelegateDedup
    ) -> None:
        """Pre-seed dedup tracker with URLs/queries from cached checkpoints."""
        for sub_question in questions:
            cached = self._retry_checkpoint.get(sub_question.question_id)
            if not cached:
                continue
            for search_round in cached.rounds:
                for query in search_round.queries:
                    normalized = _canonical_collaboration_query(query)
                    if normalized:
                        await dedup.claim_query(normalized)
            for source in cached.sources:
                if source.url:
                    await dedup.claim_url(source.url)

    async def _search_delegate(self, questions: list[Any]) -> None:
        total = len(questions)
        pending = [
            question for question in questions if question.question_id not in self._retry_checkpoint
        ]

        for sub_question in pending:
            await self._deps.emit(
                "research.agent_spawned",
                agent_id=f"{self._deps.run_id}_{sub_question.question_id}",
                question_id=sub_question.question_id,
                agent_name=sub_question.question[:60],
                task=sub_question.question,
            )

        for sub_question in questions:
            cached = self._retry_checkpoint.get(sub_question.question_id)
            if cached:
                await self._reuse_checkpoint(sub_question, cached, total)

        # Lightweight cross-sub-question dedup: prevents identical URLs and
        # queries from being fetched by multiple sub-questions in parallel.
        dedup = _DelegateDedup()
        await self._seed_dedup_from_checkpoints(questions, dedup)

        semaphore = asyncio.Semaphore(self._deps.settings.max_agents)

        async def _run_one(sub_question: Any) -> SearchResult:
            async with semaphore:
                self._deps.check_cancelled()
                await self._deps.emit(
                    "research.step_started",
                    step_id=sub_question.question_id,
                    step=sub_question.question[:60],
                    total_steps=total,
                    completed_steps=len(self._search_results),
                    elapsed_seconds=self._deps.elapsed_seconds(),
                )
                result = await self._search_question_isolated(
                    sub_question,
                    claim_query=dedup.claim_query,
                    claim_url=dedup.claim_url,
                )
                await self._deps.emit(
                    "research.step_completed",
                    step_id=sub_question.question_id,
                    step=sub_question.question[:60],
                    total_steps=total,
                    sources=len(result.sources),
                    elapsed_seconds=self._deps.elapsed_seconds(),
                )
                return result

        results = await asyncio.gather(
            *[_run_one(item) for item in pending], return_exceptions=True
        )
        for sub_question, result in zip(pending, results, strict=True):
            self._search_results.append(self._normalize_result(sub_question, result, "DELEGATE"))

    async def _search_autonomous(self, questions: list[Any]) -> None:
        total = len(questions)
        state_id = f"research_{self._deps.run_id}"
        collaboration = _AutonomousCollaborationState(
            lock=asyncio.Lock(),
            seen_queries=set(),
            seen_urls=set(),
            findings_by_question={},
            provider_successes={},
            provider_failures={},
            gaps_by_question={},
        )
        pending = [
            question for question in questions if question.question_id not in self._retry_checkpoint
        ]

        await self._emit_autonomous_spawns(pending)
        await self._seed_autonomous_collaboration(
            questions=questions,
            total=total,
            state_id=state_id,
            collaboration=collaboration,
        )

        semaphore = asyncio.Semaphore(self._deps.settings.max_agents)

        async def _run_one(sub_question: Any) -> SearchResult:
            async with semaphore:
                return await self._run_autonomous_question(
                    sub_question=sub_question,
                    total=total,
                    state_id=state_id,
                    collaboration=collaboration,
                )

        results = await asyncio.gather(
            *[_run_one(item) for item in pending], return_exceptions=True
        )
        for sub_question, result in zip(pending, results, strict=True):
            normalized = self._normalize_result(sub_question, result, "AUTONOMOUS")
            self._search_results.append(normalized)
            await self._deps.emit(
                "research.agent_completed",
                agent_id=f"{self._deps.run_id}_{sub_question.question_id}",
                question_id=sub_question.question_id,
                status="completed" if normalized.sources else "failed",
                summary=normalized.summary[:200],
            )

    async def _emit_autonomous_spawns(self, pending: list[Any]) -> None:
        for sub_question in pending:
            await self._deps.emit(
                "research.agent_spawned",
                agent_id=f"{self._deps.run_id}_{sub_question.question_id}",
                question_id=sub_question.question_id,
                agent_name=sub_question.question[:60],
                task=sub_question.question,
            )

    async def _seed_autonomous_collaboration(
        self,
        *,
        questions: list[Any],
        total: int,
        state_id: str,
        collaboration: _AutonomousCollaborationState,
    ) -> None:
        for sub_question in questions:
            cached = self._retry_checkpoint.get(sub_question.question_id)
            if not cached:
                continue
            await self._reuse_checkpoint(sub_question, cached, total)
            await self._claim_cached_round_queries(cached, collaboration)
            await self._claim_cached_source_urls(cached, collaboration)
            await self._record_autonomous_result(collaboration, sub_question.question_id, cached)
            await self._write_shared_findings(state_id, sub_question.question_id, cached.sources)
            await self._write_shared_metadata(state_id, collaboration)

    async def _claim_cached_round_queries(
        self,
        cached: SearchResult,
        collaboration: _AutonomousCollaborationState,
    ) -> None:
        for search_round in cached.rounds:
            for query in search_round.queries:
                normalized = _canonical_collaboration_query(query)
                if normalized:
                    await collaboration.claim_query(normalized)

    async def _claim_cached_source_urls(
        self,
        cached: SearchResult,
        collaboration: _AutonomousCollaborationState,
    ) -> None:
        for source in cached.sources:
            if source.url:
                await collaboration.claim_url(source.url)

    async def _run_autonomous_question(
        self,
        *,
        sub_question: Any,
        total: int,
        state_id: str,
        collaboration: _AutonomousCollaborationState,
    ) -> SearchResult:
        self._deps.check_cancelled()
        await self._deps.emit(
            "research.step_started",
            step_id=sub_question.question_id,
            step=sub_question.question[:60],
            total_steps=total,
            completed_steps=len(self._search_results),
            elapsed_seconds=self._deps.elapsed_seconds(),
        )
        peer_findings = await self._read_peer_findings(state_id, sub_question.question_id)
        result = await self._search_question_isolated(
            sub_question,
            peer_findings=peer_findings,
            claim_query=collaboration.claim_query,
            claim_url=collaboration.claim_url,
            on_event=self._build_autonomous_event_handler(collaboration),
            get_collaboration_snapshot=lambda round_number: collaboration.snapshot(
                question_id=sub_question.question_id,
                expected_sources=sub_question.expected_sources,
                round_number=round_number,
            ),
        )
        await self._record_autonomous_result(collaboration, sub_question.question_id, result)
        await self._write_shared_findings(state_id, sub_question.question_id, result.sources)
        await self._write_shared_metadata(state_id, collaboration)
        await self._deps.emit(
            "research.step_completed",
            step_id=sub_question.question_id,
            step=sub_question.question[:60],
            total_steps=total,
            sources=len(result.sources),
            elapsed_seconds=self._deps.elapsed_seconds(),
        )
        return result

    async def _read_peer_findings(self, state_id: str, question_id: str) -> list[str]:
        state = await self._deps.shared_state.read(state_id)
        return [
            item.get("snippet", "")
            for item in state.findings
            if item.get("question_id") != question_id and item.get("snippet")
        ]

    async def _write_shared_findings(
        self,
        state_id: str,
        question_id: str,
        sources: list[Any],
    ) -> None:
        findings = [
            {"question_id": question_id, "snippet": source.snippet}
            for source in sources
            if source.snippet
        ]
        if findings:
            await self._deps.shared_state.write(state_id, {"findings": findings})

    async def _write_shared_metadata(
        self,
        state_id: str,
        collaboration: _AutonomousCollaborationState,
    ) -> None:
        snapshot = await collaboration.snapshot(question_id="", expected_sources=1, round_number=1)
        metadata = {
            "query_ledger": snapshot.get("peer_queries", []),
            "preferred_providers": snapshot.get("preferred_providers", []),
            "provider_successes": snapshot.get("provider_successes", {}),
            "provider_failures": snapshot.get("provider_failures", {}),
            "peer_gaps": snapshot.get("peer_gaps", []),
            "shared_source_count": snapshot.get("shared_source_count", 0),
        }
        await self._deps.shared_state.write(state_id, {"metadata": metadata})

    async def _record_autonomous_result(
        self,
        collaboration: _AutonomousCollaborationState,
        question_id: str,
        result: SearchResult,
    ) -> None:
        findings = [
            {
                "url": source.url or "",
                "title": source.title,
                "snippet": source.snippet,
            }
            for source in result.sources
            if source.snippet or source.url or source.title
        ]
        await collaboration.record_findings(question_id, findings)
        gaps = [
            search_round.rationale
            for search_round in result.rounds
            if not search_round.sufficient and search_round.rationale
        ]
        await collaboration.record_gaps(question_id, gaps)

    def _build_autonomous_event_handler(
        self,
        collaboration: _AutonomousCollaborationState,
    ) -> Callable[[str, dict[str, Any]], Awaitable[None]]:
        async def _handler(event_type: str, data: dict[str, Any]) -> None:
            if event_type == "search.query_timing":
                await collaboration.record_query_outcome(
                    str(data.get("provider", "")),
                    int(data.get("hit_count", 0) or 0),
                )
            await self._deps.search_event_handler(event_type, data)

        return _handler

    async def _reuse_checkpoint(self, sub_question: Any, cached: SearchResult, total: int) -> None:
        self._search_results.append(cached)
        logger.info("Retry: reusing checkpoint for %s", sub_question.question_id)
        await self._deps.emit(
            "research.step_started",
            step_id=sub_question.question_id,
            step=sub_question.question[:60],
            total_steps=total,
            completed_steps=len(self._search_results) - 1,
            elapsed_seconds=self._deps.elapsed_seconds(),
            reused=True,
        )
        await self._deps.emit_restored_search_events(sub_question.question_id, cached)
        await self._deps.emit(
            "research.step_completed",
            step_id=sub_question.question_id,
            step=sub_question.question[:60],
            total_steps=total,
            completed_steps=len(self._search_results),
            sources=len(cached.sources),
            elapsed_seconds=self._deps.elapsed_seconds(),
            reused=True,
        )

    def _normalize_result(
        self, sub_question: Any, result: SearchResult | BaseException, mode: str
    ) -> SearchResult:
        if isinstance(result, BaseException):
            logger.warning("%s search failed for %s: %s", mode, sub_question.question_id, result)
            return SearchResult(
                question_id=sub_question.question_id,
                rounds=[],
                sources=[],
                summary=f"Search failed: {result}",
                coverage_score=0.0,
            )
        return result

    async def _search_question(self, sub_question: Any) -> SearchResult:
        self._deps.check_cancelled()
        assert self._plan is not None
        sub_question_timeout_s = self._sub_question_timeout_seconds(sub_question)
        t0 = time.perf_counter()
        try:
            context = SearchContext(
                run_id=self._deps.run_id,
                plan_id=self._plan.plan_id,
                user_query=self._plan.query,
                prior_findings=[
                    result.summary for result in self._search_results if result.summary
                ],
                excluded_urls=[
                    source.url
                    for result in self._search_results
                    for source in result.sources
                    if source.url
                ],
                max_sub_question_budget_ms=int(sub_question_timeout_s * 1000),
                salvage_on_cancel=True,
            )
            result = await asyncio.wait_for(
                self._deps.search_executor.search(sub_question, context),
                timeout=sub_question_timeout_s,
            )
            result.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return result
        except TimeoutError:
            logger.warning(
                "SearchExecutor timed out after %ds for %s",
                sub_question_timeout_s,
                sub_question.question_id,
            )
            return SearchResult(
                question_id=sub_question.question_id,
                rounds=[],
                sources=[],
                summary="Search timed out",
                coverage_score=0.0,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
        except Exception:
            logger.warning(
                "SearchExecutor failed for %s, falling back to agent",
                sub_question.question_id,
                exc_info=True,
            )
            result = await self._search_question_fallback(sub_question)
            result.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return result

    async def _search_question_fallback(self, sub_question: Any) -> SearchResult:
        agent = Agent(
            role=f"searcher_{sub_question.question_id}",
            llm=self._deps.llm_adapter,
            system_prompt=_SEARCHER_SYSTEM_PROMPT,
            tools=[self._deps.web_search_tool],
            event_emitter=self._deps.event_emitter,
            max_turns=self._deps.settings.max_search_rounds * 3 + 1,
        )
        sub_question_timeout_s = self._sub_question_timeout_seconds(sub_question)
        try:
            output = await asyncio.wait_for(
                agent.arun(self._build_search_task(sub_question)),
                timeout=sub_question_timeout_s,
            )
        except TimeoutError:
            output = '{"sources": [], "summary": "Search timed out", "queries_used": []}'
        return process_agent_search_output(sub_question, output)

    async def _search_question_isolated(
        self,
        sub_question: Any,
        *,
        peer_findings: list[str] | None = None,
        claim_query: Callable[[str], Awaitable[bool]] | None = None,
        claim_url: Callable[[str], Awaitable[bool]] | None = None,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        get_collaboration_snapshot: Callable[[int], Awaitable[dict[str, Any]]] | None = None,
    ) -> SearchResult:
        self._deps.check_cancelled()
        assert self._plan is not None
        sub_question_timeout_s = self._sub_question_timeout_seconds(sub_question)
        t0 = time.perf_counter()
        search_executor = SearchExecutor(
            self._deps.llm_adapter,
            self._deps.web_search,
            max_search_rounds=self._deps.settings.max_search_rounds,
            on_event=on_event or self._deps.search_event_handler,
            claim_query=claim_query,
            claim_url=claim_url,
            check_cancelled=self._deps.check_cancelled,
            get_collaboration_snapshot=get_collaboration_snapshot,
            **self._deps.llm_kwargs,
        )
        context = SearchContext(
            run_id=self._deps.run_id,
            plan_id=self._plan.plan_id,
            user_query=self._plan.query,
            prior_findings=peer_findings or [],
            excluded_urls=[],
            max_sub_question_budget_ms=int(sub_question_timeout_s * 1000),
            salvage_on_cancel=True,
        )
        try:
            result = await asyncio.wait_for(
                search_executor.search(sub_question, context),
                timeout=sub_question_timeout_s,
            )
            result.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return result
        except TimeoutError:
            logger.warning(
                "Isolated SC timed out after %ds for %s",
                sub_question_timeout_s,
                sub_question.question_id,
            )
            return SearchResult(
                question_id=sub_question.question_id,
                rounds=[],
                sources=[],
                summary="Search timed out",
                coverage_score=0.0,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
        except Exception:
            logger.warning(
                "Isolated SC failed for %s, falling back to agent",
                sub_question.question_id,
                exc_info=True,
            )
            result = await self._search_question_fallback(sub_question)
            result.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return result

    def _build_search_task(self, sub_question: Any) -> str:
        assert self._plan is not None
        return (
            f"Research the following sub-question thoroughly:\n\n"
            f"Sub-question: {sub_question.question}\n"
            f"Overall research topic: {self._plan.query}\n"
            f"Priority: {sub_question.priority} (higher = more important)\n"
            f"Expected sources: {sub_question.expected_sources}\n\n"
            f"Search the web and collect relevant, diverse sources."
        )

    def _sub_question_timeout_seconds(self, sub_question: Any) -> int:
        configured = int(self._deps.agent_timeout_seconds())
        rounds = max(1, int(self._deps.settings.max_search_rounds))
        expected_sources = max(1, int(getattr(sub_question, "expected_sources", 1) or 1))
        # Bench evidence: a flat 300s isolated sub-question cap creates long-tail
        # stalls where one low-yield SC blocks a whole delegate batch. We keep a
        # bounded dynamic cap that scales with planned depth but still converges.
        derived = (rounds * 30) + (min(expected_sources, 6) * 10)
        return max(45, min(configured, 180, derived))


@dataclass(slots=True)
class _DelegateDedup:
    """Lightweight cross-sub-question dedup for delegate orchestration mode.

    Prevents identical URLs and queries from being fetched by multiple
    sub-questions running in parallel without the full collaboration
    overhead of autonomous mode.
    """

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _seen_queries: set[str] = field(default_factory=set)
    _seen_urls: set[str] = field(default_factory=set)

    async def claim_query(self, query: str) -> bool:
        normalized = _canonical_collaboration_query(query)
        if not normalized:
            return False
        async with self._lock:
            if normalized in self._seen_queries:
                return False
            self._seen_queries.add(normalized)
            return True

    async def claim_url(self, url: str) -> bool:
        async with self._lock:
            if url in self._seen_urls:
                return False
            self._seen_urls.add(url)
            return True


def _canonical_collaboration_query(query: str) -> str:
    return " ".join(query.split()).strip().lower()


def _rank_provider_preferences(
    successes: dict[str, int],
    failures: dict[str, int],
) -> list[str]:
    providers = set(successes) | set(failures)
    ranked = sorted(
        providers,
        key=lambda item: (
            -(successes.get(item, 0) - failures.get(item, 0)),
            -successes.get(item, 0),
            item,
        ),
    )
    return [provider for provider in ranked if successes.get(provider, 0) > 0][:3]
