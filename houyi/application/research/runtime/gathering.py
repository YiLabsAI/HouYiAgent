from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.aggregator import SourceAggregator
from houyi.application.research.runtime.processing import (
    process_agent_search_output,
    process_sub_question_execution_order,
)
from houyi.application.research.runtime.report_pipeline import ReportPipeline
from houyi.application.research.runtime.search import SearchCoordinator
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
class GatheringDeps:
    run_id: str
    llm_adapter: LLMAdapter
    web_search: WebSearchService
    web_search_tool: WebSearchTool
    settings: ResearchSettings
    shared_state: SharedStateBackend
    aggregator: SourceAggregator
    report_pipeline: ReportPipeline
    search_coordinator: SearchCoordinator
    search_event_handler: Callable[[str, dict[str, Any]], Awaitable[None]]
    event_emitter: EventEmitter | None
    emit: Callable[..., Awaitable[None]]
    emit_restored_search_events: Callable[[str, SearchResult], Awaitable[None]]
    check_cancelled: Callable[[], None]
    elapsed_seconds: Callable[[], float]
    agent_timeout_seconds: Callable[[], int]
    llm_kwargs: dict[str, Any]


@dataclass(slots=True)
class GatheringResult:
    search_results: list[SearchResult]
    aggregated_sources: AggregatedSources
    intermediate_reports: list[Any]
    search_elapsed_ms: float


class GatheringCoordinator:
    def __init__(self, deps: GatheringDeps) -> None:
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
    ) -> GatheringResult:
        self._plan = plan
        self._retry_checkpoint = retry_checkpoint
        self._intermediate_reuse_by_qid = intermediate_reuse_by_qid
        self._search_results = []

        t_search = time.perf_counter()
        questions = process_sub_question_execution_order(plan.sub_questions)
        mode = self._deps.settings.orchestration_mode

        if mode == OrchestrationMode.DELEGATE:
            await self._search_delegate(questions)
        elif mode == OrchestrationMode.AUTONOMOUS:
            await self._search_autonomous(questions)
        else:
            await self._search_direct(questions)

        aggregated = await self._deps.aggregator.aggregate(self._search_results)
        intermediate_reports: list[Any] = []
        if self._deps.settings.depth in ("standard", "deep"):
            intermediate_reports = await self._deps.report_pipeline.generate_intermediates(
                self._search_results,
                questions,
                plan.query,
                reuse_by_question_id=self._intermediate_reuse_by_qid,
                checkpoint_question_ids=frozenset(self._retry_checkpoint.keys()),
            )

        depth = self._deps.settings.depth
        depth_s = depth.value if hasattr(depth, "value") else str(depth)
        search_elapsed_ms = (time.perf_counter() - t_search) * 1000.0
        logger.info(
            "research.phase.search_done run_id=%s depth=%s elapsed_s=%.2f sub_questions=%d",
            self._deps.run_id,
            depth_s,
            search_elapsed_ms / 1000.0,
            len(plan.sub_questions),
        )
        return GatheringResult(
            search_results=list(self._search_results),
            aggregated_sources=aggregated,
            intermediate_reports=intermediate_reports,
            search_elapsed_ms=round(search_elapsed_ms, 1),
        )

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

    async def _search_delegate(self, questions: list[Any]) -> None:
        total = len(questions)
        pending = [
            question for question in questions if question.question_id not in self._retry_checkpoint
        ]

        for sub_question in pending:
            await self._deps.emit(
                "research.agent_spawned",
                agent_id=f"{self._deps.run_id}_{sub_question.question_id}",
                agent_name=sub_question.question[:60],
                task=sub_question.question,
            )

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
            self._search_results.append(self._normalize_result(sub_question, result, "DELEGATE"))

    async def _search_autonomous(self, questions: list[Any]) -> None:
        total = len(questions)
        state_id = f"research_{self._deps.run_id}"
        pending = [
            question for question in questions if question.question_id not in self._retry_checkpoint
        ]

        for sub_question in pending:
            await self._deps.emit(
                "research.agent_spawned",
                agent_id=f"{self._deps.run_id}_{sub_question.question_id}",
                agent_name=sub_question.question[:60],
                task=sub_question.question,
            )

        for sub_question in questions:
            cached = self._retry_checkpoint.get(sub_question.question_id)
            if not cached:
                continue
            await self._reuse_checkpoint(sub_question, cached, total)
            for source in cached.sources:
                if source.snippet:
                    await self._deps.shared_state.write(
                        state_id,
                        {
                            "findings": [
                                {"question_id": sub_question.question_id, "snippet": source.snippet}
                            ]
                        },
                    )

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
                state = await self._deps.shared_state.read(state_id)
                peer_findings = [
                    item.get("snippet", "")
                    for item in state.findings
                    if item.get("question_id") != sub_question.question_id
                ]
                result = await self._search_question_isolated(
                    sub_question,
                    peer_findings=[item for item in peer_findings if item],
                )
                await self._deps.shared_state.write(
                    state_id,
                    {
                        "findings": [
                            {"question_id": sub_question.question_id, "snippet": source.snippet}
                            for source in result.sources
                            if source.snippet
                        ],
                    },
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
            normalized = self._normalize_result(sub_question, result, "AUTONOMOUS")
            self._search_results.append(normalized)
            await self._deps.emit(
                "research.agent_completed",
                agent_id=f"{self._deps.run_id}_{sub_question.question_id}",
                status="completed" if normalized.sources else "failed",
                summary=normalized.summary[:200],
            )

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
            )
            return await asyncio.wait_for(
                self._deps.search_coordinator.search(sub_question, context),
                timeout=self._deps.agent_timeout_seconds(),
            )
        except TimeoutError:
            logger.warning(
                "SearchCoordinator timed out after %ds for %s",
                self._deps.agent_timeout_seconds(),
                sub_question.question_id,
            )
            return SearchResult(
                question_id=sub_question.question_id,
                rounds=[],
                sources=[],
                summary="Search timed out",
                coverage_score=0.0,
            )
        except Exception:
            logger.warning(
                "SearchCoordinator failed for %s, falling back to agent",
                sub_question.question_id,
                exc_info=True,
            )
            return await self._search_question_fallback(sub_question)

    async def _search_question_fallback(self, sub_question: Any) -> SearchResult:
        agent = Agent(
            role=f"searcher_{sub_question.question_id}",
            llm=self._deps.llm_adapter,
            system_prompt=_SEARCHER_SYSTEM_PROMPT,
            tools=[self._deps.web_search_tool],
            event_emitter=self._deps.event_emitter,
            max_turns=self._deps.settings.max_search_rounds * 3 + 1,
        )
        try:
            output = await asyncio.wait_for(
                agent.arun(self._build_search_task(sub_question)),
                timeout=self._deps.agent_timeout_seconds(),
            )
        except TimeoutError:
            output = '{"sources": [], "summary": "Search timed out", "queries_used": []}'
        return process_agent_search_output(sub_question, output)

    async def _search_question_isolated(
        self,
        sub_question: Any,
        *,
        peer_findings: list[str] | None = None,
    ) -> SearchResult:
        self._deps.check_cancelled()
        assert self._plan is not None
        coordinator = SearchCoordinator(
            self._deps.llm_adapter,
            self._deps.web_search,
            max_search_rounds=self._deps.settings.max_search_rounds,
            on_event=self._deps.search_event_handler,
            **self._deps.llm_kwargs,
        )
        context = SearchContext(
            run_id=self._deps.run_id,
            plan_id=self._plan.plan_id,
            user_query=self._plan.query,
            prior_findings=peer_findings or [],
            excluded_urls=[],
        )
        try:
            return await asyncio.wait_for(
                coordinator.search(sub_question, context),
                timeout=self._deps.agent_timeout_seconds(),
            )
        except TimeoutError:
            logger.warning(
                "Isolated SC timed out after %ds for %s",
                self._deps.agent_timeout_seconds(),
                sub_question.question_id,
            )
            return SearchResult(
                question_id=sub_question.question_id,
                rounds=[],
                sources=[],
                summary="Search timed out",
                coverage_score=0.0,
            )
        except Exception:
            logger.warning(
                "Isolated SC failed for %s, falling back to agent",
                sub_question.question_id,
                exc_info=True,
            )
            return await self._search_question_fallback(sub_question)

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
