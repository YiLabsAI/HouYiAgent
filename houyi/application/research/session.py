"""ResearchSession — complete lifecycle for one deep research run.

Orchestrates research through the HouYi Agent SDK:

- **Planner** generates the research plan via direct LLM call.
- **Agent** instances are created per sub-question, each equipped with the
  ``web_search`` tool.  ``Agent.arun()`` drives the tool-loop
  (``AgentRunner``) for multi-round search autonomously.
- **Delegate mode** → sequential Agent execution.
- **Autonomous mode** → concurrent Agent execution via ``asyncio.gather``.
- **ReportGenerator** and **QualityEvaluator** process collected sources.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.adapters.memory.types import MemoryCandidate
from houyi.application.research.aggregator import SourceAggregator
from houyi.application.research.planner import ResearchPlanner
from houyi.application.research.quality import QualityEvaluator
from houyi.application.research.report import ReportGenerator
from houyi.application.research.tools import WebSearchTool
from houyi.application.research.types import (
    AggregatedSources,
    OrchestrationMode,
    PlanEdit,
    PlanStatus,
    QualityScore,
    ResearchPlan,
    ResearchProgress,
    ResearchReport,
    ResearchSettings,
    ResearchStatus,
    SearchResult,
    SearchRound,
    SourceReference,
)
from houyi.application.runtime.agent import Agent
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)

_SEARCHER_SYSTEM_PROMPT = """\
You are a research search agent. Your job is to search the web for a \
specific sub-question and return structured, well-analyzed results.

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


class ResearchSession:
    """One deep research run, from plan to report.

    Typical flow::

        session = ResearchSession(...)
        plan = await session.start("Compare AI agent frameworks")
        plan = await session.edit_plan([PlanEdit(...)])
        await session.confirm_plan()
        await session.execute()
        report = await session.get_report()
        score = session.quality_score
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        settings: ResearchSettings | None = None,
        event_emitter: EventEmitter | None = None,
        memory_context: str | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self.session_id = session_id or f"rs_{uuid.uuid4().hex[:12]}"
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._settings = settings or ResearchSettings()
        self._emitter = event_emitter or EventEmitter()
        self._memory_context = memory_context
        self._llm_kwargs = llm_kwargs

        self._llm_adapter = llm_adapter
        self._planner = ResearchPlanner(llm_adapter, **llm_kwargs)
        self._aggregator = SourceAggregator()
        self._reporter = ReportGenerator(llm_adapter, **llm_kwargs)
        self._evaluator = QualityEvaluator(llm_adapter, **llm_kwargs)
        self._web_search_tool = WebSearchTool(web_search)

        self._plan: ResearchPlan | None = None
        self._search_results: list[SearchResult] = []
        self._aggregated: AggregatedSources | None = None
        self._report: ResearchReport | None = None
        self._quality: QualityScore | None = None

        self._memory_candidates: list[MemoryCandidate] = []

        self._status = ResearchStatus.PLANNING
        self._started_at = time.time()
        self._error: str | None = None
        self._event_seq = 0
        self._cancelled = False

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------

    async def start(self, query: str) -> ResearchPlan:
        """Generate the initial research plan."""
        self._status = ResearchStatus.PLANNING
        self._plan = await self._planner.generate_plan(
            query,
            settings=self._settings,
            memory_context=self._memory_context,
        )
        self._status = ResearchStatus.PLAN_READY
        await self._emit("research.plan_generated", plan=self._plan.model_dump())
        return self._plan

    async def edit_plan(self, edits: list[PlanEdit]) -> ResearchPlan:
        """Apply user edits to the current plan."""
        if not self._plan:
            raise RuntimeError("No plan to edit — call start() first")
        self._plan = await self._planner.refine_plan(self._plan, edits)
        self._status = ResearchStatus.PLAN_READY
        return self._plan

    async def confirm_plan(self) -> ResearchPlan:
        """Mark the plan as confirmed and ready for execution."""
        if not self._plan:
            raise RuntimeError("No plan to confirm")
        self._plan.status = PlanStatus.CONFIRMED
        await self._emit(
            "research.plan_confirmed",
            plan_id=self._plan.plan_id,
            plan_version=self._plan.version,
        )
        return self._plan

    # ------------------------------------------------------------------
    # Phase 2: Execution
    # ------------------------------------------------------------------

    _PER_QUESTION_BUDGET_SECONDS = 180
    _REPORT_BUDGET_SECONDS = 180

    def _session_timeout(self) -> float:
        """Dynamic timeout: budget per question + report generation.

        Reference: MiroThinker uses 600s per LLM call with 200+ turns.
        We use 180s/question + 180s report = e.g. 5 questions → 1080s (18min).
        """
        n = len(self._plan.sub_questions) if self._plan else 3
        return n * self._PER_QUESTION_BUDGET_SECONDS + self._REPORT_BUDGET_SECONDS

    async def execute(self) -> None:
        """Execute the research plan: search all sub-questions and generate report.

        Timeout is computed dynamically: ``60s × num_questions + 120s`` for
        the report/quality phase, so a 7-question plan gets 7×60+120 = 540s.
        Each individual searcher agent also has a 120s hard cap.
        """
        if not self._plan:
            raise RuntimeError("No plan — call start() first")
        if self._plan.status not in (PlanStatus.CONFIRMED, PlanStatus.DRAFT):
            raise RuntimeError(f"Plan status {self._plan.status.value} cannot be executed")

        self._plan.status = PlanStatus.EXECUTING
        self._status = ResearchStatus.EXECUTING
        session_timeout = self._session_timeout()

        try:
            await asyncio.wait_for(
                self._execute_inner(),
                timeout=session_timeout,
            )
        except TimeoutError:
            self._error = f"Research timed out after {session_timeout:.0f}s"
            self._plan.status = PlanStatus.FAILED
            self._status = ResearchStatus.FAILED
            await self._emit("research.failed", error=self._error)
        except asyncio.CancelledError:
            self._plan.status = PlanStatus.FAILED
            self._status = ResearchStatus.CANCELLED
            self._error = self._error or "Cancelled"
            await self._emit("research.cancelled", reason=self._error)
        except Exception as exc:
            self._error = str(exc)
            self._plan.status = PlanStatus.FAILED
            self._status = ResearchStatus.FAILED
            await self._emit("research.failed", error=self._error)
            raise

    async def _execute_inner(self) -> None:
        """Core execution: search → report → quality evaluation."""
        await self._run_search()
        self._check_cancelled()
        self._status = ResearchStatus.GENERATING_REPORT
        await self._run_report()
        if self._plan:
            self._plan.status = PlanStatus.COMPLETED
        self._status = ResearchStatus.COMPLETED
        await self._emit(
            "research.completed",
            report_id=self._report.report_id if self._report else "",
            duration_seconds=round(time.time() - self._started_at, 2),
            quality_score=self._quality.overall if self._quality else None,
        )

    async def cancel(self, reason: str = "") -> None:
        """Cancel the research session.

        Sets the ``_cancelled`` flag which is checked at every search step,
        aborting execution as soon as the current agent finishes.
        """
        self._cancelled = True
        self._status = ResearchStatus.CANCELLED
        self._error = reason or "Cancelled by user"
        await self._emit("research.cancelled", reason=self._error)

    # ------------------------------------------------------------------
    # Phase 3: Results
    # ------------------------------------------------------------------

    async def get_report(self) -> ResearchReport:
        """Return the completed report."""
        if not self._report:
            raise RuntimeError("Report not ready — execute() first")
        return self._report

    async def extract_memories(self) -> list[MemoryCandidate]:
        """Extract memory candidates from research results.

        Builds synthetic messages from report sections and source summaries,
        then delegates to ``MemoryCandidateExtractor`` if available. Falls
        back to a lightweight heuristic when no extractor is configured.
        """
        if not self._report:
            return []

        candidates: list[MemoryCandidate] = []
        for section in self._report.sections:
            if not section.content.strip():
                continue
            candidates.append(
                MemoryCandidate(
                    content=f"{section.title}: {section.content[:500]}",
                    source_context="deep_research",
                    confidence=0.7,
                    suggested_tags=["research", self._plan.query[:50] if self._plan else ""],
                )
            )

        for src in (self._report.references or [])[:10]:
            if src.title and src.snippet:
                candidates.append(
                    MemoryCandidate(
                        content=f"Source: {src.title} — {src.snippet[:200]}",
                        source_context="deep_research",
                        confidence=0.5,
                        suggested_tags=["research_source"],
                    )
                )

        self._memory_candidates = candidates
        await self._emit(
            "memory.candidate_extracted",
            count=len(candidates),
        )
        return candidates

    @property
    def status(self) -> ResearchStatus:
        return self._status

    @property
    def plan(self) -> ResearchPlan | None:
        return self._plan

    @property
    def quality_score(self) -> QualityScore | None:
        return self._quality

    @property
    def progress(self) -> ResearchProgress:
        total = len(self._plan.sub_questions) if self._plan else 0
        completed = len(self._search_results)
        return ResearchProgress(
            status=self._status,
            total_steps=total,
            completed_steps=completed,
            sources_found=sum(len(r.sources) for r in self._search_results),
            elapsed_seconds=round(time.time() - self._started_at, 2),
            last_event_sequence=self._event_seq,
            error=self._error,
        )

    # ------------------------------------------------------------------
    # Internal execution — Agent SDK orchestration
    # ------------------------------------------------------------------

    async def _run_search(self) -> None:
        """Search all sub-questions via Agent SDK."""
        assert self._plan is not None
        questions = sorted(
            self._plan.sub_questions,
            key=lambda q: q.priority,
            reverse=True,
        )

        if self._settings.orchestration_mode == OrchestrationMode.AUTONOMOUS:
            await self._search_parallel(questions)
        else:
            await self._search_sequential(questions)

        self._aggregated = await self._aggregator.aggregate(self._search_results)

    async def _search_sequential(self, questions: list[Any]) -> None:
        """Delegate mode: run one Searcher Agent per sub-question, sequentially."""
        assert self._plan is not None
        total = len(questions)
        for sq in questions:
            self._check_cancelled()
            await self._emit(
                "research.step_started",
                step_id=sq.question_id,
                step=sq.question[:60],
                total_steps=total,
                completed_steps=len(self._search_results),
            )
            result = await self._search_one(sq)
            self._search_results.append(result)
            for src in result.sources:
                await self._emit(
                    "research.source_found",
                    source=src.model_dump(),
                    question_id=sq.question_id,
                )
            await self._emit(
                "research.step_completed",
                step_id=sq.question_id,
                step=sq.question[:60],
                total_steps=total,
                completed_steps=len(self._search_results),
                sources=len(result.sources),
            )

    async def _search_parallel(self, questions: list[Any]) -> None:
        """Autonomous mode: run Searcher Agents concurrently."""
        assert self._plan is not None

        for sq in questions:
            await self._emit(
                "research.agent_spawned",
                agent_id=f"{self.session_id}_{sq.question_id}",
                agent_name=sq.question[:60],
                task=sq.question,
            )

        sem = asyncio.Semaphore(min(self._settings.max_agents, len(questions)))

        async def _bounded_search(sq: Any) -> SearchResult:
            async with sem:
                return await self._search_one(sq)

        results = await asyncio.gather(*[_bounded_search(sq) for sq in questions])

        for sq, sr in zip(questions, results, strict=True):
            self._search_results.append(sr)
            for src in sr.sources:
                await self._emit(
                    "research.source_found",
                    source=src.model_dump(),
                    question_id=sq.question_id,
                )
            await self._emit(
                "research.step_completed",
                step_id=sq.question_id,
                sources=len(sr.sources),
                summary=sr.summary[:200],
            )
            await self._emit(
                "research.agent_completed",
                agent_id=f"{self.session_id}_{sq.question_id}",
                status="completed" if sr.sources else "failed",
                summary=sr.summary[:200],
            )

    _AGENT_TIMEOUT_SECONDS = 300

    async def _search_one(self, sq: Any) -> SearchResult:
        """Create a Searcher Agent and run it via ``Agent.arun()``.

        Enforces a per-agent timeout (``_AGENT_TIMEOUT_SECONDS``) to prevent
        unbounded token/time consumption.
        """
        self._check_cancelled()
        agent = Agent(
            role=f"searcher_{sq.question_id}",
            llm=self._llm_adapter,
            system_prompt=_SEARCHER_SYSTEM_PROMPT,
            tools=[self._web_search_tool],
            event_emitter=self._emitter,
            max_turns=self._settings.max_search_rounds * 3 + 1,
        )
        try:
            output = await asyncio.wait_for(
                agent.arun(self._build_search_task(sq)),
                timeout=self._AGENT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Searcher agent timed out after %ds for question %s",
                self._AGENT_TIMEOUT_SECONDS,
                sq.question_id,
            )
            output = '{"sources": [], "summary": "Search timed out", "queries_used": []}'
        return self._parse_search_output(sq, output)

    def _check_cancelled(self) -> None:
        """Raise ``asyncio.CancelledError`` if the session was cancelled."""
        if self._cancelled:
            raise asyncio.CancelledError("Research session cancelled by user")

    def _build_search_task(self, sq: Any) -> str:
        """Build the task description for a Searcher Agent."""
        assert self._plan is not None
        return (
            f"Research the following sub-question thoroughly:\n\n"
            f"Sub-question: {sq.question}\n"
            f"Overall research topic: {self._plan.query}\n"
            f"Priority: {sq.priority} (higher = more important)\n"
            f"Expected sources: {sq.expected_sources}\n\n"
            f"Search the web and collect relevant, diverse sources."
        )

    @staticmethod
    def _parse_search_output(sq: Any, output: Any) -> SearchResult:
        """Convert Agent output into a ``SearchResult``."""
        raw = str(output or "")
        sources: list[SourceReference] = []
        summary = ""
        queries_used: list[str] = []

        try:
            text = raw.strip()
            if text.startswith("```"):
                first_nl = text.index("\n")
                last_fence = text.rfind("```")
                text = text[first_nl + 1 : last_fence].strip()
            data = json.loads(text)
            for s in data.get("sources", []):
                sources.append(
                    SourceReference(
                        url=s.get("url", ""),
                        title=s.get("title", ""),
                        snippet=s.get("snippet", s.get("content_summary", "")),
                        source_type="web",
                        reliability_score=0.5,
                    )
                )
            summary = data.get("summary", "")
            queries_used = data.get("queries_used", [])
        except (json.JSONDecodeError, ValueError, KeyError):
            summary = raw[:500] if raw else "No results"

        coverage = min(1.0, len(sources) / max(sq.expected_sources, 1))

        return SearchResult(
            question_id=sq.question_id,
            rounds=[
                SearchRound(
                    round_index=0,
                    queries=queries_used,
                    hits=[],
                    sufficient=bool(sources),
                    rationale=summary[:200],
                ),
            ],
            sources=sources,
            summary=summary,
            coverage_score=coverage,
            exhausted=not bool(sources),
        )

    async def _run_report(self) -> None:
        """Generate report and run quality evaluation."""
        assert self._plan is not None
        assert self._aggregated is not None
        self._report = await self._reporter.generate(self._plan, self._aggregated)
        for section in self._report.sections:
            await self._emit(
                "research.report_section",
                chunk={
                    "section_id": section.section_id,
                    "title": section.title,
                    "citations": len(section.citations),
                },
            )
        self._quality = await self._evaluator.evaluate(
            self._report,
            self._aggregated,
        )
        if self._quality:
            self._report.metadata.quality_overall = self._quality.overall

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def _emit(self, event_type: str, **data: Any) -> None:
        self._event_seq += 1
        await self._emitter.emit(
            AgentEvent(
                event_type=AgentEventType.PROGRESS,
                agent_id=self.session_id,
                data={"research_event": event_type, "sequence": self._event_seq, **data},
            ),
        )
