"""ResearchSession — complete lifecycle for one deep research run.

Three orchestration modes control the search phase:

- **DIRECT** — single ``SearchCoordinator``, serial execution, shared context.
- **DELEGATE** — Supervisor fans out N isolated ``SearchCoordinator`` instances
  in parallel; each has a clean context with no prior_findings pollution.
- **AUTONOMOUS** — parallel isolated coordinators with ``SharedState``
  collaboration; agents share discoveries and adjust search strategies.

Pipeline stages (serial): ClarificationAgent → PlannerAgent → *search* →
SourceAggregator → ConflictResolver → ValidationAgent → ReportGenerator →
QualityEvaluator.
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
from houyi.application.research.clarification import ClarificationAgent, ClarificationResult
from houyi.application.research.intermediate import IntermediateReport, IntermediateReportGenerator
from houyi.application.research.planner import ResearchPlanner
from houyi.application.research.quality import QualityEvaluator
from houyi.application.research.report import ReportGenerator
from houyi.application.research.report_pipeline import ReportPipeline
from houyi.application.research.search import SearchCoordinator
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
    SearchContext,
    SearchResult,
    SearchRound,
    SourceReference,
)
from houyi.application.research.url_validator import URLValidator
from houyi.application.research.validation import ValidationAgent, ValidationReport
from houyi.application.runtime.agent import Agent
from houyi.application.runtime.agent_team import AgentTeamManager
from houyi.application.runtime.conflict import ConflictRecord, ConflictResolver
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.application.runtime.message_bus import AgentMessage, AgentMessageBus, AgentMessageType
from houyi.application.runtime.shared_state import InMemoryStateBackend, SharedStateBackend
from houyi.skills.web_search.service import WebSearchService

_EVENT_TO_A2A: dict[str, AgentMessageType] = {
    "research.source_found": AgentMessageType.SOURCE_DISCOVERED,
    "research.step_completed": AgentMessageType.QUESTION_COVERED,
    "research.agent_spawned": AgentMessageType.TASK_DELEGATE,
    "research.agent_completed": AgentMessageType.TASK_RESULT,
    "research.step_started": AgentMessageType.TASK_PROGRESS,
    "research.conflict_detected": AgentMessageType.CONFLICT_DETECTED,
    "research.plan_generated": AgentMessageType.FINDING_PUBLISHED,
}

_AGENT_EVENT_TO_A2A: dict[AgentEventType, AgentMessageType] = {
    AgentEventType.TEAM_AGENT_SPAWNED: AgentMessageType.TASK_DELEGATE,
    AgentEventType.TEAM_AGENT_COMPLETED: AgentMessageType.TASK_RESULT,
    AgentEventType.TOOL_COMPLETED: AgentMessageType.SOURCE_DISCOVERED,
    AgentEventType.PROGRESS: AgentMessageType.TASK_PROGRESS,
}

logger = logging.getLogger(__name__)


def _topo_sort_questions(questions: list[Any]) -> list[Any]:
    """Topologically sort sub-questions respecting ``depends_on``, then priority."""
    by_id = {q.question_id: q for q in questions}
    visited: set[str] = set()
    result: list[Any] = []

    def _visit(qid: str) -> None:
        if qid in visited:
            return
        visited.add(qid)
        q = by_id.get(qid)
        if q is None:
            return
        for dep in q.depends_on:
            _visit(dep)
        result.append(q)

    for q in sorted(questions, key=lambda q: q.priority, reverse=True):
        _visit(q.question_id)
    return result


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
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
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
        message_bus: AgentMessageBus | None = None,
        shared_state: SharedStateBackend | None = None,
        memory_context: str | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self.session_id = session_id or f"rs_{uuid.uuid4().hex[:12]}"
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._settings = settings or ResearchSettings()
        self._emitter = event_emitter or EventEmitter()
        self._bus = message_bus
        self._shared_state: SharedStateBackend = shared_state or InMemoryStateBackend()
        self._memory_context = memory_context
        self._llm_kwargs = llm_kwargs

        self._llm_adapter = llm_adapter
        self._web_search = web_search
        self._planner = ResearchPlanner(llm_adapter, **llm_kwargs)
        self._aggregator = SourceAggregator()
        self._clarifier = ClarificationAgent(llm_adapter, **llm_kwargs)
        self._search_coordinator = SearchCoordinator(
            llm_adapter,
            web_search,
            max_search_rounds=self._settings.max_search_rounds,
            on_event=self._on_search_event,
            **llm_kwargs,
        )
        self._web_search_tool = WebSearchTool(web_search)
        self._team_mgr = AgentTeamManager(
            llm_adapter=llm_adapter,
            event_emitter=self._emitter,
        )
        self._report_pipeline = ReportPipeline(
            reporter=ReportGenerator(llm_adapter, **llm_kwargs),
            validator=ValidationAgent(llm_adapter, **llm_kwargs),
            evaluator=QualityEvaluator(llm_adapter, **llm_kwargs),
            url_validator=URLValidator(max_concurrent=5, timeout=10),
            conflict_resolver=ConflictResolver(llm=llm_adapter),
            intermediate_gen=IntermediateReportGenerator(llm_adapter, **llm_kwargs),
            web_search=web_search,
            emit=self._emit,
        )

        if self._bus is not None:
            self._bus.register_agent(self.session_id)
            self._emitter.on_any(self._bridge_agent_events_to_bus)

        self._plan: ResearchPlan | None = None
        self._search_results: list[SearchResult] = []
        self._intermediate_reports: list[IntermediateReport] = []
        self._conflicts: list[ConflictRecord] = []
        self._aggregated: AggregatedSources | None = None
        self._report: ResearchReport | None = None
        self._quality: QualityScore | None = None

        self._clarification: ClarificationResult | None = None
        self._validation: ValidationReport | None = None
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
        """Generate the initial research plan.

        For standard/deep depth, runs ClarificationAgent first to detect
        ambiguity. If the query can be refined without user input, the
        improved version is used for planning.
        """
        self._status = ResearchStatus.PLANNING

        effective_query = query
        if self._settings.depth in ("standard", "deep"):
            self._clarification = await self._clarifier.analyze(query)
            if self._clarification.refined_query and self._clarification.confidence < 0.7:
                effective_query = self._clarification.refined_query
                await self._emit(
                    "research.query_refined",
                    original=query,
                    refined=effective_query,
                    issues=self._clarification.issues,
                )

        self._plan = await self._planner.generate_plan(
            effective_query,
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

        Budget varies by orchestration mode:
        - DIRECT: n × 180s (serial SearchCoordinator).
        - DELEGATE: ceil(n / max_agents) × 300s (parallel isolated SC).
        - AUTONOMOUS: ceil(n / max_agents) × 300s + 60s (parallel + SharedState).
        All modes add 180s for report generation + quality evaluation.
        """
        n = len(self._plan.sub_questions) if self._plan else 3
        mode = self._settings.orchestration_mode
        if mode in (OrchestrationMode.DELEGATE, OrchestrationMode.AUTONOMOUS):
            batches = max(1, -(-n // self._settings.max_agents))
            extra = 60 if mode == OrchestrationMode.AUTONOMOUS else 0
            return batches * self._AGENT_TIMEOUT_SECONDS + self._REPORT_BUDGET_SECONDS + extra
        return n * self._PER_QUESTION_BUDGET_SECONDS + self._REPORT_BUDGET_SECONDS

    async def execute(self) -> None:
        """Execute the research plan: search all sub-questions and generate report.

        Timeout varies by mode (5 sub-questions, max_agents=5):
        - DIRECT:     5×180+180 = 1080s  (serial SearchCoordinator)
        - DELEGATE:   ceil(5/5)×300+180 = 480s  (parallel isolated SC)
        - AUTONOMOUS: ceil(5/5)×300+240 = 540s  (parallel + SharedState)
        Each individual SearchCoordinator also has a 300s hard cap.
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
        """Search all sub-questions using the configured orchestration mode.

        Modes:
          DIRECT     — Single SearchCoordinator, serial, shared context
                       (prior_findings accumulate across questions).
          DELEGATE   — Supervisor fans out N isolated SearchCoordinators
                       in parallel; no prior_findings pollution.
          AUTONOMOUS — Parallel isolated SearchCoordinators with SharedState
                       collaboration; agents see each other's discoveries.

        Respects ``depends_on``: questions are topologically sorted so
        dependencies are searched first, with priority as tiebreaker.
        """
        assert self._plan is not None
        questions = _topo_sort_questions(self._plan.sub_questions)
        mode = self._settings.orchestration_mode

        if mode == OrchestrationMode.DELEGATE:
            await self._search_delegate(questions)
        elif mode == OrchestrationMode.AUTONOMOUS:
            await self._search_autonomous(questions)
        else:
            await self._search_direct(questions)

        self._aggregated = await self._aggregator.aggregate(self._search_results)

        if self._settings.depth in ("standard", "deep"):
            self._intermediate_reports = await self._report_pipeline.generate_intermediates(
                self._search_results,
                questions,
                self._plan.query,
            )

    # ── DIRECT mode ────────────────────────────────────────────

    async def _search_direct(self, questions: list[Any]) -> None:
        """DIRECT: serial SearchCoordinator with shared context."""
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
            result = await self._search_question(sq)
            self._search_results.append(result)
            await self._emit(
                "research.step_completed",
                step_id=sq.question_id,
                step=sq.question[:60],
                total_steps=total,
                completed_steps=len(self._search_results),
                sources=len(result.sources),
            )

    # ── DELEGATE mode ──────────────────────────────────────────

    async def _search_delegate(self, questions: list[Any]) -> None:
        """DELEGATE mode: Supervisor fans out N SearchAgents in parallel.

        Each SearchAgent holds an **independent** SearchCoordinator instance
        with a clean SearchContext (no prior_findings pollution).
        Results are collected via fan-in after all agents complete.
        """
        assert self._plan is not None
        total = len(questions)

        for sq in questions:
            await self._emit(
                "research.agent_spawned",
                agent_id=f"{self.session_id}_{sq.question_id}",
                agent_name=sq.question[:60],
                task=sq.question,
            )

        sem = asyncio.Semaphore(self._settings.max_agents)

        async def _run_one(sq: Any) -> SearchResult:
            async with sem:
                self._check_cancelled()
                await self._emit(
                    "research.step_started",
                    step_id=sq.question_id,
                    step=sq.question[:60],
                    total_steps=total,
                    completed_steps=len(self._search_results),
                )
                result = await self._search_question_isolated(sq)
                await self._emit(
                    "research.step_completed",
                    step_id=sq.question_id,
                    step=sq.question[:60],
                    total_steps=total,
                    sources=len(result.sources),
                )
                return result

        results = await asyncio.gather(
            *[_run_one(sq) for sq in questions],
            return_exceptions=True,
        )

        for sq, r in zip(questions, results, strict=True):
            if isinstance(r, BaseException):
                logger.warning("DELEGATE search failed for %s: %s", sq.question_id, r)
                r = SearchResult(
                    question_id=sq.question_id,
                    rounds=[],
                    sources=[],
                    summary=f"Search failed: {r}",
                    coverage_score=0.0,
                )
            self._search_results.append(r)

    # ── AUTONOMOUS mode ────────────────────────────────────────

    async def _search_autonomous(self, questions: list[Any]) -> None:
        """AUTONOMOUS: parallel isolated SC + SharedState collaboration.

        Each agent publishes its discovered sources to ``SharedStateBackend``
        via the ``findings`` field.  Peer agents read the shared state before
        searching, enabling de-duplication and multi-perspective adjustment.
        """
        assert self._plan is not None
        total = len(questions)
        state_id = f"research_{self.session_id}"
        shared = self._shared_state

        for sq in questions:
            await self._emit(
                "research.agent_spawned",
                agent_id=f"{self.session_id}_{sq.question_id}",
                agent_name=sq.question[:60],
                task=sq.question,
            )

        sem = asyncio.Semaphore(self._settings.max_agents)

        async def _run_one(sq: Any) -> SearchResult:
            async with sem:
                self._check_cancelled()
                await self._emit(
                    "research.step_started",
                    step_id=sq.question_id,
                    step=sq.question[:60],
                    total_steps=total,
                    completed_steps=len(self._search_results),
                )
                state = await shared.read(state_id)
                peer_findings = [
                    f.get("snippet", "")
                    for f in state.findings
                    if f.get("question_id") != sq.question_id
                ]
                result = await self._search_question_isolated(
                    sq,
                    peer_findings=[p for p in peer_findings if p],
                )
                await shared.write(
                    state_id,
                    {
                        "findings": [
                            {"question_id": sq.question_id, "snippet": src.snippet}
                            for src in result.sources
                            if src.snippet
                        ],
                    },
                )
                await self._emit(
                    "research.step_completed",
                    step_id=sq.question_id,
                    step=sq.question[:60],
                    total_steps=total,
                    sources=len(result.sources),
                )
                return result

        results = await asyncio.gather(
            *[_run_one(sq) for sq in questions],
            return_exceptions=True,
        )

        for sq, r in zip(questions, results, strict=True):
            if isinstance(r, BaseException):
                logger.warning("AUTONOMOUS search failed for %s: %s", sq.question_id, r)
                r = SearchResult(
                    question_id=sq.question_id,
                    rounds=[],
                    sources=[],
                    summary=f"Search failed: {r}",
                    coverage_score=0.0,
                )
            self._search_results.append(r)
            await self._emit(
                "research.agent_completed",
                agent_id=f"{self.session_id}_{sq.question_id}",
                status="completed" if r.sources else "failed",
                summary=r.summary[:200],
            )

    _AGENT_TIMEOUT_SECONDS = 300

    async def _search_question(self, sq: Any) -> SearchResult:
        """DIRECT: search one sub-question with shared SearchCoordinator.

        Uses accumulated ``prior_findings`` from earlier questions.
        Falls back to Agent tool-loop if the coordinator errors out.
        """
        self._check_cancelled()
        assert self._plan is not None
        try:
            context = SearchContext(
                session_id=self.session_id,
                plan_id=self._plan.plan_id,
                user_query=self._plan.query,
                prior_findings=[sr.summary for sr in self._search_results if sr.summary],
                excluded_urls=[s.url for sr in self._search_results for s in sr.sources if s.url],
            )
            result = await asyncio.wait_for(
                self._search_coordinator.search(sq, context),
                timeout=self._AGENT_TIMEOUT_SECONDS,
            )
            return result
        except TimeoutError:
            logger.warning(
                "SearchCoordinator timed out after %ds for %s",
                self._AGENT_TIMEOUT_SECONDS,
                sq.question_id,
            )
            return SearchResult(
                question_id=sq.question_id,
                rounds=[],
                sources=[],
                summary="Search timed out",
                coverage_score=0.0,
            )
        except Exception:
            logger.warning(
                "SearchCoordinator failed for %s, falling back to agent",
                sq.question_id,
                exc_info=True,
            )
            return await self._search_question_fallback(sq)

    async def _search_question_fallback(self, sq: Any) -> SearchResult:
        """Fallback: single-agent tool-loop when SearchCoordinator fails."""
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
            output = '{"sources": [], "summary": "Search timed out", "queries_used": []}'
        return _parse_search_output(sq, output)

    async def _search_question_isolated(
        self,
        sq: Any,
        *,
        peer_findings: list[str] | None = None,
    ) -> SearchResult:
        """DELEGATE/AUTONOMOUS: search with a fresh isolated SearchCoordinator.

        Creates a new ``SearchCoordinator`` per call — no prior_findings
        pollution.  *peer_findings* (AUTONOMOUS only) carries discoveries
        published by other agents via SharedState.
        """
        self._check_cancelled()
        assert self._plan is not None
        sc = SearchCoordinator(
            self._llm_adapter,
            self._web_search,
            max_search_rounds=self._settings.max_search_rounds,
            on_event=self._on_search_event,
            **self._llm_kwargs,
        )
        context = SearchContext(
            session_id=self.session_id,
            plan_id=self._plan.plan_id,
            user_query=self._plan.query,
            prior_findings=peer_findings or [],
            excluded_urls=[],
        )
        try:
            return await asyncio.wait_for(
                sc.search(sq, context),
                timeout=self._AGENT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Isolated SC timed out after %ds for %s",
                self._AGENT_TIMEOUT_SECONDS,
                sq.question_id,
            )
            return SearchResult(
                question_id=sq.question_id,
                rounds=[],
                sources=[],
                summary="Search timed out",
                coverage_score=0.0,
            )
        except Exception:
            logger.warning(
                "Isolated SC failed for %s, falling back to agent",
                sq.question_id,
                exc_info=True,
            )
            return await self._search_question_fallback(sq)

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

    async def _run_report(self) -> None:
        """Delegate to ReportPipeline: conflicts → report → validation → repair → quality."""
        assert self._plan is not None
        assert self._aggregated is not None

        result = await self._report_pipeline.run(
            self._plan,
            self._aggregated,
            self._search_results,
            self._intermediate_reports or None,
            self._settings,
        )
        self._report = result.report
        self._quality = result.quality
        self._validation = result.validation
        self._conflicts = result.conflicts

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def _on_search_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Callback from SearchCoordinator — forward to SSE stream."""
        if event_type == "search.queries_generated":
            await self._emit(
                "research.search_queries",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                queries=data.get("queries", []),
            )
        elif event_type == "search.source_discovered":
            await self._emit(
                "research.source_found",
                title=data.get("title", ""),
                url=data.get("url", ""),
                snippet=data.get("snippet", ""),
                question_id=data.get("question_id", ""),
                query=data.get("query", ""),
            )

    async def _emit(self, event_type: str, **data: Any) -> None:
        self._event_seq += 1
        merged = {"research_event": event_type, "sequence": self._event_seq, **data}
        await self._emitter.emit(
            AgentEvent(
                event_type=AgentEventType.PROGRESS,
                agent_id=self.session_id,
                data=merged,
            ),
        )
        if self._bus is not None:
            a2a_type = _EVENT_TO_A2A.get(event_type)
            if a2a_type is not None:
                await self._bus.publish(
                    f"research.{self.session_id}",
                    AgentMessage(
                        sender_id=self.session_id,
                        message_type=a2a_type,
                        topic=f"research.{self.session_id}",
                        payload=merged,
                    ),
                )

    async def _bridge_agent_events_to_bus(self, event: AgentEvent) -> None:
        """Bridge internal agent events to MessageBus for A2A observability.

        Registered as a wildcard listener on the shared EventEmitter so that
        events from all modes (DIRECT/DELEGATE/AUTONOMOUS) flow through the
        unified Pub/Sub channel.
        """
        if self._bus is None:
            return
        if event.event_type == AgentEventType.PROGRESS:
            return
        a2a_type = _AGENT_EVENT_TO_A2A.get(event.event_type)
        if a2a_type is None:
            return
        await self._bus.publish(
            f"research.{self.session_id}",
            AgentMessage(
                sender_id=event.agent_id or self.session_id,
                message_type=a2a_type,
                topic=f"research.{self.session_id}",
                payload={"agent_event": event.event_type.value, **event.data},
            ),
        )
