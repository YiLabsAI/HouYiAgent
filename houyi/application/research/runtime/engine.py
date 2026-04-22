"""ResearchRuntime — complete lifecycle for one deep research run.

Three orchestration modes control the search phase:

- **DIRECT** — single ``SearchExecutor``, serial execution, shared context.
- **DELEGATE** — Supervisor fans out N isolated ``SearchExecutor`` instances
  in parallel; each has a clean context with no prior_findings pollution.
- **AUTONOMOUS** — parallel isolated coordinators with ``SharedState``
  collaboration; agents share discoveries and adjust search strategies.

Pipeline stages (serial): PlannerAgent → *search* →
SourceAggregator → ConflictResolver → ReportGenerator → URLValidator →
QualityEvaluator.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.adapters.memory.builder import MemoryCandidateBuilder
from houyi.adapters.memory.types import (
    MemoryBuildInput,
    MemoryBuildItem,
    MemoryCandidate,
    MemoryScope,
    MemorySourceKind,
)
from houyi.application.research.aggregator import SourceAggregator
from houyi.application.research.planner import ResearchPlanner, validate_research_plan
from houyi.application.research.quality import QualityEvaluator
from houyi.application.research.report import ReportGenerator
from houyi.application.research.runtime.coordinator import (
    CoordinatorServices,
    ResearchCoordinator,
)
from houyi.application.research.runtime.errors import (
    ResearchCancelledError,
    ResearchPlanMissingError,
    ResearchReportNotReadyError,
    ResearchStateError,
    ResearchTimeoutError,
)
from houyi.application.research.runtime.event_bridge import ResearchEventBridge
from houyi.application.research.runtime.intermediate import (
    IntermediateReport,
    IntermediateReportGenerator,
)
from houyi.application.research.runtime.planning import (
    PlanningCoordinator,
    PlanningRequest,
)
from houyi.application.research.runtime.processing import process_agent_search_output
from houyi.application.research.runtime.report_pipeline import ReportPipeline
from houyi.application.research.runtime.search_executor import SearchExecutor
from houyi.application.research.runtime.synthesis import SynthesisCoordinator
from houyi.application.research.runtime.time_policy import TimeBudgetPolicy
from houyi.application.research.runtime.tools import WebSearchTool
from houyi.application.research.types import (
    AggregatedSources,
    ClarificationResult,
    PlanEdit,
    PlanStatus,
    QualityScore,
    ResearchPlan,
    ResearchProgress,
    ResearchReport,
    ResearchSettings,
    ResearchStatus,
    SearchResult,
)
from houyi.application.research.url_validator import URLValidator
from houyi.application.runtime.agent_team import AgentTeamManager
from houyi.application.runtime.conflict import ConflictRecord, ConflictResolver
from houyi.application.runtime.events import EventEmitter
from houyi.application.runtime.message_bus import AgentMessageBus
from houyi.application.runtime.shared_state import InMemoryStateBackend, SharedStateBackend
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)


_parse_search_output = process_agent_search_output


class ResearchRuntime:
    """One deep research run, from plan to report.

    Typical flow::

        runtime = ResearchRuntime(...)
        plan = await runtime.start("Compare AI agent frameworks")
        plan = await runtime.edit_plan([PlanEdit(...)])
        await runtime.confirm_plan()
        await runtime.execute()
        report = await runtime.get_report()
        score = runtime.quality_score
    """

    _AGENT_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        run_id: str | None = None,
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
        self.run_id = run_id or f"rr_{uuid.uuid4().hex[:12]}"
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.started_at = 0.0
        self._settings = settings or ResearchSettings()
        self._emitter = event_emitter or EventEmitter()
        self._bus = message_bus
        self._shared_state: SharedStateBackend = shared_state or InMemoryStateBackend()
        self._memory_context = memory_context
        self._llm_kwargs = llm_kwargs
        self._time_budget_policy = TimeBudgetPolicy()
        self._event_bridge = ResearchEventBridge(
            run_id=self.run_id,
            emitter=self._emitter,
            message_bus=self._bus,
        )

        self._llm_adapter = llm_adapter
        self._web_search = web_search
        self._planner = ResearchPlanner(llm_adapter, **llm_kwargs)
        self._aggregator = SourceAggregator()
        self._search_executor = SearchExecutor(
            llm_adapter,
            web_search,
            max_search_rounds=self._settings.max_search_rounds,
            on_event=self._event_bridge.on_search_event,
            check_cancelled=self._check_cancelled,
            **llm_kwargs,
        )
        self._web_search_tool = WebSearchTool(web_search)
        self._team_mgr = AgentTeamManager(
            llm_adapter=llm_adapter,
            event_emitter=self._emitter,
        )
        self._report_pipeline = ReportPipeline(
            reporter=ReportGenerator(llm_adapter, **llm_kwargs),
            evaluator=QualityEvaluator(llm_adapter, **llm_kwargs),
            url_validator=URLValidator(max_concurrent=5, timeout=10),
            conflict_resolver=ConflictResolver(llm=llm_adapter),
            intermediate_gen=IntermediateReportGenerator(llm_adapter, **llm_kwargs),
            emit=self._emit,
        )
        self._research = ResearchCoordinator(
            CoordinatorServices(
                run_id=self.run_id,
                llm_adapter=self._llm_adapter,
                web_search=self._web_search,
                web_search_tool=self._web_search_tool,
                settings=self._settings,
                shared_state=self._shared_state,
                aggregator=self._aggregator,
                report_pipeline=self._report_pipeline,
                search_executor=self._search_executor,
                search_event_handler=self._event_bridge.on_search_event,
                event_emitter=self._emitter,
                emit=self._emit,
                emit_restored_search_events=self._event_bridge.emit_restored_search_events,
                check_cancelled=self._check_cancelled,
                elapsed_seconds=lambda: round(time.time() - self._started_at, 2),
                agent_timeout_seconds=lambda: self._AGENT_TIMEOUT_SECONDS,
                llm_kwargs=self._llm_kwargs,
            )
        )
        self._synthesis = SynthesisCoordinator(self._report_pipeline)
        self._planning = PlanningCoordinator(self._planner, self._emit)
        self._memory_builder = MemoryCandidateBuilder()

        if self._bus is not None:
            self._bus.register_agent(self.run_id)
            self._emitter.on_any(self._event_bridge.bridge_agent_event)

        self._plan: ResearchPlan | None = None
        self._search_results: list[SearchResult] = []
        self._intermediate_reports: list[IntermediateReport] = []
        self._conflicts: list[ConflictRecord] = []
        self._aggregated: AggregatedSources | None = None
        self._report: ResearchReport | None = None
        self._quality: QualityScore | None = None
        self._phase_timings_ms: dict[str, float] = {}
        self._search_elapsed_ms: float = 0.0
        self._aggregate_ms: float = 0.0
        self._intermediate_ms: float = 0.0

        self._clarification: ClarificationResult | None = None
        self._memory_candidates: list[MemoryCandidate] = []

        self._status = ResearchStatus.PLANNING
        self._started_at = time.time()
        self._error: str | None = None
        self._cancelled = False
        self._execution_phase: str = "init"

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------

    async def start(self, query: str) -> ResearchPlan:
        """Generate the initial research plan.

        The planner always generates the first draft. For standard/deep
        depth, it may return an internal clarification signal and a refined
        query when missing constraints would materially change the plan.
        """
        self._status = ResearchStatus.PLANNING

        result = await self._planning.start(
            PlanningRequest(
                query=query,
                settings=self._settings,
                memory_context=self._memory_context,
            )
        )
        self._clarification = result.clarification
        self._plan = result.plan
        self._status = ResearchStatus.PLAN_READY
        return self._plan

    async def edit_plan(self, edits: list[PlanEdit]) -> ResearchPlan:
        """Apply user edits to the current plan."""
        if not self._plan:
            raise ResearchPlanMissingError("No plan to edit — call start() first")
        self._plan = self._planning.edit(self._plan, edits)
        self._status = ResearchStatus.PLAN_READY
        return self._plan

    async def confirm_plan(self) -> ResearchPlan:
        """Mark the plan as confirmed and ready for execution."""
        if not self._plan:
            raise ResearchPlanMissingError("No plan to confirm")
        try:
            self._plan = await self._planning.confirm(self._plan)
        except ValueError as exc:
            raise ResearchStateError(str(exc)) from exc
        self._status = ResearchStatus.PLAN_READY
        return self._plan

    # ------------------------------------------------------------------
    # Phase 2: Execution
    # ------------------------------------------------------------------

    _PER_QUESTION_BUDGET_SECONDS = 120
    # Report pipeline (conflicts + sections + URL validation + quality) can exceed 10+ LLM
    # calls × 30–120s each on standard/deep; budget must not assume "search is the long pole".
    _REPORT_BUDGET_BY_DEPTH: dict[str, int] = {"quick": 600, "standard": 1200, "deep": 1500}

    def _report_budget_seconds(self) -> int:
        """Wall-clock budget for search → intermediates → report pipeline (depth-aware)."""
        return self._time_budget_policy.report_budget_seconds(self._settings)

    def _runtime_timeout(self) -> float:
        """Dynamic timeout: budget per question + report generation.

        On retry, checkpointed (already-completed) questions are excluded
        from the budget — only remaining questions count.

        When **all** sub-questions are checkpointed, remaining search work is 0 — do not use
        ``max(1, …)`` to inject a fake per-question slot; that mis-accounts wall time vs the
        report pipeline (the long pole when search is skipped).

        Budget varies by orchestration mode (report budget from ``_report_budget_seconds()``):
        - DIRECT: remaining × 120s + report budget.
        - DELEGATE / AUTONOMOUS: batch×300s + report budget (+ extras); no batches when remaining=0.
        """
        checkpoint = getattr(self, "_retry_checkpoint", {})
        return self._time_budget_policy.runtime_timeout_seconds(
            self._settings,
            self._plan,
            len(checkpoint),
        )

    def _reset_execution_state(self) -> None:
        """Reset mutable execution state before a new run."""
        self._search_results.clear()
        self._intermediate_reports.clear()
        self._conflicts.clear()
        self._aggregated = None
        self._report = None
        self._quality = None
        self._phase_timings_ms = {}
        self._search_elapsed_ms = 0.0
        self._aggregate_ms = 0.0
        self._intermediate_ms = 0.0
        self._error = None

    async def execute(self) -> None:
        """Execute the research plan: search all sub-questions and generate report.

        Timeout varies by mode and depth (report budget: quick 600s / standard 1200s / deep 1500s):
        - DIRECT: remaining×120s + report budget (remaining = sub-questions not in checkpoint).
        - DELEGATE/AUTONOMOUS: batch×300 + report budget + extras; all-checkpoint → report budget only.
        Each individual SearchExecutor also has a 300s hard cap.
        """
        if not self._plan:
            raise ResearchPlanMissingError("No plan — call start() first")
        if self._plan.status not in (PlanStatus.CONFIRMED, PlanStatus.DRAFT):
            raise ResearchStateError(f"Plan status {self._plan.status.value} cannot be executed")
        validation_error = validate_research_plan(self._plan)
        if validation_error is not None:
            raise ResearchStateError(validation_error)

        self._plan.status = PlanStatus.EXECUTING
        self._status = ResearchStatus.EXECUTING
        self._intermediate_reuse_by_qid = {ir.question_id: ir for ir in self._intermediate_reports}
        self._retry_checkpoint: dict[str, SearchResult] = {
            sr.question_id: sr for sr in self._search_results if sr.sources
        }
        total_sq = len(self._plan.sub_questions)
        if (
            self._settings.depth in ("standard", "deep")
            and total_sq
            and len(self._retry_checkpoint) == total_sq
            and not self._intermediate_reuse_by_qid
        ):
            logger.info(
                "research.intermediate_reuse: full search checkpoint but no prior intermediates "
                "— will generate %d intermediate reports (LLM); persist after last run enables reuse",
                total_sq,
            )
        self._reset_execution_state()
        self._started_at = time.time()
        self.started_at = self._started_at
        self._execution_phase = "search"
        runtime_timeout = self._runtime_timeout()

        try:
            await asyncio.wait_for(
                self._execute_inner(),
                timeout=runtime_timeout,
            )
        except TimeoutError:
            d = self._settings.depth
            depth_s = d.value if hasattr(d, "value") else str(d)
            logger.warning(
                "research.runtime_timeout run_id=%s depth=%s phase=%s budget_s=%.0f",
                self.run_id,
                depth_s,
                self._execution_phase,
                runtime_timeout,
            )
            self._error = f"Research timed out after {runtime_timeout:.0f}s"
            self._plan.status = PlanStatus.FAILED
            self._status = ResearchStatus.FAILED
            await self._emit("research.failed", error=self._error)
            raise ResearchTimeoutError(self._error) from None
        except asyncio.CancelledError:
            self._plan.status = PlanStatus.FAILED
            self._error = self._error or "Cancelled"
            self._status = ResearchStatus.CANCELLED
            await self._emit("research.cancelled", reason=self._error)
            raise ResearchCancelledError(self._error) from None
        except Exception as exc:
            if not self._phase_timings_ms and self._execution_phase == "report":
                self._phase_timings_ms = {
                    "partial_total_ms": round((time.time() - self._started_at) * 1000.0, 1)
                }
            self._error = str(exc)
            self._plan.status = PlanStatus.FAILED
            self._status = ResearchStatus.FAILED
            await self._emit("research.failed", error=self._error)
            raise

    async def _execute_inner(self) -> None:
        """Core execution: search → report → quality evaluation."""
        self._execution_phase = "search"
        await self._run_search()
        self._check_cancelled()
        self._status = ResearchStatus.GENERATING_REPORT
        total = len(self._plan.sub_questions) if self._plan else 0
        await self._emit(
            "research.step_started",
            step_id="report_generation",
            step="Generating report...",
            total_steps=total,
            completed_steps=total,
            elapsed_seconds=round(time.time() - self._started_at, 2),
        )
        self._execution_phase = "report"
        try:
            await self._run_report()
        except asyncio.CancelledError:
            await self._event_bridge.emit_report_generation_end(
                total_steps=total,
                elapsed_seconds=round(time.time() - self._started_at, 2),
                error="cancelled",
            )
            raise
        except Exception as exc:
            await self._event_bridge.emit_report_generation_end(
                total_steps=total,
                elapsed_seconds=round(time.time() - self._started_at, 2),
                error=str(exc),
            )
            raise
        await self._event_bridge.emit_report_generation_end(
            total_steps=total,
            elapsed_seconds=round(time.time() - self._started_at, 2),
        )
        self._execution_phase = "done"
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
        """Cancel the research runtime.

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
            raise ResearchReportNotReadyError("Report not ready — execute() first")
        return self._report

    async def extract_memories(self) -> list[MemoryCandidate]:
        """Extract memory candidates from research results.

        Builds synthetic messages from report sections and source summaries,
        then delegates to ``MemoryCandidateExtractor`` if available. Falls
        back to a lightweight heuristic when no extractor is configured.
        """
        if not self._report:
            return []
        query_tag = self._plan.query[:50] if self._plan else ""
        memory_input = MemoryBuildInput(
            source_type=MemorySourceKind.SEARCH,
            scope=MemoryScope.USER,
            source_context="deep_research",
            items=[
                *[
                    MemoryBuildItem(
                        content=f"{section.title}: {section.content[:500]}",
                        role="report_section",
                        source_ids=[section.section_id],
                        source_context="deep_research",
                        suggested_tags=[tag for tag in ["research", query_tag] if tag],
                        confidence=0.7,
                        metadata={"kind": "report_section", "title": section.title},
                    )
                    for section in self._report.sections
                    if section.content.strip()
                ],
                *[
                    MemoryBuildItem(
                        content=f"Source: {src.title} — {src.snippet[:200]}",
                        role="search_source",
                        source_ids=[src.reference_id],
                        source_context="deep_research",
                        suggested_tags=["research_source"],
                        confidence=0.5,
                        metadata={"kind": "source_reference", "title": src.title, "url": src.url},
                    )
                    for src in (self._report.references or [])[:10]
                    if src.title and src.snippet
                ],
            ],
            metadata={"query": self._plan.query if self._plan else "", "run_id": self.run_id},
        )
        candidates = await self._memory_builder.build(memory_input)
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
    def phase_timings_ms(self) -> dict[str, float]:
        return dict(self._phase_timings_ms)

    @property
    def search_elapsed_ms(self) -> float:
        return self._search_elapsed_ms

    @property
    def per_question_elapsed_ms(self) -> list[dict[str, Any]]:
        """Per-sub-question wall-clock timing for observability."""
        return [
            {
                "question_id": sr.question_id,
                "elapsed_ms": sr.elapsed_ms,
                "rounds": len(sr.rounds),
                "sources": len(sr.sources),
                "exhausted": sr.exhausted,
            }
            for sr in self._search_results
        ]

    @property
    def aggregate_ms(self) -> float:
        return self._aggregate_ms

    @property
    def intermediate_ms(self) -> float:
        return self._intermediate_ms

    @property
    def started_at_timestamp(self) -> float:
        return self.started_at

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
            last_event_sequence=self._event_bridge.event_sequence,
            error=self._error,
        )

    # ------------------------------------------------------------------
    # Internal execution — Agent SDK orchestration
    # ------------------------------------------------------------------

    async def _run_search(self) -> None:
        """Search all sub-questions using the configured orchestration mode.

        Modes:
          DIRECT     — Single SearchExecutor, serial, shared context
                       (prior_findings accumulate across questions).
          DELEGATE   — Supervisor fans out N isolated SearchExecutors
                       in parallel; no prior_findings pollution.
          AUTONOMOUS — Parallel isolated SearchExecutors with SharedState
                       collaboration; agents see each other's discoveries.

        Respects ``depends_on``: questions are topologically sorted so
        dependencies are searched first, with priority as tiebreaker.
        """
        assert self._plan is not None
        result = await self._research.run(
            plan=self._plan,
            retry_checkpoint=getattr(self, "_retry_checkpoint", {}),
            intermediate_reuse_by_qid=getattr(self, "_intermediate_reuse_by_qid", None) or {},
        )
        self._search_results = result.search_results
        self._aggregated = result.aggregated_sources
        self._intermediate_reports = result.intermediate_reports
        self._search_elapsed_ms = result.search_elapsed_ms
        self._aggregate_ms = result.aggregate_ms
        self._intermediate_ms = result.intermediate_ms

    def _check_cancelled(self) -> None:
        """Raise ``asyncio.CancelledError`` if the runtime was cancelled."""
        if self._cancelled:
            raise asyncio.CancelledError("Research runtime cancelled by user")

    async def _run_report(self) -> None:
        """Delegate to ReportPipeline: conflicts → report → URL validation → quality."""
        assert self._plan is not None
        assert self._aggregated is not None

        result = await self._synthesis.run(
            plan=self._plan,
            aggregated_sources=self._aggregated,
            search_results=self._search_results,
            intermediate_reports=self._intermediate_reports or None,
            settings=self._settings,
        )
        self._report = result.report
        self._quality = result.quality
        self._conflicts = result.conflicts
        self._phase_timings_ms = result.phase_timings_ms

    async def _emit(self, event_type: str, **data: Any) -> None:
        await self._event_bridge.emit(event_type, **data)
