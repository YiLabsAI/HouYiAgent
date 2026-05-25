"""Post-search report pipeline: conflicts → report → URL validation → quality.

Extracted from ResearchRuntime to keep the runtime orchestrator lean.
The pipeline is stateless — all mutable data flows in/out via parameters
and the returned ReportPipelineResult.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from houyi.application.research.quality import QualityEvaluator
from houyi.application.research.report import ReportGenerator
from houyi.application.research.runtime.intermediate import (
    IntermediateReport,
    IntermediateReportGenerator,
)
from houyi.application.research.types import (
    AggregatedSources,
    QualityScore,
    ResearchPlan,
    ResearchReport,
    ResearchSettings,
    SearchResult,
)
from houyi.application.research.url_validator import URLValidator
from houyi.application.runtime.conflict import AgentTaskResult, ConflictRecord, ConflictResolver

logger = logging.getLogger(__name__)
_INTERMEDIATE_TARGET_BY_DEPTH = {"standard": 3, "deep": 5}

EmitFn = Callable[..., Awaitable[None]]


@dataclass
class ReportPipelineResult:
    """Immutable output of a single report pipeline run."""

    report: ResearchReport
    quality: QualityScore | None = None
    conflicts: list[ConflictRecord] = field(default_factory=list)
    phase_timings_ms: dict[str, float] = field(default_factory=dict)


class ReportPipeline:
    """Runs the post-search stages: conflict detection → report generation →
    URL validation → quality evaluation.

    Stateless: each run() call receives all required data and returns a
    ReportPipelineResult.
    """

    def __init__(
        self,
        *,
        reporter: ReportGenerator,
        evaluator: QualityEvaluator,
        url_validator: URLValidator,
        conflict_resolver: ConflictResolver,
        intermediate_gen: IntermediateReportGenerator,
        emit: EmitFn,
    ) -> None:
        self._reporter = reporter
        self._evaluator = evaluator
        self._url_validator = url_validator
        self._conflict_resolver = conflict_resolver
        self._intermediate_gen = intermediate_gen
        self._emit = emit

    async def _generate_pending_intermediates(
        self,
        pending: list[tuple[SearchResult, str]],
        plan_query: str,
        one_fn: Any,
    ) -> dict[str, IntermediateReport]:
        """Run parallel intermediate generation with serial fallback."""
        if not pending:
            return {}
        try:
            results = await asyncio.gather(*[one_fn(sr, qt) for sr, qt in pending])
            return {sr.question_id: ir for (sr, _), ir in zip(pending, results, strict=True)}
        except Exception:
            logger.warning(
                "Parallel intermediate generation failed, falling back to serial",
                exc_info=True,
            )
        gen_by_qid: dict[str, IntermediateReport] = {}
        for sr, qt in pending:
            try:
                gen_by_qid[sr.question_id] = await self._intermediate_gen.generate(
                    sr, qt, plan_query
                )
            except Exception:
                logger.warning("Intermediate report failed for %s", sr.question_id, exc_info=True)
        return gen_by_qid

    async def generate_intermediates(
        self,
        search_results: list[SearchResult],
        questions: list[Any],
        plan_query: str,
        *,
        depth: str = "standard",
        reuse_by_question_id: dict[str, IntermediateReport] | None = None,
        checkpoint_question_ids: frozenset[str] | None = None,
    ) -> list[IntermediateReport]:
        """Generate intermediate reports per sub-question (parallel, with reuse on retry).

        When reuse_by_question_id and checkpoint_question_ids are set,
        sub-questions whose search was **skipped** via checkpoint reuse the
        prior intermediate report — avoids N redundant LLM calls on report-timeout
        retries (root cause of multi-minute report phases).
        """
        qid_to_text = {sq.question_id: sq.question for sq in questions}
        reuse = reuse_by_question_id or {}
        ck = checkpoint_question_ids or frozenset()
        t0 = time.perf_counter()
        reused_count = 0
        selected_ids = _select_intermediate_targets(search_results, questions, depth)

        pending: list[tuple[SearchResult, str]] = []
        for sr in search_results:
            q_text = qid_to_text.get(sr.question_id, "")
            if sr.question_id in ck and sr.question_id in reuse:
                reused_count += 1
                continue
            if sr.question_id not in selected_ids:
                continue
            pending.append((sr, q_text))

        sem = asyncio.Semaphore(4)

        async def _one(sr: SearchResult, q_text: str) -> IntermediateReport:
            async with sem:
                return await self._intermediate_gen.generate(sr, q_text, plan_query)

        gen_by_qid = await self._generate_pending_intermediates(pending, plan_query, _one)

        out: list[IntermediateReport] = []
        for sr in search_results:
            if sr.question_id in ck and sr.question_id in reuse:
                ir = reuse[sr.question_id]
                out.append(ir)
                await self._emit(
                    "research.intermediate_report",
                    question_id=sr.question_id,
                    confidence=ir.confidence,
                    key_findings=ir.key_findings[:3],
                    reused_from_checkpoint=True,
                )
                continue
            generated = gen_by_qid.get(sr.question_id)
            if generated is None:
                continue
            ir = generated
            out.append(ir)
            await self._emit(
                "research.intermediate_report",
                question_id=sr.question_id,
                confidence=ir.confidence,
                key_findings=ir.key_findings[:3],
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "research.intermediate_reports phase=done elapsed_s=%.2f reused=%d generated=%d pending_llm=%d selected=%d total=%d",
            elapsed,
            reused_count,
            len(gen_by_qid),
            len(pending),
            len(selected_ids),
            len(search_results),
        )
        return out

    async def run(
        self,
        plan: ResearchPlan,
        aggregated: AggregatedSources,
        search_results: list[SearchResult],
        intermediate_reports: list[IntermediateReport] | None,
        settings: ResearchSettings,
    ) -> ReportPipelineResult:
        """Execute the full report pipeline and return results."""
        depth_val = (
            settings.depth.value if hasattr(settings.depth, "value") else str(settings.depth)
        )
        quality_enabled = settings.enable_quality_evaluation
        timings: dict[str, float] = {}
        t_run0 = time.perf_counter()

        try:
            conflicts = await self._run_conflict_phase(search_results, settings, timings)
            report, timing_budget = await self._run_generation_phase(
                plan,
                aggregated,
                intermediate_reports,
                settings,
                timings,
            )
            # Summary and URL validation touch disjoint slices of the report
            # (summary reads sections, URL check reads aggregated.sources);
            # run them concurrently to shave the summary latency off the
            # critical path.
            summary_coro = self._reporter.complete_summary(report)
            url_coro = self._execute_url_validation(aggregated, report)
            summary_ms, url_elapsed = await asyncio.gather(summary_coro, url_coro)
            timings["report_summary_ms"] = summary_ms
            timings["url_validate_ms"] = url_elapsed
            quality = None
            if quality_enabled:
                quality = await self._run_final_quality_phase(
                    report,
                    aggregated,
                    timing_budget,
                    timings,
                )
            else:
                timings["quality_disabled"] = 1.0
            return self._build_pipeline_result(
                report,
                quality,
                conflicts,
                depth_val,
                timings,
                t_run0,
            )
        except Exception:
            timings["partial_total_ms"] = (time.perf_counter() - t_run0) * 1000.0
            logger.warning(
                "research.report_pipeline phase=failed depth=%s partial_timings_ms=%s",
                depth_val,
                {k: round(v, 1) for k, v in timings.items()},
                exc_info=True,
            )
            raise

    async def _run_conflict_phase(
        self,
        search_results: list[SearchResult],
        settings: ResearchSettings,
        timings: dict[str, float],
    ) -> list[ConflictRecord]:
        await self._emit("research.pipeline_phase", phase="conflict_detection")
        t = time.perf_counter()
        conflicts = await self._detect_conflicts(search_results, settings)
        timings["conflicts_ms"] = (time.perf_counter() - t) * 1000.0
        return conflicts

    async def _run_generation_phase(
        self,
        plan: ResearchPlan,
        aggregated: AggregatedSources,
        intermediate_reports: list[IntermediateReport] | None,
        settings: ResearchSettings,
        timings: dict[str, float],
    ) -> tuple[ResearchReport, dict[str, Any]]:
        await self._emit("research.pipeline_phase", phase="report_generation")
        t = time.perf_counter()
        report, gen_timings = await self._reporter.generate(
            plan,
            aggregated,
            intermediate_reports=intermediate_reports or None,
            defer_summary=True,
        )
        timings["report_generate_ms"] = (time.perf_counter() - t) * 1000.0
        timings["report_sections_ms"] = gen_timings.get("report_sections_ms", 0.0)
        timings["report_summary_ms"] = gen_timings.get("report_summary_ms", 0.0)
        timing_budget = _record_budget_timings(gen_timings, settings, timings)
        await _emit_report_sections(self._emit, report)
        return report, timing_budget

    async def _run_final_quality_phase(
        self,
        report: ResearchReport,
        aggregated: AggregatedSources,
        timing_budget: dict[str, Any],
        timings: dict[str, float],
    ) -> QualityScore | None:
        quality, quality_timings = await self._execute_quality_evaluation(
            report,
            aggregated,
            phase_name="quality_evaluation",
            timing_prefix="quality_",
            report_char_limit=timing_budget["quality_report_chars"],
            fact_section_char_limit=timing_budget["quality_fact_section_chars"],
            fact_source_char_limit=timing_budget["quality_fact_source_chars"],
        )
        if quality:
            timings.update(quality_timings)
            report.metadata.quality_overall = quality.overall
        return quality

    @staticmethod
    def _build_pipeline_result(
        report: ResearchReport,
        quality: QualityScore | None,
        conflicts: list[ConflictRecord],
        depth_val: str,
        timings: dict[str, float],
        t_run0: float,
    ) -> ReportPipelineResult:
        return _build_pipeline_result(report, quality, conflicts, depth_val, timings, t_run0)

    async def _execute_url_validation(
        self,
        aggregated: AggregatedSources,
        report: ResearchReport,
    ) -> float:
        """Execute URL validation and return elapsed time in milliseconds.

        URL reachability is the only remaining validation in the pipeline
        (the per-section content validator was removed on 2026-04-22).  It
        reads from aggregated sources and mutates report citations only,
        so it can safely run concurrently with summary completion.
        """
        await self._emit("research.pipeline_phase", phase="url_validation")
        t = time.perf_counter()
        await self._validate_urls(aggregated, report)
        return (time.perf_counter() - t) * 1000.0

    async def _execute_quality_evaluation(
        self,
        report: ResearchReport,
        aggregated: AggregatedSources,
        *,
        phase_name: str,
        timing_prefix: str,
        report_char_limit: int | None = None,
        fact_section_char_limit: int | None = None,
        fact_source_char_limit: int | None = None,
    ) -> tuple[QualityScore | None, dict[str, float]]:
        try:
            await self._emit("research.pipeline_phase", phase=phase_name)
            t = time.perf_counter()
            quality, quality_timings = await self._evaluator.evaluate_with_breakdown(
                report,
                aggregated,
                report_char_limit=report_char_limit,
                fact_section_char_limit=fact_section_char_limit,
                fact_source_char_limit=fact_source_char_limit,
            )
            elapsed_ms = (time.perf_counter() - t) * 1000.0
            timings = {f"{timing_prefix}ms": elapsed_ms}
            timings.update(
                {
                    f"{timing_prefix}{key[len('quality_') :]}"
                    if key.startswith("quality_")
                    else f"{timing_prefix}{key}": value
                    for key, value in quality_timings.items()
                }
            )
            timings[f"{timing_prefix}score"] = quality.overall
            return quality, timings
        except Exception:
            logger.warning(
                "Quality evaluation failed — skipping (report still usable)",
                exc_info=True,
            )
            return None, {}

    async def _detect_conflicts(
        self,
        search_results: list[SearchResult],
        settings: ResearchSettings,
    ) -> list[ConflictRecord]:
        """Detect contradictions between sub-question answers.

        Only runs for deep mode since sub-questions intentionally cover
        different aspects — pairwise comparison is only warranted when
        thoroughness justifies the cost.  Resolution always uses fast
        source-voting (no LLM call) to avoid the O(n²) latency explosion.
        """
        if settings.depth != "deep" or len(search_results) < 2:
            return []
        agent_results = [
            AgentTaskResult(
                agent_id=sr.question_id,
                task=sr.question_id,
                output=sr.summary,
                success=bool(sr.sources),
            )
            for sr in search_results
        ]
        conflicts: list[ConflictRecord] = []
        try:
            detected = await self._conflict_resolver.detect(agent_results)
            for conflict in detected:
                resolution = self._conflict_resolver._resolve_via_voting(conflict)
                conflict.resolution = resolution
                await self._emit(
                    "research.conflict_detected",
                    agent_a=conflict.agent_a_id,
                    agent_b=conflict.agent_b_id,
                    method=resolution.method,
                    confidence=resolution.confidence,
                )
            conflicts = detected
        except Exception:
            logger.warning("Conflict detection failed", exc_info=True)
        return conflicts

    async def _validate_urls(
        self,
        aggregated: AggregatedSources,
        report: ResearchReport,
    ) -> None:
        urls = [s.url for s in aggregated.sources if s.url]
        if not urls:
            return
        try:
            vr = await self._url_validator.validate(urls)
            unreachable = {r.url for r in vr.results if not r.reachable}
            if not unreachable:
                return
            degraded_count = 0
            for src in aggregated.sources:
                if src.url in unreachable:
                    src.reliability_score = max(0.0, src.reliability_score - 0.3)
                    degraded_count += 1
            # Intentionally keep section.citations and report.references
            # intact: removing refs strips inline citations from the final
            # article, destroying evidence grounding that RACE evaluates.
            # FACT evaluation handles URL validity independently.
            await self._emit(
                "research.url_validation",
                total=vr.total,
                reachable=vr.reachable,
                unreachable=vr.unreachable,
                degraded_refs=degraded_count,
                error_rate=vr.error_rate,
            )
        except Exception:
            logger.warning("URL validation failed", exc_info=True)


def _build_pipeline_result(
    report: ResearchReport,
    quality: QualityScore | None,
    conflicts: list[ConflictRecord],
    depth_val: str,
    timings: dict[str, float],
    t_run0: float,
) -> ReportPipelineResult:
    timings["total_ms"] = (time.perf_counter() - t_run0) * 1000.0
    logger.info(
        "research.report_pipeline phase=done depth=%s timings_ms=%s",
        depth_val,
        {k: round(v, 1) for k, v in timings.items()},
    )
    rounded_timings = {k: round(v, 1) for k, v in timings.items()}
    return ReportPipelineResult(
        report=report,
        quality=quality,
        conflicts=conflicts,
        phase_timings_ms=rounded_timings,
    )


async def _emit_report_sections(emit: EmitFn, report: ResearchReport) -> None:
    for section in report.sections:
        await emit(
            "research.report_section",
            chunk={
                "section_id": section.section_id,
                "title": section.title,
                "citations": len(section.citations),
            },
        )


def _record_budget_timings(
    gen_timings: dict[str, Any],
    settings: ResearchSettings,
    timings: dict[str, float],
) -> dict[str, Any]:
    section_input_metrics = gen_timings.get("section_input_metrics", [])
    timing_budget = _timing_budget(section_input_metrics, settings)
    timings["quality_budget_mode"] = 1.0 if timing_budget["quality_compact"] else 0.0
    return timing_budget


def _select_intermediate_targets(
    search_results: list[SearchResult],
    questions: list[Any],
    depth: str,
) -> set[str]:
    target = _INTERMEDIATE_TARGET_BY_DEPTH.get(depth, len(search_results))
    if len(search_results) <= target:
        return {sr.question_id for sr in search_results if sr.sources}
    priority_by_qid = {
        getattr(question, "question_id", ""): int(getattr(question, "priority", 0) or 0)
        for question in questions
    }
    ranked = sorted(
        search_results,
        key=lambda sr: (
            priority_by_qid.get(sr.question_id, 0),
            len(sr.sources),
            sr.coverage_score,
            1 if sr.error is None else 0,
        ),
        reverse=True,
    )
    return {sr.question_id for sr in ranked[:target] if sr.sources}


def _timing_budget(
    section_input_metrics: list[dict[str, Any]],
    settings: ResearchSettings,
) -> dict[str, Any]:
    metrics = list(section_input_metrics or [])
    ranked = sorted(
        metrics,
        key=lambda item: (
            int(item.get("relevant_source_count", 0) or 0),
            int(item.get("intermediate_context_chars", 0) or 0),
        ),
        reverse=True,
    )
    dense = any(
        int(item.get("relevant_source_count", 0) or 0) >= 24
        or int(item.get("intermediate_context_chars", 0) or 0) >= 1200
        for item in ranked
    )
    quality_compact = dense or len(ranked) >= 5
    return {
        "quality_compact": quality_compact,
        "quality_report_chars": 5500 if quality_compact else None,
        "quality_fact_section_chars": 2200 if quality_compact else None,
        "quality_fact_source_chars": 1400 if quality_compact else None,
    }
