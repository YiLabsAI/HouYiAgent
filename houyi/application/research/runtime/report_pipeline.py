"""Post-search report pipeline: conflicts → report → validation → repair → quality.

Extracted from ``ResearchRuntime`` to keep the runtime orchestrator lean.
The pipeline is stateless — all mutable data flows in/out via parameters
and the returned ``ReportPipelineResult``.
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
    SourceReference,
)
from houyi.application.research.url_validator import URLValidator
from houyi.application.research.validation import ValidationAgent, ValidationReport
from houyi.application.runtime.conflict import AgentTaskResult, ConflictRecord, ConflictResolver
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)
_REPAIR_SECTION_CONCURRENCY = 2

EmitFn = Callable[..., Awaitable[None]]


@dataclass
class ReportPipelineResult:
    """Immutable output of a single report pipeline run."""

    report: ResearchReport
    quality: QualityScore | None = None
    validation: ValidationReport | None = None
    conflicts: list[ConflictRecord] = field(default_factory=list)
    phase_timings_ms: dict[str, float] = field(default_factory=dict)


class ReportPipeline:
    """Runs the post-search stages: conflict detection → report generation →
    URL validation → section repair → quality evaluation.

    Stateless: each ``run()`` call receives all required data and returns a
    ``ReportPipelineResult``.
    """

    def __init__(
        self,
        *,
        reporter: ReportGenerator,
        validator: ValidationAgent,
        evaluator: QualityEvaluator,
        url_validator: URLValidator,
        conflict_resolver: ConflictResolver,
        intermediate_gen: IntermediateReportGenerator,
        web_search: WebSearchService,
        emit: EmitFn,
    ) -> None:
        self._reporter = reporter
        self._validator = validator
        self._evaluator = evaluator
        self._url_validator = url_validator
        self._conflict_resolver = conflict_resolver
        self._intermediate_gen = intermediate_gen
        self._web_search = web_search
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
        reuse_by_question_id: dict[str, IntermediateReport] | None = None,
        checkpoint_question_ids: frozenset[str] | None = None,
    ) -> list[IntermediateReport]:
        """Generate intermediate reports per sub-question (parallel, with reuse on retry).

        When ``reuse_by_question_id`` and ``checkpoint_question_ids`` are set,
        sub-questions whose search was **skipped** via checkpoint reuse the
        prior intermediate report — avoids N redundant LLM calls on report-timeout
        retries (root cause of multi-minute report phases).
        """
        qid_to_text = {sq.question_id: sq.question for sq in questions}
        reuse = reuse_by_question_id or {}
        ck = checkpoint_question_ids or frozenset()
        t0 = time.perf_counter()
        reused_count = 0

        pending: list[tuple[SearchResult, str]] = []
        for sr in search_results:
            q_text = qid_to_text.get(sr.question_id, "")
            if sr.question_id in ck and sr.question_id in reuse:
                reused_count += 1
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
            "research.intermediate_reports phase=done elapsed_s=%.2f reused=%d generated=%d pending_llm=%d",
            elapsed,
            reused_count,
            len(gen_by_qid),
            len(pending),
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
        timings: dict[str, float] = {}
        t_run0 = time.perf_counter()

        try:
            await self._emit("research.pipeline_phase", phase="conflict_detection")
            t = time.perf_counter()
            conflicts = await self._detect_conflicts(search_results, settings)
            timings["conflicts_ms"] = (time.perf_counter() - t) * 1000.0

            await self._emit("research.pipeline_phase", phase="report_generation")
            t = time.perf_counter()
            report = await self._reporter.generate(
                plan,
                aggregated,
                intermediate_reports=intermediate_reports or None,
            )
            timings["report_generate_ms"] = (time.perf_counter() - t) * 1000.0
            for section in report.sections:
                await self._emit(
                    "research.report_section",
                    chunk={
                        "section_id": section.section_id,
                        "title": section.title,
                        "citations": len(section.citations),
                    },
                )

            validation: ValidationReport | None = None
            if settings.depth in ("standard", "deep"):
                url_elapsed, validation_result = await asyncio.gather(
                    self._execute_url_validation(aggregated, report),
                    self._execute_validation(report, plan.query),
                )
                timings["url_validate_ms"] = url_elapsed
                validation, validation_elapsed = validation_result
                timings["validation_ms"] = validation_elapsed
            else:
                timings["url_validate_ms"] = await self._execute_url_validation(aggregated, report)

            if validation and validation.sections_needing_rewrite > 0:
                await self._emit(
                    "research.validation_issues",
                    sections_flagged=validation.sections_needing_rewrite,
                    overall_score=validation.overall_score,
                )
                t = time.perf_counter()
                await self._repair_weak_sections(
                    validation,
                    report,
                    plan,
                    aggregated,
                )
                timings["repair_ms"] = (time.perf_counter() - t) * 1000.0

            quality = None
            try:
                await self._emit("research.pipeline_phase", phase="quality_evaluation")
                t = time.perf_counter()
                quality = await self._evaluator.evaluate(report, aggregated)
                timings["quality_ms"] = (time.perf_counter() - t) * 1000.0
            except Exception:
                logger.warning(
                    "Quality evaluation failed — skipping (report still usable)",
                    exc_info=True,
                )
            if quality:
                report.metadata.quality_overall = quality.overall

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
                validation=validation,
                conflicts=conflicts,
                phase_timings_ms=rounded_timings,
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

    async def _execute_url_validation(
        self,
        aggregated: AggregatedSources,
        report: ResearchReport,
    ) -> float:
        """Execute URL validation and return elapsed time in milliseconds.

        This is a standalone execution unit that can run concurrently with
        validation since it only reads from aggregated sources and mutates
        report citations (no overlap with validation logic).
        """
        await self._emit("research.pipeline_phase", phase="url_validation")
        t = time.perf_counter()
        await self._validate_urls(aggregated, report)
        return (time.perf_counter() - t) * 1000.0

    async def _execute_validation(
        self,
        report: ResearchReport,
        query: str,
    ) -> tuple[ValidationReport | None, float]:
        """Execute report validation and return (result, elapsed_ms).

        Runs concurrently with URL validation when depth permits. Isolated
        failure handling ensures a validation error does not cascade to
        other pipeline stages.
        """
        try:
            await self._emit("research.pipeline_phase", phase="validation")
            t = time.perf_counter()
            validation = await self._validator.validate(report, query)
            return validation, (time.perf_counter() - t) * 1000.0
        except Exception:
            logger.warning(
                "Validation stage failed — skipping (report still usable)",
                exc_info=True,
            )
            return None, 0.0

    async def _detect_conflicts(
        self,
        search_results: list[SearchResult],
        settings: ResearchSettings,
    ) -> list[ConflictRecord]:
        """Detect contradictions between sub-question answers.

        Only runs for ``deep`` mode since sub-questions intentionally cover
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
            broken_ref_ids: set[str] = set()
            for src in aggregated.sources:
                if src.url in unreachable:
                    src.reliability_score = max(0.0, src.reliability_score - 0.3)
                    if hasattr(src, "reference_id") and src.reference_id:
                        broken_ref_ids.add(src.reference_id)
            if broken_ref_ids:
                for section in report.sections:
                    section.citations = [
                        c for c in section.citations if c.reference_id not in broken_ref_ids
                    ]
                report.references = [
                    r for r in report.references if r.reference_id not in broken_ref_ids
                ]
            await self._emit(
                "research.url_validation",
                total=vr.total,
                reachable=vr.reachable,
                unreachable=vr.unreachable,
                removed_refs=len(broken_ref_ids),
                error_rate=vr.error_rate,
            )
        except Exception:
            logger.warning("URL validation failed", exc_info=True)

    async def _repair_weak_sections(
        self,
        validation: ValidationReport,
        report: ResearchReport,
        plan: ResearchPlan,
        aggregated: AggregatedSources,
    ) -> None:
        """Rewrite low-quality sections concurrently using bounded parallelism.

        Strategy:
        1. Collect all sections needing repair (read-only scan).
        2. Spawn parallel rewrite tasks with a semaphore to protect LLM API.
        3. Apply results atomically to avoid partial report state exposure.

        The semaphore value (_REPAIR_SECTION_CONCURRENCY) caps concurrent
        LLM calls to prevent rate-limit thrashing while still reducing
        tail latency for multi-section repairs.
        """
        sections_to_repair = self._collect_sections_to_repair(validation, report, plan)
        if not sections_to_repair:
            return

        semaphore = asyncio.Semaphore(_REPAIR_SECTION_CONCURRENCY)
        tasks = [
            self._rewrite_single_section(
                index=index,
                section_validation=section_validation,
                outline_section=outline_section,
                plan_query=plan.query,
                aggregated=aggregated,
                semaphore=semaphore,
            )
            for index, section_validation, outline_section in sections_to_repair
        ]
        repaired_sections = await asyncio.gather(*tasks)

        repaired = 0
        for item in repaired_sections:
            if item is None:
                continue
            index, new_section = item
            if 0 <= index < len(report.sections):
                report.sections[index] = new_section
                repaired += 1

        if repaired:
            await self._emit(
                "research.sections_repaired",
                repaired_count=repaired,
            )

    def _collect_sections_to_repair(
        self,
        validation: ValidationReport,
        report: ResearchReport,
        plan: ResearchPlan,
    ) -> list[tuple[int, Any, Any]]:
        """Scan validation results and return (index, validation, outline) tuples.

        Filters out sections that do not need rewrite or lack matching
        outline sections. The returned list drives the parallel rewrite phase.
        """
        sections_to_repair: list[tuple[int, Any, Any]] = []
        for section_validation in validation.sections:
            if not section_validation.needs_rewrite:
                continue
            report_index = self._find_report_section_index(report, section_validation.title)
            if report_index is None:
                continue
            outline_section = self._find_outline_section(plan, section_validation.title)
            if outline_section is None:
                continue
            sections_to_repair.append((report_index, section_validation, outline_section))
        return sections_to_repair

    def _find_report_section_index(self, report: ResearchReport, title: str) -> int | None:
        for index, section in enumerate(report.sections):
            if section.title == title:
                return index
        return None

    def _find_outline_section(self, plan: ResearchPlan, title: str) -> Any | None:
        for section in plan.outline:
            if section.title == title:
                return section
        return None

    async def _rewrite_single_section(
        self,
        *,
        index: int,
        section_validation: Any,
        outline_section: Any,
        plan_query: str,
        aggregated: AggregatedSources,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, Any] | None:
        """Rewrite one section under semaphore guard.

        Acquires the semaphore before calling the LLM to enforce the concurrency
        limit. Returns (index, new_section) on success, None on failure so
        the caller can apply results atomically without partial mutations.
        """
        try:
            repair_sources = await self._gather_repair_sources(section_validation, aggregated)
            async with semaphore:
                new_section = await self._reporter._generate_section(
                    plan_query,
                    outline_section.title,
                    outline_section.objective,
                    repair_sources,
                    intermediate_context="",
                )
            return index, new_section
        except Exception:
            logger.warning("Repair failed for section %s", section_validation.title, exc_info=True)
            return None

    async def _gather_repair_sources(
        self,
        section_validation: Any,
        aggregated: AggregatedSources,
    ) -> list[SourceReference]:
        """Assemble source material for a section rewrite.

        Combines the top 20 aggregated sources with optional extra web search
        results when the validation suggests specific queries. The extra
        search is bounded to 2 queries to limit latency.
        """
        repair_sources = list(aggregated.sources[:20])
        if (
            hasattr(section_validation, "suggested_queries")
            and section_validation.suggested_queries
        ):
            extra = await self._search_extra_sources_for_repair(
                section_validation.suggested_queries
            )
            if extra:
                repair_sources.extend(extra)
        return repair_sources

    async def _search_extra_sources_for_repair(self, queries: list[str]) -> list[SourceReference]:
        """Fetch additional sources for repair via web search.

        Bounded to 2 queries and 3 results per query to keep latency
        predictable. Failures are logged but do not block the repair
        (best-effort enrichment).
        """
        extra_sources: list[SourceReference] = []
        for q in queries[:2]:
            try:
                response = await self._web_search.search(q, max_results=3, include_content=True)
                for r in response.results:
                    extra_sources.append(
                        SourceReference(
                            url=r.url,
                            title=r.title,
                            snippet=r.snippet,
                            source_type="web",
                            reliability_score=0.5,
                        )
                    )
            except Exception:
                logger.warning("Re-search failed for query: %s", q, exc_info=True)
        return extra_sources
