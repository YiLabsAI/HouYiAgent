"""Post-search report pipeline: conflicts → report → validation → repair → quality.

Extracted from ``ResearchSession`` to keep the session orchestrator lean.
The pipeline is stateless — all mutable data flows in/out via parameters
and the returned ``ReportPipelineResult``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from houyi.application.research.intermediate import IntermediateReport, IntermediateReportGenerator
from houyi.application.research.quality import QualityEvaluator
from houyi.application.research.report import ReportGenerator
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

EmitFn = Callable[..., Awaitable[None]]


@dataclass
class ReportPipelineResult:
    """Immutable output of a single report pipeline run."""

    report: ResearchReport
    quality: QualityScore | None = None
    validation: ValidationReport | None = None
    conflicts: list[ConflictRecord] = field(default_factory=list)


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_intermediates(
        self,
        search_results: list[SearchResult],
        questions: list[Any],
        plan_query: str,
    ) -> list[IntermediateReport]:
        """Generate intermediate reports for each sub-question's results."""
        qid_to_text = {sq.question_id: sq.question for sq in questions}
        reports: list[IntermediateReport] = []
        for sr in search_results:
            q_text = qid_to_text.get(sr.question_id, "")
            try:
                ir = await self._intermediate_gen.generate(sr, q_text, plan_query)
                reports.append(ir)
                await self._emit(
                    "research.intermediate_report",
                    question_id=sr.question_id,
                    confidence=ir.confidence,
                    key_findings=ir.key_findings[:3],
                )
            except Exception:
                logger.warning(
                    "Intermediate report failed for %s",
                    sr.question_id,
                    exc_info=True,
                )
        return reports

    async def run(
        self,
        plan: ResearchPlan,
        aggregated: AggregatedSources,
        search_results: list[SearchResult],
        intermediate_reports: list[IntermediateReport] | None,
        settings: ResearchSettings,
    ) -> ReportPipelineResult:
        """Execute the full report pipeline and return results."""
        conflicts = await self._detect_conflicts(search_results)

        report = await self._reporter.generate(
            plan,
            aggregated,
            intermediate_reports=intermediate_reports or None,
        )
        for section in report.sections:
            await self._emit(
                "research.report_section",
                chunk={
                    "section_id": section.section_id,
                    "title": section.title,
                    "citations": len(section.citations),
                },
            )

        await self._validate_urls(aggregated, report)

        validation: ValidationReport | None = None
        if settings.depth in ("standard", "deep"):
            validation = await self._validator.validate(report, plan.query)
            if validation.sections_needing_rewrite > 0:
                await self._emit(
                    "research.validation_issues",
                    sections_flagged=validation.sections_needing_rewrite,
                    overall_score=validation.overall_score,
                )
                await self._repair_weak_sections(
                    validation,
                    report,
                    plan,
                    aggregated,
                )

        quality = await self._evaluator.evaluate(report, aggregated)
        if quality:
            report.metadata.quality_overall = quality.overall

        return ReportPipelineResult(
            report=report,
            quality=quality,
            validation=validation,
            conflicts=conflicts,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    async def _detect_conflicts(
        self,
        search_results: list[SearchResult],
    ) -> list[ConflictRecord]:
        if len(search_results) < 2:
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
                resolution = await self._conflict_resolver.resolve(conflict)
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
        repaired = 0
        for sv in validation.sections:
            if not sv.needs_rewrite:
                continue
            matching = [s for s in report.sections if s.title == sv.title]
            if not matching:
                continue
            outline_match = [o for o in plan.outline if o.title == sv.title]
            if not outline_match:
                continue
            outline_sec = outline_match[0]

            repair_sources = aggregated.sources[:20]
            if hasattr(sv, "suggested_queries") and sv.suggested_queries:
                extra = await self._re_search_for_repair(sv.suggested_queries)
                if extra:
                    repair_sources = list(repair_sources) + extra

            try:
                new_section = await self._reporter._generate_section(
                    plan.query,
                    outline_sec.title,
                    outline_sec.objective,
                    repair_sources,
                    intermediate_context="",
                )
                idx = report.sections.index(matching[0])
                report.sections[idx] = new_section
                repaired += 1
            except Exception:
                logger.warning("Repair failed for section %s", sv.title, exc_info=True)
        if repaired:
            await self._emit(
                "research.sections_repaired",
                repaired_count=repaired,
            )

    async def _re_search_for_repair(self, queries: list[str]) -> list[SourceReference]:
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
