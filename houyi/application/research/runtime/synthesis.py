from __future__ import annotations

from dataclasses import dataclass

from houyi.application.research.runtime.intermediate import IntermediateReport
from houyi.application.research.runtime.report_pipeline import ReportPipeline
from houyi.application.research.types import (
    AggregatedSources,
    QualityScore,
    ResearchPlan,
    ResearchReport,
    ResearchSettings,
    SearchResult,
)
from houyi.application.runtime.conflict import ConflictRecord


@dataclass(slots=True)
class SynthesisResult:
    report: ResearchReport
    quality: QualityScore | None
    conflicts: list[ConflictRecord]
    phase_timings_ms: dict[str, float]


class SynthesisCoordinator:
    def __init__(self, report_pipeline: ReportPipeline) -> None:
        self._report_pipeline = report_pipeline

    async def run(
        self,
        *,
        plan: ResearchPlan,
        aggregated_sources: AggregatedSources,
        search_results: list[SearchResult],
        intermediate_reports: list[IntermediateReport] | None,
        settings: ResearchSettings,
    ) -> SynthesisResult:
        result = await self._report_pipeline.run(
            plan,
            aggregated_sources,
            search_results,
            intermediate_reports,
            settings,
        )
        return SynthesisResult(
            report=result.report,
            quality=result.quality,
            conflicts=result.conflicts,
            phase_timings_ms=result.phase_timings_ms,
        )
