from __future__ import annotations

from dataclasses import dataclass, field

from houyi.adapters.memory.types import MemoryCandidate
from houyi.application.research.runtime.intermediate import IntermediateReport
from houyi.application.research.types import (
    AggregatedSources,
    QualityScore,
    ResearchPlan,
    ResearchReport,
    SearchResult,
)
from houyi.application.research.validation import ValidationReport
from houyi.application.runtime.conflict import ConflictRecord


@dataclass(slots=True)
class ResearchRunState:
    plan: ResearchPlan | None = None
    search_results: list[SearchResult] = field(default_factory=list)
    intermediate_reports: list[IntermediateReport] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    aggregated_sources: AggregatedSources | None = None
    report: ResearchReport | None = None
    quality_score: QualityScore | None = None
    validation_report: ValidationReport | None = None
    memory_candidates: list[MemoryCandidate] = field(default_factory=list)
    error: str | None = None
    event_sequence: int = 0
    cancelled: bool = False
    execution_phase: str = "init"
