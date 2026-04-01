"""Research domain types.

All Pydantic models and enums for the Deep Research engine:
plans, search, sources, reports, quality evaluation, and session lifecycle.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrchestrationMode(str, Enum):
    """Multi-agent orchestration strategy for a research session."""

    DELEGATE = "delegate"
    AUTONOMOUS = "autonomous"


class ResearchDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class SearchStrategy(str, Enum):
    WEB = "web"
    LOCAL_FILE = "local_file"
    RAG = "rag"
    MIXED = "mixed"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanEditOperation(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"
    SET_PRIORITY = "set_priority"
    SET_STRATEGY = "set_strategy"


class ResearchStatus(str, Enum):
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    EXECUTING = "executing"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportStyle(str, Enum):
    BRIEF = "brief"
    DETAILED = "detailed"
    ACADEMIC = "academic"


class ExportFormat(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    PPTX = "pptx"
    DOCX = "docx"


class ReportChunkType(str, Enum):
    SECTION_START = "section_start"
    SECTION_DELTA = "section_delta"
    SECTION_COMPLETE = "section_complete"
    SUMMARY = "summary"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Research Plan
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A single sub-question in a research plan."""

    question_id: str = Field(default_factory=lambda: f"sq_{uuid.uuid4().hex[:8]}")
    question: str
    priority: int = Field(default=3, ge=1, le=5)
    search_strategy: SearchStrategy = SearchStrategy.WEB
    expected_sources: int = 5
    depends_on: list[str] = Field(default_factory=list)


class OutlineSection(BaseModel):
    """A section in the research report outline."""

    section_id: str = Field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:8]}")
    title: str
    objective: str
    related_question_ids: list[str] = Field(default_factory=list)
    required_depth: str = "standard"


class ResearchSettings(BaseModel):
    """User-configurable settings for a research session."""

    depth: ResearchDepth = ResearchDepth.STANDARD
    orchestration_mode: OrchestrationMode = OrchestrationMode.DELEGATE
    max_agents: int = 5
    model_profile: str = "auto"
    report_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.MARKDOWN],
    )
    max_search_rounds: int = 5


class PlanEdit(BaseModel):
    """A single edit operation on a research plan."""

    op: PlanEditOperation
    question_id: str | None = None
    target_question: str | None = None
    after_question_id: str | None = None
    new_priority: int | None = None
    new_search_strategy: SearchStrategy | None = None
    reason: str | None = None


class ResearchPlan(BaseModel):
    """A research plan produced by the planner and optionally edited by the user."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    query: str
    version: int = 1
    outline: list[OutlineSection] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    settings: ResearchSettings = Field(default_factory=ResearchSettings)
    estimated_duration_min: int = 5
    created_at: float = Field(default_factory=time.time)
    status: PlanStatus = PlanStatus.DRAFT


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchContext(BaseModel):
    """Context passed to SearchCoordinator for a sub-question search."""

    session_id: str
    plan_id: str
    user_query: str
    plan_version: int = 1
    memory_recalls: list[dict] = Field(default_factory=list)
    prior_findings: list[str] = Field(default_factory=list)
    excluded_urls: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    max_results_per_round: int = 8
    max_total_sources: int = 100


class SearchHit(BaseModel):
    """A single hit from a web search round."""

    url: str
    title: str
    snippet: str = ""
    content: str | None = None
    provider: str = ""
    domain: str | None = None
    published_at: float | None = None
    rank: int = 0


class SearchRound(BaseModel):
    """One round of search within a sub-question investigation."""

    round_index: int
    queries: list[str]
    hits: list[SearchHit] = Field(default_factory=list)
    sufficient: bool = False
    rationale: str = ""


class SourceReference(BaseModel):
    """A deduplicated, scored information source."""

    reference_id: str = Field(default_factory=lambda: f"ref_{uuid.uuid4().hex[:8]}")
    url: str | None = None
    title: str = ""
    snippet: str = ""
    source_type: str = "web"
    reliability_score: float = 0.5
    accessed_at: float = Field(default_factory=time.time)


class SearchResult(BaseModel):
    """Result of investigating one sub-question across multiple search rounds."""

    question_id: str
    rounds: list[SearchRound] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    summary: str = ""
    coverage_score: float = 0.0
    exhausted: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Source Aggregation
# ---------------------------------------------------------------------------


class AggregatedSources(BaseModel):
    """Sources collected from all sub-questions, deduplicated and ranked."""

    sources: list[SourceReference] = Field(default_factory=list)
    deduplicated_count: int = 0
    grouped_by_question: dict[str, list[str]] = Field(default_factory=dict)
    coverage_by_question: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """An inline citation linking report text to a source."""

    reference_id: str
    text_span: str = ""
    context: str = ""


class ReportSection(BaseModel):
    """One section of the final research report."""

    section_id: str = Field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:8]}")
    title: str
    content: str = ""
    citations: list[Citation] = Field(default_factory=list)


class ReportChunk(BaseModel):
    """A streaming chunk emitted during report generation."""

    chunk_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:8]}")
    report_id: str = ""
    sequence: int = 0
    chunk_type: ReportChunkType = ReportChunkType.SECTION_DELTA
    section_id: str | None = None
    section_title: str | None = None
    content_delta: str = ""
    citations_added: list[Citation] = Field(default_factory=list)


class ReportMetadata(BaseModel):
    """Metadata attached to a completed report."""

    style: ReportStyle = ReportStyle.DETAILED
    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.MARKDOWN],
    )
    source_count: int = 0
    section_count: int = 0
    quality_overall: float | None = None
    generated_by_mode: OrchestrationMode = OrchestrationMode.DELEGATE
    duration_seconds: float | None = None


class ResearchReport(BaseModel):
    """The final structured research report."""

    report_id: str = Field(default_factory=lambda: f"rpt_{uuid.uuid4().hex[:8]}")
    title: str = ""
    summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    references: list[SourceReference] = Field(default_factory=list)
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)
    generated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Quality Evaluation (RACE + FACT)
# ---------------------------------------------------------------------------


class RACEScore(BaseModel):
    """RACE framework score (report quality)."""

    comprehensiveness: float = 0.0
    depth: float = 0.0
    instruction_following: float = 0.0
    readability: float = 0.0
    overall: float = 0.0


class FACTScore(BaseModel):
    """FACT framework score (citation quality)."""

    citation_accuracy: float = 0.0
    effective_citations: int = 0


class QualityDetail(BaseModel):
    """Granular feedback for one evaluation criterion."""

    criterion: str
    score: float
    max_score: float = 100.0
    reasoning: str = ""


class QualityScore(BaseModel):
    """Combined RACE + FACT quality evaluation."""

    race: RACEScore = Field(default_factory=RACEScore)
    fact: FACTScore = Field(default_factory=FACTScore)
    overall: float = 0.0
    details: list[QualityDetail] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Session Progress
# ---------------------------------------------------------------------------


class ResearchProgress(BaseModel):
    """Snapshot of a research session's execution progress."""

    status: ResearchStatus = ResearchStatus.PLANNING
    total_steps: int = 0
    completed_steps: int = 0
    active_agents: list[str] = Field(default_factory=list)
    sources_found: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    current_action: str | None = None
    last_event_sequence: int = 0
    error: str | None = None
