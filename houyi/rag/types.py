"""Core data types for Houyi RAG."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RAGMode(str, Enum):
    """RAG operating mode."""

    AGENTIC = "agentic"
    INDEXED = "indexed"
    AUTO = "auto"


class RetrievalStrategy(str, Enum):
    """Retrieval strategy for Indexed mode."""

    BM25 = "bm25"
    VECTOR = "vector"
    GRAPH = "graph"


class QueryType(str, Enum):
    """Query type classification for strategy selection."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    CAUSAL = "causal"
    COMPLEX = "complex"


class Document(BaseModel):
    """A source document."""

    doc_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    source: str = Field(default="", description="Source path or URL")
    doc_type: str = Field(default="text", description="Document type (text/pdf/excel)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A document chunk with optional embedding."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    context_text: str = Field(
        default="",
        description="Contextual description (from Contextual Retrieval)",
    )
    start_idx: int = Field(default=0, description="Start position in document")
    end_idx: int = Field(default=0, description="End position in document")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A knowledge graph entity."""

    entity_id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(default="unknown", description="Entity type")
    embedding: list[float] | None = Field(default=None, description="Entity embedding")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """A relationship between entities."""

    rel_id: str = Field(..., description="Unique relation identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    rel_type: str = Field(..., description="Relation type (causal/temporal/contains)")
    weight: float = Field(default=1.0, description="Relation weight")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Source(BaseModel):
    """A citation source reference."""

    file_path: str = Field(..., description="Source file path")
    location: str = Field(default="", description="Location within file (line number, page, etc.)")
    snippet: str = Field(default="", description="Relevant text snippet")
    score: float = Field(default=0.0, description="Relevance score")


class SearchResult(BaseModel):
    """A single search result."""

    chunk_id: str = Field(default="", description="Chunk ID (if from index)")
    content: str = Field(..., description="Result content")
    score: float = Field(default=0.0, description="Relevance score")
    source: Source | None = Field(default=None, description="Source reference")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def file_path(self) -> str:
        """Get file path from source (convenience property)."""
        return self.source.file_path if self.source else ""


class RetrievalResult(BaseModel):
    """Complete retrieval result with answer and sources."""

    answer: str = Field(..., description="Generated answer")
    sources: list[Source] = Field(default_factory=list, description="Citation sources")
    confidence: float = Field(default=0.0, description="Answer confidence score (0-1)")
    search_results: list[SearchResult] = Field(
        default_factory=list, description="Raw search results"
    )
    mode_used: RAGMode = Field(default=RAGMode.AUTO, description="Which mode was used")
    strategies_used: list[RetrievalStrategy] = Field(
        default_factory=list, description="Which strategies were used"
    )
    quality: QualitySummary | None = Field(
        default=None, description="Quality assessment summary (v1.1)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualitySummary(BaseModel):
    """Quality assessment summary for search results (v1.1).

    Provides real-time quality feedback for the current search,
    including score distribution and relevance assessment.
    """

    min_score: float = Field(default=0.0, description="Minimum score in results")
    max_score: float = Field(default=0.0, description="Maximum score in results")
    avg_score: float = Field(default=0.0, description="Average score across results")
    above_threshold_count: int = Field(
        default=0, description="Count of results above 60% threshold"
    )
    total_count: int = Field(default=0, description="Total number of results")
    relevance: str = Field(
        default="unknown",
        description="Overall relevance assessment: high/medium/low/unknown",
    )
    coverage: str = Field(
        default="unknown",
        description="Coverage assessment: high/medium/low/unknown",
    )
    confidence_level: str = Field(
        default="unknown",
        description="Confidence level: high/medium/low/unknown",
    )
    suggestion: str | None = Field(default=None, description="Improvement suggestion if applicable")
    score_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Score distribution buckets: {'80-100': 5, '60-80': 3, ...}",
    )

    @classmethod
    def from_scores(cls, scores: list[float]) -> QualitySummary:
        """Create QualitySummary from a list of scores."""
        if not scores:
            return cls()

        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        above_threshold = sum(1 for s in scores if s >= 0.6)

        distribution = cls._build_score_distribution(scores)
        relevance = cls._assess_relevance(avg_score)
        coverage = cls._assess_coverage(above_threshold, len(scores))
        confidence_level = cls._assess_confidence(min_score, max_score, avg_score)
        suggestion = cls._build_suggestion(avg_score, above_threshold, len(scores))

        return cls(
            min_score=min_score,
            max_score=max_score,
            avg_score=avg_score,
            above_threshold_count=above_threshold,
            total_count=len(scores),
            relevance=relevance,
            coverage=coverage,
            confidence_level=confidence_level,
            suggestion=suggestion,
            score_distribution=distribution,
        )

    @staticmethod
    def _build_score_distribution(scores: list[float]) -> dict[str, int]:
        distribution = {"80-100": 0, "60-80": 0, "40-60": 0, "20-40": 0, "0-20": 0}
        for score in scores:
            if score >= 0.8:
                distribution["80-100"] += 1
            elif score >= 0.6:
                distribution["60-80"] += 1
            elif score >= 0.4:
                distribution["40-60"] += 1
            elif score >= 0.2:
                distribution["20-40"] += 1
            else:
                distribution["0-20"] += 1
        return distribution

    @staticmethod
    def _assess_relevance(avg_score: float) -> str:
        eps = 1e-9
        if avg_score >= 0.7 - eps:
            return "high"
        if avg_score >= 0.5 - eps:
            return "medium"
        return "low"

    @staticmethod
    def _assess_coverage(above_threshold: int, total_count: int) -> str:
        above_ratio = above_threshold / total_count if total_count else 0
        if above_ratio >= 0.6:
            return "high"
        if above_ratio >= 0.3:
            return "medium"
        return "low"

    @staticmethod
    def _assess_confidence(min_score: float, max_score: float, avg_score: float) -> str:
        score_range = max_score - min_score
        if score_range < 0.3 and avg_score >= 0.6:
            return "high"
        if score_range < 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _build_suggestion(avg_score: float, above_threshold: int, total_count: int) -> str | None:
        if avg_score < 0.5:
            return "Consider refining your query or expanding the knowledge base"
        if above_threshold < total_count * 0.5:
            return "Try more specific keywords for better results"
        return None
