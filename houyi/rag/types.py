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
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A knowledge graph entity."""

    entity_id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(default="unknown", description="Entity type")
    embedding: list[float] | None = Field(
        default=None, description="Entity embedding"
    )
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
    location: str = Field(
        default="", description="Location within file (line number, page, etc.)"
    )
    snippet: str = Field(default="", description="Relevant text snippet")
    score: float = Field(default=0.0, description="Relevance score")


class SearchResult(BaseModel):
    """A single search result."""

    chunk_id: str = Field(default="", description="Chunk ID (if from index)")
    content: str = Field(..., description="Result content")
    score: float = Field(default=0.0, description="Relevance score")
    source: Source | None = Field(default=None, description="Source reference")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Complete retrieval result with answer and sources."""

    answer: str = Field(..., description="Generated answer")
    sources: list[Source] = Field(default_factory=list, description="Citation sources")
    confidence: float = Field(
        default=0.0, description="Answer confidence score (0-1)"
    )
    search_results: list[SearchResult] = Field(
        default_factory=list, description="Raw search results"
    )
    mode_used: RAGMode = Field(
        default=RAGMode.AUTO, description="Which mode was used"
    )
    strategies_used: list[RetrievalStrategy] = Field(
        default_factory=list, description="Which strategies were used"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
