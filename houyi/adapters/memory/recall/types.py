"""Value types for the memory recall pipeline.

These types are the stable contract between router, retrievers,
fusion, and unknown-answer guarding. Everything else in this package
depends on this module; this module depends on types.AtomicFact
and nothing else inside the recall layer.

- QueryType is the closed enum of six universal query types.
 The router classifies every incoming query into exactly one of these
 so dispatch has a total function from query type to retrieval plan.
- RecallReason is the closed enum used by the unknown-answer
 guard. The guard maps zero-or-more candidate signals to exactly one
 reason; callers should surface suggested_action to the prompt
 layer rather than silently filtering.
- RecallCandidate wraps an AtomicFact plus score,
 provenance and per-retriever signals. Wrapping rather than
 subclassing keeps the immutable Pydantic AtomicFact clean and
 lets fusion freely re-score / re-explain without mutating the
 underlying fact.
- RetrieverContext is the runtime carrier (namespace, as_of,
 budgets, optional LLM adapter for iterative recall). Passing it
 explicitly keeps the retrievers stateless and lets tests inject fakes
 without monkey-patching module globals.

This module deliberately defines a fresh RetrieverKind enum
rather than reusing the legacy RecallMatchMethod (LEXICAL /
EMBEDDING / RULE / HYBRID). The retriever taxonomy is structural
(which index served the hit) rather than scoring-method (lexical vs
embedding), and mixing the two would create lossy translations at
every fusion step.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from houyi.adapters.memory.types import AtomicFact

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QueryType(str, Enum):
    """Six universal query types for routing.

    The router maps every query to exactly one of these. Adding a new
    type changes routing and fusion behavior; do not extend casually.
    """

    FACTUAL_LOOKUP = "factual_lookup"
    """Single-entity attribute query, e.g. "Martin lives where?"."""

    RELATIONAL_CHAIN = "relational_chain"
    """Multi-hop reasoning over related entities."""

    TEMPORAL_QUERY = "temporal_query"
    """Time-bounded query: before/after/latest/as-of."""

    THEMATIC_SUMMARY = "thematic_summary"
    """Open-domain aggregation over a topic."""

    PROCEDURAL_RECALL = "procedural_recall"
    """How-to or strategy reuse queries."""

    NEGATION_CHECK = "negation_check"
    """Yes/no presence check used to avoid unsupported answers."""


class RecallReason(str, Enum):
    """Why the recall result is what it is — drives the prompt layer.

    These are the unknown-answer guard signals plus the positive
    SUFFICIENT outcome. The guard is the only producer of these
    values; retrievers and fusion never set them.
    """

    SUFFICIENT = "sufficient"
    """Candidates pass the guard; LLM may use them as-is."""

    NO_CANDIDATES = "no_candidates"
    """Zero hits across all retrievers."""

    LOW_EVIDENCE = "low_evidence"
    """Top score below EVIDENCE_THRESHOLD; candidates kept for trace."""

    CONTRADICTING_EVIDENCE = "contradicting_evidence"
    """Multiple candidates with CONTRADICTS edge and no recency winner."""

    EXPLICIT_ABSENCE = "explicit_absence"
    """negation_check query and EntityStateView has no active row."""


class RetrieverKind(str, Enum):
    """Which retriever produced a candidate — needed by fusion weights.

    These names mirror the file layout under recall/retrievers/ so
    log lines like matched_by=ENTITY_STATE are grep-able straight
    to the implementation. RAW_TURN is reserved for a raw-turn
    fallback so downstream weights can stay stable when that retriever
    is enabled. VECTOR is reserved for the two-stage FTS5 prefilter
    + sqlite-vec rerank retriever ( / ).
    """

    ENTITY_STATE = "entity_state"
    TIMELINE = "timeline"
    ITERATIVE = "iterative"
    RAW_TURN = "raw_turn"
    VECTOR = "vector"
    GRAPH = "graph"
    EVENT = "event"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class RecallQuery(BaseModel):
    """The user-facing query handed to the orchestrator.

    Kept deliberately small: the router decides the rest from text
    alone. Optional fields are caller hints, not requirements; the
    pipeline must work with just text.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    """The raw query string, language-agnostic."""

    namespace: str = "default"
    """Memory namespace to search; matches the writer namespace."""

    as_of: float | None = None
    """Wall-clock seconds for temporal queries; None means "now"."""

    entity_hint: str | None = None
    """Optional caller-supplied entity name to bias factual_lookup."""

    attribute_hint: str | None = None
    """Optional caller-supplied attribute to bias factual_lookup."""

    top_k: int = 5
    """Soft cap on candidates returned to the caller."""

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("RecallQuery.text must be non-empty")
        return v

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("RecallQuery.top_k must be > 0")
        return v


@runtime_checkable
class SourceChunkReader(Protocol):
    """Read original source text by provenance anchor."""

    def read_source_chunk(self, source_anchor: str) -> str | None:
        """Return source text for source_anchor or None when missing."""
        ...


class RetrieverContext(BaseModel):
    """Runtime carrier passed through router → retrievers → fusion.

    Stays separate from RecallQuery because the query is part
    of the request and the context is part of the wiring (the same
    query can be replayed offline against a different LLM adapter or
    namespace without re-typing it).

    The llm_adapter is the only non-Pydantic field — it is
    duck-typed (await llm_adapter.chat(messages, ...) returning an
    object with .content) to avoid pulling the adapter ABC into the
    recall layer's type surface.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    started_at: float = Field(default_factory=time.time)
    """Used by retrievers for deadline accounting; unset means no deadline."""

    deadline_ms: int | None = None
    """Soft deadline; retrievers SHOULD return early but MAY exceed."""

    query_type: QueryType | None = None
    """The classified type of the incoming query."""

    llm_adapter: Any | None = None
    """Required only by iterative retrievers; others ignore it."""

    source_reader: SourceChunkReader | None = None
    """Optional provenance reader used when structured evidence is weak."""

    max_source_reads: int = 3
    """Maximum source chunks a recall call may rehydrate."""

    debug_trace: bool = False
    """When true, retrievers attach extra detail to RecallCandidate.signals."""

    @field_validator("max_source_reads")
    @classmethod
    def _max_source_reads_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("RetrieverContext.max_source_reads must be > 0")
        return v


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class RecallCandidate(BaseModel):
    """One hit, post-retrieval, pre-fusion.

    Wraps an AtomicFact rather than copying its fields so the
    underlying 6-tuple stays the source of truth. score is the
    retriever-local score (e.g. exact-match=10.0 for EntityState,
    BM25-normalized for raw text); fusion produces a separate
    fused_score written into signals['fused'] so the original
    score is never lost.
    """

    model_config = ConfigDict(frozen=False)

    fact: AtomicFact
    """The atomic memory tuple; immutable inside its own model."""

    score: float = 0.0
    """Retriever-local score; semantics depend on matched_by."""

    matched_by: RetrieverKind
    """Which retriever produced this hit; drives fusion weight lookup."""

    retriever_name: str = ""
    """Human-readable label, e.g. "EntityStateRetriever[exact]"."""

    signals: dict[str, Any] = Field(default_factory=dict)
    """Per-retriever debug payload + fusion intermediates (e.g. boosts)."""

    explanation: str = ""
    """One-line explanation surfaced in the trace; never used for routing."""

    @field_validator("score")
    @classmethod
    def _score_finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf check
            raise ValueError("RecallCandidate.score must be finite")
        return v


class RecallResult(BaseModel):
    """Final output of the recall pipeline.

    The orchestrator returns exactly one RecallResult per query.
    Callers MUST inspect reason before using
    candidates. When reason != SUFFICIENT, candidates are
    kept for audit but the prompt layer must honor
    suggested_action.
    """

    model_config = ConfigDict(frozen=False)

    candidates: list[RecallCandidate] = Field(default_factory=list)
    """Top-K post-fusion hits; ordered by fused score, descending."""

    query_type: QueryType
    """The router's classification of the input query."""

    reason: RecallReason = RecallReason.SUFFICIENT
    """IDK guard's verdict; see RecallReason."""

    suggested_action: str = ""
    """Prompt-layer hint, e.g. "admit_unknown" or "ask_user_clarify"."""

    trace: dict[str, Any] = Field(default_factory=dict)
    """Per-stage debug info (router decision, fusion weights, guard signals)."""

    def is_sufficient(self) -> bool:
        """Convenience predicate: True iff guard passed."""
        return self.reason == RecallReason.SUFFICIENT

    def top(self) -> RecallCandidate | None:
        """Highest-scoring candidate or None when empty."""
        return self.candidates[0] if self.candidates else None
