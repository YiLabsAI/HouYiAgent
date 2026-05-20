"""Memory Engine data types."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryScope(str, Enum):
    """Scope of a memory record."""

    SESSION = "session"
    USER = "user"
    WORKSPACE = "workspace"


class MemoryType(str, Enum):
    """Semantic category of a memory record.

    The enum is split into two tiers by lifecycle and retrieval treatment:

    Required tier - present in every deployment and wired into
    the routing/fusion tables:
    - FACT: general-purpose factual claims (default bucket).
    - EVENT: time-bounded occurrences; must carry an event_time on the
    underlying AtomicFact and is served by the timeline index.
    - PREFERENCE: user preferences; longer-lived than typical facts and
    rarely invalidated by later facts on the same subject.
    - PROFILE: stable identity attributes; high retrieval priority.
    - PROCEDURE: "how to do" knowledge; shared across agents.
    - CONSTRAINT: hard rules injected with elevated priority.
    - STRATEGY: products of the evolution pipeline with their own
    lifecycle (e.g. rubric scores, success counts).

    Optional extension tier (domain-specific):
    - PROJECT: project-scoped metadata; kept for backwards compatibility
    and can otherwise be expressed as a FACT plus scope.
    - CODE_STRUCTURE: coding-agent specific; not enabled by default.

    Values deliberately not included here (because they belong on the
    unit_kind axis rather than the semantic-bucket axis): OBSERVATION,
    REFLECTION, TRAJECTORY. A trajectory of tool calls, for example, is
    expressed as (unit_kind=TRAJECTORY, memory_type=PROCEDURE).
    """

    # Required tier
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"
    PROFILE = "profile"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    STRATEGY = "strategy"

    # Optional extension tier
    PROJECT = "project"
    CODE_STRUCTURE = "code_structure"


class CandidateStatus(str, Enum):
    """Lifecycle status of a memory candidate."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"


class RecallMatchMethod(str, Enum):
    """How a memory was matched during retrieval."""

    LEXICAL = "lexical"
    EMBEDDING = "embedding"
    RULE = "rule"
    HYBRID = "hybrid"


class MemoryLifecyclePolicy(str, Enum):
    """Agent memory lifecycle on termination."""

    PROMOTE_ON_COMPLETE = "promote"
    DISCARD_ON_FAILURE = "discard"
    PERSIST_SUB_SCOPE = "persist"
    EPHEMERAL = "ephemeral"


class MemorySourceKind(str, Enum):
    """Normalized source families supported by memory candidate building."""

    CONVERSATION = "conversation"
    SEARCH = "search"
    AUTO_DREAM = "auto_dream"


# ---------------------------------------------------------------------------
# Supporting value objects
# ---------------------------------------------------------------------------


class MemoryProvenance(BaseModel):
    """Tracks where a memory was extracted from."""

    source_type: str = "conversation"
    source_ids: list[str] = Field(default_factory=list)
    extracted_by: str = "unknown"
    extraction_timestamp: float = Field(default_factory=time.time)


class RelevanceDetail(BaseModel):
    """Breakdown of retrieval scoring components."""

    lexical_score: float = 0.0
    embedding_score: float = 0.0
    recency_score: float = 0.0
    rule_bonus: float = 0.0
    final_score: float = 0.0


class DedupMatch(BaseModel):
    """Result of deduplication check against an existing memory."""

    existing_memory_id: str
    similarity: float
    relation: str  # "duplicate" | "conflict" | "update" | "complement"


class TTLPolicy(BaseModel):
    """Default TTL configuration per memory type."""

    default_ttl: float | None = None
    per_type: dict[str, float | None] = Field(default_factory=dict)


class MemoryBuildItem(BaseModel):
    """A normalized source item that may become a memory candidate."""

    content: str = ""
    role: str = ""
    source_ids: list[str] = Field(default_factory=list)
    source_context: str = ""
    suggested_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_type: MemoryType | None = None
    confidence: float | None = None


class MemoryBuildInput(BaseModel):
    """Generic input payload for memory candidate building."""

    source_type: MemorySourceKind = MemorySourceKind.CONVERSATION
    scope: MemoryScope = MemoryScope.USER
    source_context: str = ""
    items: list[MemoryBuildItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class MemoryRecord(BaseModel):
    """A single memory entry."""

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    scope: MemoryScope = MemoryScope.SESSION
    key: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    ttl: float | None = None
    memory_type: MemoryType = MemoryType.FACT
    tags: list[str] = Field(default_factory=list)
    valid_from: float | None = None
    valid_to: float | None = None
    confidence: float = 1.0
    decay: float = 1.0
    provenance: MemoryProvenance | None = None
    embedding: list[float] | None = None

    @property
    def is_expired(self) -> bool:
        """Check if this record has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl


class MemoryCandidate(BaseModel):
    """A memory extraction candidate awaiting approval."""

    candidate_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    scope: MemoryScope = MemoryScope.USER
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    source_type: str = MemorySourceKind.CONVERSATION.value
    source_message_ids: list[str] = Field(default_factory=list)
    source_context: str = ""
    confidence: float = 0.0
    extracted_at: float = Field(default_factory=time.time)
    status: CandidateStatus = CandidateStatus.PENDING
    suggested_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    dedup_matches: list[DedupMatch] | None = None


class MemoryRecall(BaseModel):
    """A single memory hit from retrieval."""

    memory_id: str
    score: float = 0.0
    matched_by: RecallMatchMethod = RecallMatchMethod.HYBRID
    injection_slot: str = "memory_recalls"
    explanation: str = ""
    relevance_detail: RelevanceDetail = Field(default_factory=RelevanceDetail)


class MemoryPolicy(BaseModel):
    """Controls extraction and retrieval behaviour."""

    auto_approve: bool = False
    max_recalls_per_turn: int = 5
    scope_priority: list[MemoryScope] = Field(
        default_factory=lambda: [
            MemoryScope.SESSION,
            MemoryScope.USER,
            MemoryScope.WORKSPACE,
        ]
    )
    ttl_policy: TTLPolicy = Field(default_factory=TTLPolicy)
    extraction_interval: int = 3
    min_confidence: float = 0.6


class ForgettingPolicy(BaseModel):
    """Active forgetting configuration.

    decay(t) = initial_decay * exp(-rate * days_since_last_access)
    """

    natural_decay_enabled: bool = True
    natural_decay_threshold: float = 0.1
    natural_decay_rates: dict[str, float] = Field(
        default_factory=lambda: {
            # Required tier: smaller rate = more stable memory.
            "profile": 0.001,
            "preference": 0.005,
            "fact": 0.01,
            "event": 0.015,
            "procedure": 0.01,
            "constraint": 0.0,
            "strategy": 0.002,
            # Optional extension tier.
            "project": 0.02,
            "code_structure": 0.01,
        }
    )
    conflict_supersede: bool = True
    explicit_forget: bool = True
    capacity_eviction_enabled: bool = True
    max_memories_per_scope: int = 10000


# ---------------------------------------------------------------------------
# Context types for pipeline stages
# ---------------------------------------------------------------------------


class ExtractionContext(BaseModel):
    """Context passed to the extraction pipeline."""

    session_id: str = ""
    user_id: str = ""
    workspace_id: str = ""
    turn_index: int = 0
    active_tags: list[str] = Field(default_factory=list)


class SessionContext(BaseModel):
    """Context passed to the retrieval pipeline."""

    session_id: str = ""
    user_id: str = ""
    workspace_id: str = ""
    recent_topics: list[str] = Field(default_factory=list)
    active_tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Atomic fact schema (structured payload for atomic memory units)
# ---------------------------------------------------------------------------


class Certainty(str, Enum):
    """Confidence tier attached to an AtomicFact.

    Used to route facts between the main store and the candidate inbox:
    - CERTAIN: admitted to the main store directly.
    - PROBABLE: admitted with reduced retrieval weight.
    - VAGUE: rejected from the main store; kept only as a candidate.
    """

    CERTAIN = "certain"
    PROBABLE = "probable"
    VAGUE = "vague"


class AtomicFact(BaseModel):
    """Structured payload for an atomic-level memory unit.

    An AtomicFact carries a single subject/predicate/object triple together
    with bi-temporal validity, source anchoring, and a certainty tier. It is
    the smallest unit that the writer pipeline can persist, deduplicate,
    update, and invalidate. Higher-level units (observations, reflections)
    aggregate one or more AtomicFacts.

    Field groups:
    - Triple: subject / predicate / object form the semantic core
    consumed by factual and relational retrieval paths.
    - Time: event_time is a free-form description of when the event
    occurred (e.g. "2023-05" or "2023-05-01..2023-05-15");
    valid_from / valid_to are epoch seconds that define when the
    fact itself is considered active. valid_to is None means the fact
    is currently active.
    - Certainty: see Certainty.
    - Source: source_anchor links the fact back to the chunk or message
    that produced it. source_offset is an optional (start, end)
    character range into that source. keywords is a shortlist used
    for lexical boosting.
    - Qualifiers: optional string-keyed metadata for multi-argument facts
    that cannot be expressed by the core triple alone (location, amount,
    co-agent, etc.).

    Immutability: AtomicFact instances are treated as immutable value
    objects. Updates must produce a new instance rather than mutating
    existing fields; see the writer pipeline for UPDATES / REPLACES
    semantics.
    """

    # Triple (required)
    subject: str
    predicate: str
    object: str

    # Temporal (all optional)
    event_time: str | None = None
    valid_from: float | None = None
    valid_to: float | None = None

    # Certainty (required; no default to force an explicit decision)
    certainty: Certainty

    # Source (source_anchor required; missing anchor must yield schema_invalid)
    source_anchor: str
    source_offset: tuple[int, int] | None = None
    keywords: list[str] = Field(default_factory=list)

    # Extensibility for multi-argument facts
    qualifiers: dict[str, str] | None = None

    # Write semantics: when True the object is appended to any existing value
    # for the same (subject, predicate) pair rather than replacing it.
    # Use for open-ended sets (visited places, collected items, known contacts).
    # Leave False (default) for single-valued attributes (current job, address).
    accumulate: bool = False

    @field_validator("subject", "predicate", "object", "source_anchor", mode="before")
    @classmethod
    def _require_nonempty_string(cls, value: object) -> str:
        """Reject empty or whitespace-only strings on required text fields.

        Runs in before mode so callers that pass non-string types get a
        consistent ValueError rather than a later coercion surprise.
        """
        if not isinstance(value, str):
            raise ValueError("expected a string value")
        text: str = value
        stripped = text.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: object) -> list[str]:
        """Drop empty/whitespace entries and strip surrounding whitespace.

        Keeps downstream lexical boosting free of blank tokens without
        forcing callers to pre-clean their inputs.
        """
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("keywords must be a list of strings")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("keywords must contain only strings")
            trimmed = item.strip()
            if trimmed:
                cleaned.append(trimmed)
        return cleaned

    @model_validator(mode="after")
    def _validate_invariants(self) -> AtomicFact:
        """Cross-field invariants that cannot be expressed on single fields.

        Enforced rules:
        - valid_to must be >= valid_from when both are set.
        - source_offset end must be >= start and both must be >= 0.
        """
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to < self.valid_from
        ):
            raise ValueError("valid_to must be >= valid_from")

        if self.source_offset is not None:
            start, end = self.source_offset
            if start < 0 or end < 0:
                raise ValueError("source_offset bounds must be non-negative")
            if end < start:
                raise ValueError("source_offset end must be >= start")

        return self

    @property
    def is_active(self) -> bool:
        """Whether this fact is currently considered in force.

        A fact is active when valid_to is unset (open-ended validity).
        Facts that have been invalidated carry a concrete valid_to and
        are therefore treated as historical.
        """
        return self.valid_to is None

    @property
    def admits_to_main_store(self) -> bool:
        """Whether the writer pipeline should persist this fact to the main store.

        Vague facts are always routed to the candidate inbox instead so the
        main store never mixes high-confidence facts with guesswork.
        """
        return self.certainty is not Certainty.VAGUE


# ---------------------------------------------------------------------------
# Entity state view (materialized current/historical entity attributes)
# ---------------------------------------------------------------------------


class EntityStateRecord(BaseModel):
    """A single row in the materialized entity-state view.

    Each row represents the value of one (namespace, entity, attribute)
    triple over a closed-open validity interval [valid_from, valid_to).
    A row whose valid_to is None is considered the currently active
    version. The view is designed to be queried directly by the recall
    fast-path (factual lookup, single-hop, temporal as-of) without touching
    the AtomicFact store.

    The triple shape mirrors AtomicFact:
    - namespace separates tenants (workspace / user / session id).
    - entity corresponds to AtomicFact.subject.
    - attribute corresponds to AtomicFact.predicate.
    - value is the stringified AtomicFact.object.
    - source_unit_id links back to the AtomicFact / MemoryUnit that
    produced this state row, so retrieval can fetch the original
    provenance and source anchor on demand.
    """

    state_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    namespace: str
    entity: str
    attribute: str
    value: str
    certainty: Certainty = Certainty.CERTAIN
    valid_from: float = Field(default_factory=time.time)
    valid_to: float | None = None
    source_unit_id: str | None = None
    qualifiers: dict[str, str] | None = None
    created_at: float = Field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        """Active state has open-ended validity (valid_to is None)."""
        return self.valid_to is None


# ---------------------------------------------------------------------------
# Raw conversation turn log (L0 — synchronous write tier)
# ---------------------------------------------------------------------------


class RawTurn(BaseModel):
    """A single conversation turn in the raw-turn L0 log.

    The L0 tier ( / ) is the synchronous "always write,
    never throw away" layer: every user/assistant turn lands here
    verbatim before any extraction pipeline runs. The downstream L1
    extractor reads from this log; the LLM unknown-answer fallback may
    also rehydrate raw text when atomic facts are insufficient.

    turn_index is a per-(namespace, session_id) monotonic counter
    assigned by the storage layer at append time; callers should not set
    it. role follows the OpenAI chat convention (user /
    assistant / system / tool).
    """

    turn_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    namespace: str = "default"
    session_id: str
    turn_index: int = 0
    role: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
