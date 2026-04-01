"""Memory Engine data types."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryScope(str, Enum):
    """Scope of a memory record."""

    SESSION = "session"
    USER = "user"
    WORKSPACE = "workspace"


class MemoryType(str, Enum):
    """Semantic category of a memory record."""

    PROFILE = "profile"
    PREFERENCE = "preference"
    PROJECT = "project"
    FACT = "fact"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"


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
    """Sub-Agent memory lifecycle on termination."""

    PROMOTE_ON_COMPLETE = "promote"
    DISCARD_ON_FAILURE = "discard"
    PERSIST_SUB_SCOPE = "persist"
    EPHEMERAL = "ephemeral"


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
    source_message_ids: list[str] = Field(default_factory=list)
    source_context: str = ""
    confidence: float = 0.0
    extracted_at: float = Field(default_factory=time.time)
    status: CandidateStatus = CandidateStatus.PENDING
    suggested_tags: list[str] = Field(default_factory=list)
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
            "profile": 0.001,
            "preference": 0.005,
            "fact": 0.01,
            "project": 0.02,
            "procedure": 0.01,
            "constraint": 0.0,
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
