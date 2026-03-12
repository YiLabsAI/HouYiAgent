"""Context Engine data types.

Defines the core data structures for context planning, rendering, and tracking.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContextBlockType(str, Enum):
    """Type of context block in a ContextPlan."""

    SYSTEM = "system"
    SUMMARY = "summary"
    RECENT = "recent"
    PINNED = "pinned"
    TOOL_SUMMARY = "tool_summary"
    MEMORY = "memory"


class ContextSourceKind(str, Enum):
    """Origin/category of a context candidate before planning."""

    SYSTEM = "system"
    CURRENT_TURN = "current_turn"
    RECENT = "recent"
    PINNED = "pinned"
    TOOL_SUMMARY = "tool_summary"
    SUMMARY = "summary"
    MEMORY = "memory"


class ContextBlock(BaseModel):
    """A single block in a ContextPlan.

    Each block represents a logical segment of the context window
    (system instructions, recent messages, summaries, etc.).
    """

    block_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    block_type: ContextBlockType
    content: str | list[dict[str, Any]] = ""
    token_count: int = 0
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_message_list(self) -> bool:
        """Whether content is a list of message dicts (vs. a plain string)."""
        return isinstance(self.content, list)


class TaskBoundary(BaseModel):
    """Task-scoped boundary metadata for one planning run."""

    boundary_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_kind: str = "chat"
    scope: str = "conversation"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSelectionPolicy(BaseModel):
    """Planning policy that controls which candidate sources may be assembled."""

    policy_name: str = "default"
    allow_memory: bool = True
    allow_tool_summaries: bool = True
    allow_pinned: bool = True
    max_recent_messages: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextCandidate(BaseModel):
    """Structured context input consumed by ContextPlanner."""

    candidate_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: ContextSourceKind
    block_type: ContextBlockType
    content: str | list[dict[str, Any]] = ""
    pinned: bool = False
    priority: int = 0
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextUsage(BaseModel):
    """Token usage snapshot for a context window."""

    model: str = ""
    max_context_tokens: int = 0
    used_tokens: int = 0
    reserved_output_tokens: int = 0
    available_tokens: int = 0
    block_breakdown: dict[str, int] = Field(default_factory=dict)
    dropped_blocks: list[str] = Field(default_factory=list)
    drop_reasons: dict[str, str] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    @property
    def utilization(self) -> float:
        """Context window utilization ratio (0.0 ~ 1.0)."""
        if self.max_context_tokens <= 0:
            return 0.0
        return self.used_tokens / self.max_context_tokens


class NormalizedUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    cached_prompt_tokens: int = 0
    total_tokens: int = 0
    usage_confidence: str = "fallback"
    usage_source: str = "fallback"
    first_token_ms: float | None = None
    decode_tokens_per_second: float | None = None
    end_to_end_tokens_per_second: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BudgetDecision(BaseModel):
    context_window: int = 0
    input_budget: int = 0
    output_budget: int = 0
    reasoning_budget: int = 0
    answer_reserve: int = 0
    tool_reserve: int = 0
    should_set_max_tokens: bool = False
    max_tokens_to_send: int | None = None
    max_tokens_source: str | None = None
    finish_reason_policy: str = "provider_raw_passthrough"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannedContextUsage(ContextUsage):
    planned_prompt_tokens: int = 0
    available_input_tokens: int = 0


class CompactionMetrics(BaseModel):
    compression_ratio: float = 1.0
    retained_entity_coverage: float = 1.0
    summary_coherence_score: float | None = None
    pin_violation_count: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    messages_compacted: int = 0
    retained_refs_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompactionRecord(BaseModel):
    compaction_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    trigger: str = "threshold"
    summary: str = ""
    source_message_ids: list[str] = Field(default_factory=list)
    pinned_message_ids: list[str] = Field(default_factory=list)
    retained_refs: list[str] = Field(default_factory=list)
    metrics: CompactionMetrics = Field(default_factory=CompactionMetrics)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPlan(BaseModel):
    """Assembled context plan ready for rendering.

    A ContextPlan is the output of ContextPlanner: an ordered list of
    ContextBlocks that fit within the model's context window, plus
    usage tracking metadata.
    """

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    blocks: list[ContextBlock] = Field(default_factory=list)
    usage: ContextUsage = Field(default_factory=ContextUsage)
    created_at: float = Field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        """Total token count across all blocks."""
        return sum(b.token_count for b in self.blocks)

    def get_blocks_by_type(self, block_type: ContextBlockType) -> list[ContextBlock]:
        """Get all blocks of a given type."""
        return [b for b in self.blocks if b.block_type == block_type]
