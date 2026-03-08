"""Context Engine data types.

Defines the core data structures for context planning, rendering, and tracking.
Phase 2 adds CompactionRecord/CompactionMetrics.
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


class ContextUsage(BaseModel):
    """Token usage snapshot for a context window."""

    model: str = ""
    max_context_tokens: int = 0
    used_tokens: int = 0
    reserved_output_tokens: int = 0
    available_tokens: int = 0
    block_breakdown: dict[str, int] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    @property
    def utilization(self) -> float:
        """Context window utilization ratio (0.0 ~ 1.0)."""
        if self.max_context_tokens <= 0:
            return 0.0
        return self.used_tokens / self.max_context_tokens


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
