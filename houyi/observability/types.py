"""Observability type definitions.

Defines Span schema with AI-native fields for LLM/Agent tracing.
Aligned with OpenTelemetry semantic conventions (OTel).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpanType(str, Enum):
    """Span type for hierarchical tracing.

    - execution: Root span for entire agent execution
    - node: IR node execution (LLM/TOOL/VERIFY/etc.)
    - llm: LLM call within a node
    - tool: Tool/skill invocation
    - retriever: RAG retrieval operation
    - retry: Failed retry attempt
    - internal: Tool-internal sub-operation (provider query, content fetch, etc.)
    """

    EXECUTION = "execution"
    NODE = "node"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    RETRY = "retry"
    INTERNAL = "internal"


class SpanStatus(str, Enum):
    """Span execution status."""

    OK = "ok"
    ERROR = "error"


class TokenUsage(BaseModel):
    """Token usage for LLM calls."""

    input: int = 0
    output: int = 0
    total: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            total=self.total + other.total,
        )


class CostInfo(BaseModel):
    """Cost information for LLM calls."""

    usd: float = 0.0
    currency: str = "USD"


class SpanEvent(BaseModel):
    """Event within a span."""

    name: str
    timestamp: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class SpanSchema(BaseModel):
    """Span schema for observability.

    Represents a single operation in the trace with AI-native fields.
    """

    # Core identifiers
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None

    # Timing
    start_time: float
    end_time: float | None = None

    # Status
    status: SpanStatus = SpanStatus.OK
    status_description: str | None = None

    # Span type and hierarchy
    span_type: SpanType = SpanType.NODE
    node_id: str | None = None  # IR node_id when span_type in (node, llm, tool, retriever)

    # AI-native fields (LLM)
    model: str | None = None
    provider: str | None = None
    tokens: TokenUsage | None = None
    cost: CostInfo | None = None
    cache_hit: bool | None = None

    # AI-native fields (Tool)
    tool_name: str | None = None

    # AI-native fields (Retriever)
    kb_name: str | None = None
    docs_count: int | None = None
    top_k: int | None = None

    # Parallel execution fields
    group_id: str | None = None  # Parallel group identifier
    lane_id: int | None = None  # Lane within parallel group
    seq: int | None = None  # Sequence number within lane

    # Checkpoint/restore lineage
    parent_trace_id: str | None = None  # Original trace if restored
    restore_checkpoint_id: str | None = None  # Checkpoint restored from
    replay_mode: bool = False  # Whether this is a replay

    # Generic attributes (for extensibility)
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Events within span
    events: list[SpanEvent] = Field(default_factory=list)

    # Children spans (for tree structure)
    children: list[SpanSchema] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get span duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0

    model_config = ConfigDict(use_enum_values=True)
