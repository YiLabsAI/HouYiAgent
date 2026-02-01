"""Tooling IR: response schemas for tool-calling outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolResultIR(BaseModel):
    """Normalized tool execution result."""

    call_id: str | None = Field(default=None, description="Tool call identifier")
    raw: dict[str, Any] = Field(default_factory=dict, description="Raw tool output")
    content: str = Field(default="", description="Serialized output for LLM consumption")
    is_error: bool = Field(default=False, description="Whether the tool result is an error")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Tool result metadata")


class ToolOverrideIR(BaseModel):
    """Tool override metadata produced by hooks."""

    model_config = ConfigDict(populate_by_name=True)

    from_tool: str | None = Field(default=None, alias="from", description="Original tool name")
    to: str | None = Field(default=None, description="Overridden tool name")
    allowed: bool = Field(default=False, description="Whether override was allowed")
    applied: bool = Field(default=False, description="Whether override was applied")


class ToolCallTraceIR(BaseModel):
    """Trace entry for a tool call round."""

    tool_name: str | None = Field(default=None, description="Resolved tool name")
    requested_tool_name: str | None = Field(default=None, description="LLM-requested tool name")
    tool_call_id: str | None = Field(default=None, description="Tool call identifier")
    args: dict[str, Any] = Field(default_factory=dict, description="Parsed tool arguments")
    result: ToolResultIR = Field(default_factory=ToolResultIR, description="Tool execution result")
    tool_override: ToolOverrideIR | None = Field(
        default=None,
        description="Tool override details when hooks attempt replacement",
    )


class ToolErrorIR(BaseModel):
    """Normalized tool error info exposed to UI/logging."""

    tool_name: str | None = Field(default=None, description="Resolved tool name")
    requested_tool_name: str | None = Field(default=None, description="LLM-requested tool name")
    tool_call_id: str | None = Field(default=None, description="Tool call identifier")
    error: dict[str, Any] | None = Field(default=None, description="Error payload")


class LLMToolCallOutputIR(BaseModel):
    """Output schema for LLM tool-calling responses."""

    type: Literal["llm_response"] = Field(default="llm_response", description="Output type")
    content: str = Field(default="", description="Final assistant content")
    tool_calls: list[ToolCallTraceIR] = Field(
        default_factory=list,
        description="Tool call trace entries",
    )
    finish_reason: str | None = Field(default=None, description="Final completion finish reason")
    tool_finish_reason: str | None = Field(default=None, description="Tool-call finish reason")
    tool_call_rounds: int = Field(default=0, description="Number of tool-call rounds")
    max_rounds_reached: bool = Field(default=False, description="Whether max rounds were hit")
    tool_errors: list[ToolErrorIR] = Field(default_factory=list, description="Tool errors")
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation messages (assistant/user/tool)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for observability (cache hits, ids, etc.)",
    )
    error: str | None = Field(default=None, description="Top-level error message")


class ToolNodeOutputIR(BaseModel):
    """Output schema for Tool nodes."""

    type: Literal["tool_result"] = Field(default="tool_result", description="Output type")
    output: dict[str, Any] = Field(default_factory=dict, description="Tool output payload")
    is_error: bool = Field(default=False, description="Whether the tool failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Tool metadata")
