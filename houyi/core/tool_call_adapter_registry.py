"""Adapter registry for tool-call LLM adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from houyi.core.skill import SkillSpec
from houyi.core.tool_call_adapter import normalize_adapter_response
from houyi.core.tool_call_adapter_hooks import ToolCallAdapterContext, resolve_tool_call_adapter
from houyi.llm.models import ADAPTER_REAL


@dataclass(frozen=True)
class ToolCallAdapterRequest:
    """Request for building a tool-call adapter."""

    adapter_name: str = ADAPTER_REAL
    tool_names: list[str] = field(default_factory=list)
    skills: list[SkillSpec] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    parallel_tool_calls: bool | None = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: ToolCallAdapterContext | None = None

    def __post_init__(self) -> None:
        # Backward compatibility: older call sites built requests via `context=...`.
        if self.context is None:
            return
        object.__setattr__(self, "adapter_name", self.context.adapter_name)
        object.__setattr__(self, "tool_names", list(self.context.tool_names))
        object.__setattr__(self, "skills", list(self.context.skills))
        object.__setattr__(self, "tool_sequence", list(self.context.tool_sequence))
        object.__setattr__(self, "parallel_tool_calls", self.context.parallel_tool_calls)
        object.__setattr__(self, "now", self.context.now)


class ToolCallAdapterRegistry:
    """Resolve tool-call adapters using registered hooks and fallback factories."""

    class _NormalizedAdapter:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        @property
        def inner(self) -> Any:
            return self._inner

        async def chat(
            self,
            messages: list[Any],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            response = await self._inner.chat(messages, tools=tools, **kwargs)
            return normalize_adapter_response(response)

    def resolve(
        self,
        request: ToolCallAdapterRequest,
        *,
        fallback_factory: Callable[[], Any],
    ) -> Any:
        context = request.context or ToolCallAdapterContext(
            adapter_name=request.adapter_name,
            tool_names=request.tool_names,
            skills=request.skills,
            tool_sequence=request.tool_sequence,
            parallel_tool_calls=request.parallel_tool_calls,
            now=request.now,
        )
        adapter = resolve_tool_call_adapter(context)
        if adapter is not None:
            return self._NormalizedAdapter(adapter)
        return self._NormalizedAdapter(fallback_factory())
