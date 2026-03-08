"""Tool-calling response assembly for console execution."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from houyi.application.tool_calling.tool_call_output import (
    assemble_tool_call_output,
    build_llm_cache_key,
)
from houyi.application.workflow.serialization import to_wire_data
from houyi.domain.skill.spec import SkillSpec
from houyi.interface.protocol.ir import ExecutionIR, NodeExecutionIR

from ..gateway.events import StreamingOutputEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolCallContext:
    session_id: str
    execution: ExecutionIR
    node_id: str
    node_exec: NodeExecutionIR
    messages: list[dict[str, Any]]
    response: Any
    tool_trace: list[dict[str, Any]]
    base_adapter: Any
    tool_model: str
    prompt: str
    user_content: str | None
    max_tool_calls: int
    skills: list[SkillSpec]
    final_chat_kwargs: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    llm_cache: dict[str, Any] | None = None
    created_at: datetime | None = None


class ToolCallResponseAssembler(ABC):
    """Abstract base for assembling tool-calling responses."""

    @abstractmethod
    async def assemble(self, context: ToolCallContext) -> None:
        """Assemble tool-calling outputs into node execution state."""


class ConsoleToolCallResponseAssembler(ToolCallResponseAssembler):
    """Console response assembler with streaming event emission."""

    def __init__(
        self,
        connection_manager: Any,
        record_llm_call: Callable[
            [str, str, str, str | list[dict[str, Any]], str, dict | None],
            None,
        ],
    ) -> None:
        self.connection_manager = connection_manager
        self.record_llm_call = record_llm_call

    def _build_llm_cache_key(
        self,
        *,
        adapter: Any,
        messages: list[Any],
        tools: list[dict[str, Any]],
        chat_kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        return build_llm_cache_key(
            adapter=adapter,
            messages=messages,
            tools=tools,
            chat_kwargs=chat_kwargs,
        )

    async def assemble(self, context: ToolCallContext) -> None:
        initial_response = context.response
        core_result = await assemble_tool_call_output(
            session_id=context.session_id,
            execution=context.execution,
            node_id=context.node_id,
            node_exec=context.node_exec,
            messages=context.messages,
            response=context.response,
            tool_trace=context.tool_trace,
            base_adapter=context.base_adapter,
            tool_model=context.tool_model,
            prompt=context.prompt,
            user_content=context.user_content,
            max_tool_calls=context.max_tool_calls,
            skills=context.skills,
            final_chat_kwargs=context.final_chat_kwargs,
            prompt_cache_key=context.prompt_cache_key,
            llm_cache=context.llm_cache,
        )

        output_payload = core_result.output_payload
        content = core_result.content
        messages_for_log = core_result.messages_for_log
        tool_call_rounds = core_result.tool_call_rounds
        normalized_tool_trace = core_result.normalized_tool_trace
        tool_errors = core_result.tool_errors

        max_rounds_reached = tool_call_rounds >= context.max_tool_calls
        tool_finish_reason = (
            getattr(initial_response, "finish_reason", None) if initial_response else None
        )
        finish_reason = (
            getattr(context.response, "finish_reason", None) if context.response else None
        )

        if content:
            context.node_exec.streaming_output += content
            stream_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=context.session_id,
                execution_id=context.execution.execution_id,
                node_id=context.node_id,
                chunk=content,
                is_final=False,
            )
            await self.connection_manager.send_event(context.session_id, stream_event)

        context.node_exec.outputs = to_wire_data(output_payload)

        final_event = StreamingOutputEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=context.session_id,
            execution_id=context.execution.execution_id,
            node_id=context.node_id,
            chunk="",
            is_final=True,
            metadata={
                "trace_id": output_payload.metadata.get("trace_id"),
                "usage": output_payload.metadata.get("usage"),
            },
        )
        await self.connection_manager.send_event(context.session_id, final_event)

        self.record_llm_call(
            execution_id=context.execution.execution_id,
            node_id=context.node_id,
            model=context.tool_model,
            prompt=messages_for_log or (context.user_content or ""),
            response=content,
            metadata={
                "prompt_cache_key": context.prompt_cache_key,
                "tool_calls": to_wire_data(normalized_tool_trace),
                "max_tool_calls": context.max_tool_calls,
                "tool_call_rounds": tool_call_rounds,
                "tool_finish_reason": tool_finish_reason,
                "finish_reason": finish_reason,
                "max_rounds_reached": max_rounds_reached,
                "tool_errors": to_wire_data(tool_errors),
                "tool_names": [skill.name for skill in context.skills],
                "tool_call_output": to_wire_data(output_payload),
            },
        )


__all__ = [
    "ConsoleToolCallResponseAssembler",
    "ToolCallContext",
    "ToolCallResponseAssembler",
]
