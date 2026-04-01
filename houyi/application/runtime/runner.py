"""AgentRunner: per-agent LLM tool-loop execution engine."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from houyi.application.runtime.context_strategy import ContextStrategy
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.domain.agent.spec import AgentSpec

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    """Result returned by ``AgentRunner.run``."""

    agent_id: str = ""
    task: str = ""
    output: Any = None
    success: bool = True
    error: str | None = None
    turns_used: int = 0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunner:
    """Executes a single agent's tool-loop with context management.

    Responsibilities:

    * Run up to ``max_turns`` iterations of LLM → tool-call → result.
    * Apply ``ContextStrategy`` to truncate / compress conversation context.
    * Emit ``AgentEvent`` at each lifecycle boundary.
    * Support both blocking (``run``) and streaming (``run_stream``) APIs.

    The runner is intentionally decoupled from multi-agent orchestration;
    ``SubAgentManager`` and ``AgentOrchestrator`` compose runners.
    """

    def __init__(
        self,
        spec: AgentSpec,
        *,
        llm_adapter: Any = None,
        tools: list[Any] | None = None,
        context_strategy: ContextStrategy | None = None,
        max_turns: int = 50,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self.spec = spec
        self.agent_id = f"agent_{spec.role}_{uuid.uuid4().hex[:6]}"
        self.llm_adapter = llm_adapter
        self.tools = tools or []
        self.context_strategy = context_strategy or ContextStrategy()
        self.max_turns = max_turns
        self.event_emitter = event_emitter

    async def run(self, task: str, *, session_state: dict[str, Any] | None = None) -> AgentResult:
        """Execute the agent's tool-loop and return the final result."""
        start = time.perf_counter()
        await self._emit(AgentEventType.AGENT_STARTED, {"task": task})

        turns = 0
        output: Any = None
        error: str | None = None
        success = True

        try:
            output, turns = await self._tool_loop(task, session_state or {})
        except Exception as exc:
            success = False
            error = str(exc)
            logger.exception("AgentRunner %s failed", self.agent_id)
            await self._emit(AgentEventType.AGENT_FAILED, {"error": error})
        else:
            await self._emit(AgentEventType.AGENT_COMPLETED, {"output": str(output)[:200]})

        elapsed = (time.perf_counter() - start) * 1000
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=output,
            success=success,
            error=error,
            turns_used=turns,
            duration_ms=elapsed,
        )

    async def run_stream(
        self, task: str, *, session_state: dict[str, Any] | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Streaming variant—yields events as they occur during execution."""
        collected: list[AgentEvent] = []
        original = self.event_emitter

        async def _collector(ev: AgentEvent) -> None:
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on_any(_collector)
        self.event_emitter = emitter

        try:
            start_ev = AgentEvent(
                event_type=AgentEventType.AGENT_STARTED,
                agent_id=self.agent_id,
                agent_name=self.spec.role,
                data={"task": task},
            )
            await emitter.emit(start_ev)
            yield start_ev

            turns = 0
            try:
                output, turns = await self._tool_loop(task, session_state or {})
                done_ev = AgentEvent(
                    event_type=AgentEventType.AGENT_COMPLETED,
                    agent_id=self.agent_id,
                    agent_name=self.spec.role,
                    data={"output": str(output)[:200], "turns": turns},
                )
                await emitter.emit(done_ev)
                yield done_ev
            except Exception as exc:
                fail_ev = AgentEvent(
                    event_type=AgentEventType.AGENT_FAILED,
                    agent_id=self.agent_id,
                    agent_name=self.spec.role,
                    data={"error": str(exc)},
                )
                await emitter.emit(fail_ev)
                yield fail_ev
        finally:
            self.event_emitter = original

    # ── Internal ───────────────────────────────────────────────

    async def _tool_loop(
        self,
        task: str,
        session_state: dict[str, Any],
    ) -> tuple[Any, int]:
        """Execute the agent's LLM → tool-call → result loop.

        When ``llm_adapter`` is a real ``LLMAdapter``, tool schemas are
        forwarded so the model can request tool invocations.  The loop
        continues until the model returns a response without tool_calls
        or ``max_turns`` is reached.

        Falls back to a deterministic mock when no adapter is configured.
        """
        messages: list[dict[str, Any]] = []
        if self.spec.system_prompt:
            messages.append({"role": "system", "content": self.spec.to_system_prompt()})
        messages.append({"role": "user", "content": task})

        tool_schemas = self._get_tool_schemas()
        turns = 0

        for _ in range(self.max_turns):
            turns += 1
            await self._emit(AgentEventType.LLM_STARTED, {"turn": turns})

            if self.llm_adapter is not None:
                response = await self._call_llm(messages, tool_schemas)
            else:
                response = f"Result for: {task}"

            await self._emit(AgentEventType.LLM_COMPLETED, {"turn": turns})

            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                if hasattr(response, "content") and hasattr(response, "tool_calls"):
                    return response.content, turns
                return response, turns

            assistant_content = getattr(response, "content", "") or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "") or tc.get("name", "")
                await self._emit(AgentEventType.TOOL_STARTED, {"tool": tool_name})
                tool_result = await self._execute_tool(tc)
                await self._emit(
                    AgentEventType.TOOL_COMPLETED,
                    {"tool": tool_name, "result": str(tool_result)[:100]},
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tool_name,
                        "content": str(tool_result),
                    }
                )

            self._apply_context_strategy(messages)

        return messages[-1].get("content", ""), turns

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Call the LLM adapter with optional tool schemas."""
        if hasattr(self.llm_adapter, "chat"):
            kwargs: dict[str, Any] = {}
            if tool_schemas:
                kwargs["tools"] = tool_schemas
            return await self.llm_adapter.chat(messages, **kwargs)
        if callable(self.llm_adapter):
            return await self.llm_adapter(messages)
        return f"Mock LLM response for {len(messages)} messages"

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from LLM response.  Empty list means "done".

        Handles ``LLMResponse`` objects (via attribute access), raw dicts,
        and plain-string mock responses.
        """
        tc = getattr(response, "tool_calls", None)
        if tc:
            return tc
        if isinstance(response, dict) and "tool_calls" in response:
            return response["tool_calls"]
        return []

    async def _execute_tool(self, tool_call: dict[str, Any]) -> Any:
        """Execute a tool call against registered tools.

        Supports both OpenAI-style ``{"function": {"name": ..., "arguments": ...}}``
        and flat ``{"name": ..., "arguments": ...}`` formats.
        """
        fn = tool_call.get("function", {})
        name = fn.get("name", "") or tool_call.get("name", "")
        args = fn.get("arguments", tool_call.get("arguments", {}))
        if isinstance(args, str):
            import json

            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}

        for tool in self.tools:
            tool_name = getattr(tool, "name", getattr(tool, "__name__", ""))
            if tool_name == name and callable(tool):
                result = tool(**args) if not _is_async(tool) else await tool(**args)
                return result
        return f"Tool {name!r} not found"

    def _get_tool_schemas(self) -> list[dict[str, Any]] | None:
        """Collect OpenAI-format tool schemas from registered tools and skills."""
        schemas: list[dict[str, Any]] = []
        for tool in self.tools:
            schema = getattr(tool, "schema", None)
            if schema:
                schemas.append(schema)
            elif hasattr(tool, "to_tool_schema"):
                schemas.append(tool.to_tool_schema())
        schemas.extend(self.spec.get_tool_schemas())
        return schemas or None

    def _apply_context_strategy(self, messages: list[dict[str, Any]]) -> None:
        """Apply context truncation per ``ContextStrategy``."""
        keep = self.context_strategy.keep_tool_result
        if keep < 0 or len(messages) <= keep + 1:
            return
        user_msg = messages[0]
        recent = messages[-(keep):]
        messages.clear()
        messages.append(user_msg)
        messages.extend(recent)

    async def _emit(self, event_type: AgentEventType, data: dict[str, Any]) -> None:
        if self.event_emitter is None:
            return
        event = AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            agent_name=self.spec.role,
            data=data,
        )
        await self.event_emitter.emit(event)


def _is_async(fn: Any) -> bool:
    """Check if *fn* is async — handles both functions and callable objects."""
    import asyncio

    if asyncio.iscoroutinefunction(fn):
        return True
    if not callable(fn):
        return False
    call_method = type(fn).__call__
    return asyncio.iscoroutinefunction(call_method)
