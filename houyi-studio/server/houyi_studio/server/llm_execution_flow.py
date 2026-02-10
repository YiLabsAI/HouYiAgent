"""LLM execution flow for streaming, tool calls, and deterministic replay."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from uuid import uuid4

from houyi.llm.replay import (
    ReplayDecisionKind,
    decide_replay,
)
from houyi.llm.replay import (
    get_recorded_llm_response as replay_get_recorded_llm_response,
)
from houyi.llm.replay import (
    get_recorded_tool_call_output as replay_get_recorded_tool_call_output,
)
from houyi.llm.replay import (
    record_llm_call as replay_record_llm_call,
)
from houyi.observability import Span, SpanType, TraceContext
from houyi.observability.types import TokenUsage
from houyi.protocol.ir import ExecutionIR, NodeExecutionIR
from houyi.protocol.ir.checkpoint_ir import LLMCallLog
from houyi.protocol.ir.tooling_ir import LLMToolCallOutputIR

from .events import SpanUpdateEvent, StreamingOutputEvent
from .observation_service import ObservationService
from .tool_call_service import ToolCallService

logger = logging.getLogger(__name__)

AdapterFactory = Callable[[], Any]
SleepFunc = Callable[[float], Awaitable[None]]


class LLMExecutionFlow:
    """Handle LLM streaming, tool calls, and replay logic."""

    def __init__(
        self,
        *,
        observation_service: ObservationService,
        tool_call_service: ToolCallService | None,
        adapter_factory: AdapterFactory,
        sleep_func: SleepFunc | None = None,
    ) -> None:
        self._observation_service = observation_service
        self._tool_call_service = tool_call_service
        self._adapter_factory = adapter_factory
        self._sleep_func = sleep_func or asyncio.sleep
        self.llm_call_logs: dict[str, list[LLMCallLog]] = {}
        self._llm_cache: dict[str, str] = {}

    def set_tool_call_service(self, tool_call_service: ToolCallService) -> None:
        self._tool_call_service = tool_call_service

    def set_sleep_func(self, sleep_func: SleepFunc) -> None:
        self._sleep_func = sleep_func

    def _create_llm_span(
        self,
        model: str | None,
        provider: str | None = None,
        cache_hit: bool = False,
        replay_mode: str | None = None,
    ) -> Span | None:
        """Create LLM span as child of current trace context.

        Returns None if no active trace context (instrumentation disabled).
        """
        parent = TraceContext.current()
        if parent is None:
            return None

        return Span(
            name="llm.completion",
            parent=parent,
            span_type=SpanType.LLM,
            model=model,
            provider=provider,
            cache_hit=cache_hit,
            attributes={
                "llm.model": model,
                "llm.provider": provider,
                "llm.cache_hit": cache_hit,
                "llm.replay_mode": replay_mode,
            },
        )

    async def execute_llm_real(
        self,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        enable_tool_calls: bool = False,
        tool_names: list[str] | None = None,
        tool_choice: Any | None = None,
        max_tool_calls: int = 6,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        """Execute LLM node with real API call or deterministic replay."""
        logger.info(
            "[%s] LLM call config (execution=%s): model=%s max_tokens=%s temperature=%s parallel_tool_calls=%s enable_reasoning=%s thinking_budget=%s prompt_len=%d prompt_preview=%r",
            node_id,
            execution.execution_id,
            model,
            max_tokens,
            temperature,
            parallel_tool_calls,
            enable_reasoning,
            thinking_budget,
            len(prompt or ""),
            (prompt or "")[:120],
        )

        cache_key: str | None = None

        decision = decide_replay(
            execution_metadata=execution.metadata,
            llm_call_logs=self.llm_call_logs,
            execution_id=execution.execution_id,
            node_id=node_id,
            llm_cache=self._llm_cache,
            model=model,
            prompt_cache_key=prompt_cache_key,
        )

        if decision.kind == ReplayDecisionKind.RECORDED_LLM_TEXT and decision.llm_text:
            logger.debug("Using recorded response for node %s (deterministic replay)", node_id)

            # Create LLM span for replay (cache_hit=True since we're using recorded response)
            llm_span = self._create_llm_span(
                model=model, cache_hit=True, replay_mode="deterministic"
            )

            # Emit start span so the frontend Timeline can show it
            if llm_span:
                await self._observation_service.emit(
                    SpanUpdateEvent.from_span(
                        llm_span,
                        session_id=session_id,
                        execution_id=execution.execution_id,
                    )
                )

            # Replay with streamed chunks to mimic original execution.
            words = decision.llm_text.split()
            for word in words:
                chunk = word + " "
                node_exec.streaming_output += chunk

                stream_event = StreamingOutputEvent(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    execution_id=execution.execution_id,
                    node_id=node_id,
                    chunk=chunk,
                    is_final=False,
                )
                await self._observation_service.emit(stream_event)

            final_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                chunk="",
                is_final=True,
            )
            await self._observation_service.emit(final_event)

            # Ensure deterministic replay produces the same output shape as real executions.
            # Downstream consumers (UI/tests) rely on node_exec.outputs being present.
            node_exec.outputs = {
                "result": node_exec.streaming_output,
                "metadata": {
                    "replay_mode": "deterministic",
                    "llm_cache_hit": True,
                },
            }

            # End LLM span and emit completion event
            if llm_span:
                llm_span.set_status("ok")
                llm_span.end()
                await self._observation_service.emit(
                    SpanUpdateEvent.from_span(
                        llm_span,
                        session_id=session_id,
                        execution_id=execution.execution_id,
                    )
                )

            logger.debug("Deterministic replay completed for node %s", node_id)
            return

        if decision.kind == ReplayDecisionKind.RECORDED_TOOL_OUTPUT and decision.tool_output:
            logger.debug("Using recorded tool-calling response for node %s", node_id)
            await self.replay_tool_call_log(
                session_id, execution, node_id, node_exec, decision.tool_output
            )
            return

        if enable_tool_calls or tool_names:
            handled = await self.execute_llm_tool_calls(
                session_id=session_id,
                execution=execution,
                node_id=node_id,
                node_exec=node_exec,
                prompt=prompt,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                tool_names=tool_names or [],
                tool_choice=tool_choice,
                max_tool_calls=max_tool_calls,
                max_tokens=max_tokens,
                temperature=temperature,
                parallel_tool_calls=parallel_tool_calls,
                prompt_cache_key=prompt_cache_key,
            )
            if handled:
                return

        if decision.kind == ReplayDecisionKind.CACHED_LLM_TEXT and decision.llm_text is not None:
            cache_key = decision.cache_key
            cached_response = decision.llm_text
            logger.info(
                "[%s] LLM cache hit: cache_key=%s",
                node_id,
                cache_key,
            )

            # Create LLM span for cache hit
            llm_span = self._create_llm_span(model=model, cache_hit=True)

            # Emit start span event so the frontend Timeline can show it
            if llm_span:
                await self._observation_service.emit(
                    SpanUpdateEvent.from_span(
                        llm_span,
                        session_id=session_id,
                        execution_id=execution.execution_id,
                    )
                )

            node_exec.streaming_output += cached_response
            node_exec.outputs = {
                "result": node_exec.streaming_output,
                "metadata": {
                    "llm_cache_hit": True,
                    "llm_cache_key": cache_key,
                    "prompt_cache_key": prompt_cache_key,
                },
            }

            stream_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                chunk=cached_response,
                is_final=False,
            )
            await self._observation_service.emit(stream_event)

            final_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                chunk="",
                is_final=True,
            )
            await self._observation_service.emit(final_event)

            # End LLM span and emit completion event
            if llm_span:
                llm_span.set_status("ok")
                llm_span.end()
                await self._observation_service.emit(
                    SpanUpdateEvent.from_span(
                        llm_span,
                        session_id=session_id,
                        execution_id=execution.execution_id,
                    )
                )

            return

        llm_adapter = self._adapter_factory()

        # Create LLM span for real API call
        llm_span = self._create_llm_span(model=model, cache_hit=False)

        if llm_span:
            await self._observation_service.emit(
                SpanUpdateEvent.from_span(
                    llm_span,
                    session_id=session_id,
                    execution_id=execution.execution_id,
                )
            )

        try:
            logger.debug("Creating LLM adapter and starting stream...")
            if enable_reasoning:
                logger.debug("Reasoning mode enabled with budget=%s", thinking_budget or "default")

            chunk_count = 0
            reasoning_count = 0
            reasoning_output = ""
            reasoning_started = False
            reasoning_start_time = None

            try:
                # Stream with timeout protection to avoid hanging connections.
                async with asyncio.timeout(60.0):
                    stream_kwargs: dict[str, Any] = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "enable_reasoning": enable_reasoning,
                        "thinking_budget": thinking_budget,
                    }
                    if temperature is not None:
                        stream_kwargs["temperature"] = temperature
                    if parallel_tool_calls is not None:
                        stream_kwargs["parallel_tool_calls"] = parallel_tool_calls
                    if prompt_cache_key:
                        stream_kwargs["prompt_cache_key"] = prompt_cache_key

                    async for content, reasoning in llm_adapter.stream_completion(
                        prompt,
                        **stream_kwargs,
                    ):
                        if reasoning and enable_reasoning:
                            reasoning_count += 1
                            reasoning_output += reasoning

                            if not reasoning_started:
                                reasoning_started = True
                                reasoning_start_time = datetime.now()
                                start_event = StreamingOutputEvent(
                                    event_id=f"evt_{uuid4().hex[:8]}",
                                    session_id=session_id,
                                    execution_id=execution.execution_id,
                                    node_id=node_id,
                                    chunk="\n[Reasoning Start]\n",
                                    is_final=False,
                                )
                                await self._observation_service.emit(start_event)
                                logger.debug("[%s] Reasoning started...", node_id)

                            reasoning_event = StreamingOutputEvent(
                                event_id=f"evt_{uuid4().hex[:8]}",
                                session_id=session_id,
                                execution_id=execution.execution_id,
                                node_id=node_id,
                                chunk=reasoning,
                                is_final=False,
                            )
                            await self._observation_service.emit(reasoning_event)

                            if reasoning_count % 50 == 0:
                                logger.debug("[%s] Reasoning: %d chunks", node_id, reasoning_count)
                        elif reasoning and not enable_reasoning:
                            # Discard reasoning if not enabled, but keep counts for logging.
                            reasoning_count += 1
                            reasoning_output += reasoning
                            if reasoning_count % 100 == 0:
                                logger.debug(
                                    "[%s] Discarding reasoning: %d chunks (reasoning not enabled)",
                                    node_id,
                                    reasoning_count,
                                )

                        if content:
                            if reasoning_started and reasoning_count > 0:
                                reasoning_duration = 0.0
                                if reasoning_start_time:
                                    duration_delta = datetime.now() - reasoning_start_time
                                    reasoning_duration = duration_delta.total_seconds()

                                end_message = (
                                    "\n[Reasoning End] (Duration: "
                                    f"{reasoning_duration:.1f}s)\n\n[Final Answer]\n"
                                )
                                end_event = StreamingOutputEvent(
                                    event_id=f"evt_{uuid4().hex[:8]}",
                                    session_id=session_id,
                                    execution_id=execution.execution_id,
                                    node_id=node_id,
                                    chunk=end_message,
                                    is_final=False,
                                )
                                await self._observation_service.emit(end_event)
                                logger.info(
                                    "[%s] Reasoning completed in %.1fs", node_id, reasoning_duration
                                )
                                reasoning_started = False

                            chunk_count += 1
                            node_exec.streaming_output += content

                            stream_event = StreamingOutputEvent(
                                event_id=f"evt_{uuid4().hex[:8]}",
                                session_id=session_id,
                                execution_id=execution.execution_id,
                                node_id=node_id,
                                chunk=content,
                                is_final=False,
                            )
                            await self._observation_service.emit(stream_event)

                        if chunk_count % 10 == 0 and content:
                            logger.debug("Sent %d content chunks so far", chunk_count)

            except asyncio.TimeoutError:
                logger.error("LLM API call timeout after 60 seconds")
                node_exec.error = "API call timeout"
                error_chunk = "\n[Error: API call timeout after 60 seconds]"
                node_exec.streaming_output += error_chunk
                timeout_event = StreamingOutputEvent(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    execution_id=execution.execution_id,
                    node_id=node_id,
                    chunk=error_chunk,
                    is_final=True,
                )
                await self._observation_service.emit(timeout_event)
                return

            if reasoning_output:
                logger.info(
                    "[%s] Reasoning completed: %d chunks, %d chars",
                    node_id,
                    reasoning_count,
                    len(reasoning_output),
                )
                logger.debug("[%s] Reasoning content: %s...", node_id, reasoning_output[:200])
                if not node_exec.metadata:
                    node_exec.metadata = {}
                node_exec.metadata["reasoning_content"] = reasoning_output

            final_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                chunk="",
                is_final=True,
            )
            await self._observation_service.emit(final_event)

            logger.info(
                "LLM node %s completed: %d content chunks, %d reasoning chunks",
                node_id,
                chunk_count,
                reasoning_count,
            )

            self.record_llm_call(
                execution_id=execution.execution_id,
                node_id=node_id,
                model=model,
                prompt=prompt,
                response=node_exec.streaming_output,
                metadata={
                    "chunk_count": chunk_count,
                    "reasoning_count": reasoning_count,
                    "reasoning_content": reasoning_output if reasoning_output else None,
                    "max_tokens": max_tokens,
                    "enable_reasoning": enable_reasoning,
                    "thinking_budget": thinking_budget,
                },
            )

            if not isinstance(node_exec.outputs, dict) or not node_exec.outputs:
                node_exec.outputs = {"result": node_exec.streaming_output}

            if prompt_cache_key:
                # Always emit outputs.metadata for cache observability when prompt_cache_key is set.
                # This keeps frontend behavior stable (it can always read llm_cache_hit/llm_cache_key)
                # and makes unit tests independent of NodeExecutionFlow post-processing.
                resolved_cache_key = cache_key or f"{model or ''}:{prompt_cache_key}"
                self._llm_cache[resolved_cache_key] = node_exec.streaming_output

                metadata_payload = node_exec.outputs.get("metadata")
                if not isinstance(metadata_payload, dict):
                    metadata_payload = {}
                metadata_payload.setdefault("llm_cache_hit", False)
                metadata_payload.setdefault("llm_cache_key", resolved_cache_key)
                metadata_payload.setdefault("prompt_cache_key", prompt_cache_key)
                node_exec.outputs["metadata"] = metadata_payload

            # Populate token usage from adapter if available
            adapter_usage = getattr(llm_adapter, "last_usage", None)
            if llm_span and isinstance(adapter_usage, dict):
                llm_span.tokens = TokenUsage(
                    input=adapter_usage.get("prompt_tokens", 0),
                    output=adapter_usage.get("completion_tokens", 0),
                    total=adapter_usage.get("total_tokens", 0),
                )

            # End LLM span on success
            if llm_span:
                llm_span.set_status("ok")
                llm_span.end()

                await self._observation_service.emit(
                    SpanUpdateEvent.from_span(
                        llm_span,
                        session_id=session_id,
                        execution_id=execution.execution_id,
                    )
                )

        except Exception as exc:
            logger.error("Error executing LLM node %s: %s", node_id, exc, exc_info=True)
            node_exec.error = str(exc)
            error_chunk = f"\n[Error: {str(exc)}]"
            node_exec.streaming_output += error_chunk

            if not isinstance(node_exec.outputs, dict) or not node_exec.outputs:
                node_exec.outputs = {"result": node_exec.streaming_output}

            if prompt_cache_key:
                resolved_cache_key = f"{model or ''}:{prompt_cache_key}"
                metadata_payload = node_exec.outputs.get("metadata")
                if not isinstance(metadata_payload, dict):
                    metadata_payload = {}
                metadata_payload.setdefault("llm_cache_hit", False)
                metadata_payload.setdefault("llm_cache_key", resolved_cache_key)
                metadata_payload.setdefault("prompt_cache_key", prompt_cache_key)
                node_exec.outputs["metadata"] = metadata_payload

            error_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                chunk=error_chunk,
                is_final=True,
            )
            await self._observation_service.emit(error_event)

            # End LLM span on error
            if llm_span:
                llm_span.set_status("error", str(exc))
                llm_span.end()

                await self._observation_service.emit(
                    SpanUpdateEvent.from_span(
                        llm_span,
                        session_id=session_id,
                        execution_id=execution.execution_id,
                    )
                )

    async def execute_llm_tool_calls(
        self,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
        prompt: str,
        system_prompt: str | None,
        user_prompt: str | None,
        model: str | None,
        tool_names: list[str],
        tool_choice: Any | None,
        max_tool_calls: int,
        max_tokens: int | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        prompt_cache_key: str | None = None,
    ) -> bool:
        if self._tool_call_service is None:
            logger.warning("Tool-call service is not configured; skipping tool calls.")
            return False
        return await self._tool_call_service.execute_tool_calls(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            tool_names=tool_names,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_key=prompt_cache_key,
        )

    async def execute_llm_mock(
        self,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
    ) -> None:
        """Execute LLM node with mock streaming (for testing UI)."""
        output_text = (
            f"This is the streaming output from {node_id}. "
            "The LLM is processing your request and generating a response. "
            "This demonstrates the streaming capability where tokens appear "
            "one by one in real-time, just like ChatGPT or Claude. "
            "You can see each word appearing gradually with a typing effect."
        )
        words = output_text.split()

        for i, word in enumerate(words):
            chunk = word + " "
            node_exec.streaming_output += chunk

            stream_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                chunk=chunk,
                is_final=(i == len(words) - 1),
            )
            await self._observation_service.emit(stream_event)

            await self._sleep_func(0.05)

    def record_llm_call(
        self,
        execution_id: str,
        node_id: str,
        model: str,
        prompt: str | list[dict[str, Any]],
        response: str,
        metadata: dict | None = None,
    ) -> None:
        """Record an LLM call for deterministic replay."""
        replay_record_llm_call(
            llm_call_logs=self.llm_call_logs,
            execution_id=execution_id,
            node_id=node_id,
            model=model,
            prompt=prompt,
            response=response,
            metadata=metadata,
            timestamp=datetime.now(),
        )

    def get_recorded_llm_response(self, execution_id: str, node_id: str) -> str | None:
        """Get recorded LLM response for deterministic replay."""
        return replay_get_recorded_llm_response(
            llm_call_logs=self.llm_call_logs,
            execution_id=execution_id,
            node_id=node_id,
        )

    def get_recorded_tool_call_response(
        self, execution_id: str, node_id: str
    ) -> dict[str, Any] | None:
        return replay_get_recorded_tool_call_output(
            llm_call_logs=self.llm_call_logs,
            execution_id=execution_id,
            node_id=node_id,
        )

    async def replay_tool_call_log(
        self,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
        recorded_output: dict[str, Any],
    ) -> None:
        output_payload = recorded_output
        if isinstance(recorded_output, LLMToolCallOutputIR):
            output_payload = recorded_output.model_dump(by_alias=True)

        node_exec.outputs = dict(output_payload)
        # Mark as cache-hit for deterministic replay so frontend Activity log detects it
        if isinstance(node_exec.outputs, dict):
            meta = node_exec.outputs.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["llm_cache_hit"] = True
            meta["replay_mode"] = "deterministic"
            node_exec.outputs["metadata"] = meta
        content = ""
        if isinstance(output_payload, dict):
            content = output_payload.get("content") or ""

        if content:
            node_exec.streaming_output += content
            stream_event = StreamingOutputEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                chunk=content,
                is_final=False,
            )
            await self._observation_service.emit(stream_event)

        final_event = StreamingOutputEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            execution_id=execution.execution_id,
            node_id=node_id,
            chunk="",
            is_final=True,
        )
        await self._observation_service.emit(final_event)
