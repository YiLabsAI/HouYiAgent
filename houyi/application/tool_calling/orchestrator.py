"""Tool-calling orchestrator primitives for per-round execution flow."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from houyi.application.tool_calling.budget import prepare_tool_loop_messages
from houyi.application.tool_calling.context import (
    ToolCallBatchExecutionContext,
    ToolLoopConfig,
    ToolLoopContext,
    ToolLoopRuntimeServices,
    ToolLoopState,
    ToolRoundPhaseContext,
    build_tool_call_batch_execution_context,
    build_tool_call_execution_context,
    build_tool_round_phase_context,
)
from houyi.application.tool_calling.tool_call_messages import build_assistant_tool_message

logger = logging.getLogger(__name__)

_TEXTUAL_TOOL_MARKER_RE = re.compile(
    r"\[tool(?::| call\])|<tool_call\b|<\|tool_",
    re.IGNORECASE,
)


class ToolLoopOrchestrator:
    """Application-layer helpers that drive per-round tool execution."""

    @staticmethod
    def _round_summary(response: Any, round_index: int) -> dict[str, Any]:
        response_usage = getattr(response, "usage", None)
        return {
            "round_index": round_index + 1,
            "usage": dict(response_usage) if isinstance(response_usage, dict) else None,
        }

    @staticmethod
    def _attach_round_summaries(response: Any, round_summaries: list[dict[str, Any]]) -> None:
        if hasattr(response, "metadata") and isinstance(response.metadata, dict):
            response.metadata["tool_loop_round_summaries"] = list(round_summaries)
            return
        if hasattr(response, "metadata"):
            response.metadata = {"tool_loop_round_summaries": list(round_summaries)}

    @staticmethod
    def _log_round_response(
        *,
        response: Any,
        round_index: int,
        chat_start: float,
        enable_timing: bool,
    ) -> None:
        response_content = str(getattr(response, "content", "") or "")
        response_metadata = getattr(response, "metadata", None)
        response_reasoning = (
            str(response_metadata.get("reasoning_content") or "")
            if isinstance(response_metadata, dict)
            else ""
        )
        tool_call_count = len(response.tool_calls or [])
        contains_textual_tool_markers = bool(
            _TEXTUAL_TOOL_MARKER_RE.search(response_content)
            or _TEXTUAL_TOOL_MARKER_RE.search(response_reasoning)
        )
        logger.debug(
            "[ToolCallRunner] round=%s response_shape tool_calls=%s content_len=%s reasoning_len=%s finish_reason=%s textual_tool_markers=%s",
            round_index + 1,
            tool_call_count,
            len(response_content),
            len(response_reasoning),
            getattr(response, "finish_reason", None),
            contains_textual_tool_markers,
        )
        if tool_call_count == 0 and contains_textual_tool_markers:
            logger.warning(
                "[ToolCallRunner] round=%s response contained textual tool markers without structured tool_calls",
                round_index + 1,
            )
        if enable_timing:
            chat_elapsed = time.perf_counter() - chat_start
            logger.debug(
                "[ToolCallRunner] round=%s chat=%.3fs tool_calls=%s",
                round_index + 1,
                chat_elapsed,
                tool_call_count,
            )

    @staticmethod
    def _complete_without_tools(
        *,
        response: Any,
        round_index: int,
        round_summaries: list[dict[str, Any]],
        enable_timing: bool,
        started_at_monotonic: float | None,
    ) -> Any:
        if enable_timing and started_at_monotonic is not None:
            logger.debug(
                "[ToolCallRunner] completed: rounds_used=%s total=%.3fs",
                round_index + 1,
                time.perf_counter() - started_at_monotonic,
            )
        ToolLoopOrchestrator._attach_round_summaries(response, round_summaries)
        return response

    @staticmethod
    async def execute_rounds(ctx: ToolLoopContext) -> Any:
        runner = ctx.runner
        config = ctx.config
        state = ctx.state
        services = ctx.services

        response: Any = None
        round_summaries: list[dict[str, Any]] = []
        for round_index in range(config.tool_loop_max_rounds):
            round_start = time.perf_counter() if config.tool_loop_enable_timing else 0.0
            chat_start = time.perf_counter() if config.tool_loop_enable_timing else 0.0
            chat_messages = prepare_tool_loop_messages(
                state.tool_loop_messages,
                config.tool_loop_max_message_chars,
                config.tool_loop_max_total_chars,
            )
            response, _llm_span = await runner._call_llm_with_cache(
                services.model_adapter,
                chat_messages,
                services.available_tool_schemas,
                services.model_request_options,
                services.llm_response_cache,
                round_index,
            )
            round_summaries.append(ToolLoopOrchestrator._round_summary(response, round_index))
            ToolLoopOrchestrator._log_round_response(
                response=response,
                round_index=round_index,
                chat_start=chat_start,
                enable_timing=config.tool_loop_enable_timing,
            )
            if not response.tool_calls:
                return ToolLoopOrchestrator._complete_without_tools(
                    response=response,
                    round_index=round_index,
                    round_summaries=round_summaries,
                    enable_timing=config.tool_loop_enable_timing,
                    started_at_monotonic=state.tool_loop_started_at_monotonic,
                )

            assistant_tool_message = build_assistant_tool_message(response)
            state.tool_loop_messages.append(assistant_tool_message)

            should_stop = await ToolLoopOrchestrator.execute_round_tool_phase(
                build_tool_round_phase_context(
                    loop_ctx=ctx,
                    response=response,
                    round_index=round_index,
                    round_start=round_start,
                )
            )
            if should_stop:
                ToolLoopOrchestrator._attach_round_summaries(response, round_summaries)
                return response
        if response is not None:
            ToolLoopOrchestrator._attach_round_summaries(response, round_summaries)
        return response

    @staticmethod
    async def execute_round_tool_phase(ctx: ToolRoundPhaseContext) -> bool:
        runner = ctx.runner
        config = ctx.config
        state = ctx.state
        services = ctx.services

        tool_messages: list[dict[str, Any]] = []
        tool_durations: list[float] = []
        tool_phase_start = time.perf_counter() if config.tool_loop_enable_timing else 0.0

        parsed_tool_calls, has_placeholders = runner._parse_tool_calls_for_fast_path(
            state.response.tool_calls,
            config.tool_loop_enable_fast_path,
        )
        allow_parallel = config.tool_loop_enable_parallel_calls and not (
            config.tool_loop_enable_fast_path and has_placeholders
        )
        is_parallel_batch = allow_parallel and len(parsed_tool_calls) > 1
        round_parallel_group_id = f"round_{config.round_index + 1}" if is_parallel_batch else None
        resolved_outputs = (
            None
            if is_parallel_batch
            else (
                state.tool_loop_resolved_outputs_by_tool
                if config.tool_loop_enable_fast_path
                else None
            )
        )

        if is_parallel_batch:
            results = await ToolLoopOrchestrator.execute_parallel_tool_calls(
                build_tool_call_batch_execution_context(
                    phase_ctx=ctx,
                    parsed_tool_calls=parsed_tool_calls,
                    resolved_outputs=None,
                    parallel_group_id=round_parallel_group_id,
                )
            )
            ordered_results = sorted(results, key=lambda item: item[0])
        else:
            ordered_results = await ToolLoopOrchestrator.execute_serial_tool_calls(
                build_tool_call_batch_execution_context(
                    phase_ctx=ctx,
                    parsed_tool_calls=parsed_tool_calls,
                    resolved_outputs=resolved_outputs,
                    parallel_group_id=round_parallel_group_id,
                )
            )

        for _, trace_entry, tool_message, tool_elapsed in ordered_results:
            state.tool_loop_trace_entries.append(trace_entry)
            tool_messages.append(tool_message)
            if config.tool_loop_enable_timing:
                tool_durations.append(tool_elapsed)

        state.tool_loop_messages.extend(tool_messages)
        if runner._should_exit_fast_path(
            config.tool_loop_enable_fast_path,
            has_placeholders,
            services.available_tool_names,
            state.tool_loop_invoked_tool_names,
        ):
            if config.tool_loop_enable_timing:
                logger.debug(
                    "[ToolCallRunner] fast_path=early_exit round=%s", config.round_index + 1
                )
            return True

        if config.tool_loop_enable_fast_path and config.tool_loop_enable_timing:
            logger.debug(
                "[ToolCallRunner] fast_path=continue round=%s tool_calls=%s max_rounds=%s",
                config.round_index + 1,
                len(parsed_tool_calls),
                config.tool_loop_max_rounds,
            )
        if config.tool_loop_enable_timing:
            tool_phase_elapsed = time.perf_counter() - tool_phase_start
            logger.debug(
                "[ToolCallRunner] round=%s tools=%.3fs sum=%.3fs max=%.3fs parallel=%s",
                config.round_index + 1,
                tool_phase_elapsed,
                sum(tool_durations),
                max(tool_durations, default=0.0),
                allow_parallel and len(parsed_tool_calls) > 1,
            )
            logger.debug(
                "[ToolCallRunner] round=%s total=%.3fs",
                config.round_index + 1,
                time.perf_counter() - state.tool_round_started_at_monotonic,
            )
        return False

    @staticmethod
    async def execute_parallel_tool_calls(
        batch_ctx: ToolCallBatchExecutionContext,
    ) -> list[tuple[int, dict[str, Any], dict[str, Any], float]]:
        runner = batch_ctx.runner
        config = batch_ctx.config
        state = batch_ctx.state

        semaphore = asyncio.Semaphore(config.tool_loop_max_parallel_calls)

        async def _run_one(
            tool_call: Any,
            index: int,
            parsed_args: dict[str, Any] | None,
        ) -> tuple[int, dict[str, Any], dict[str, Any], float]:
            async with semaphore:
                return await runner._handle_tool_call_impl(
                    build_tool_call_execution_context(
                        batch_ctx=batch_ctx,
                        index=index,
                        tool_call=tool_call,
                        parsed_args=parsed_args,
                        resolved_outputs=None,
                    )
                )

        return await asyncio.gather(
            *[
                _run_one(tool_call, index, parsed_args)
                for index, (tool_call, parsed_args) in enumerate(state.parsed_tool_calls)
            ]
        )

    @staticmethod
    async def execute_serial_tool_calls(
        batch_ctx: ToolCallBatchExecutionContext,
    ) -> list[tuple[int, dict[str, Any], dict[str, Any], float]]:
        runner = batch_ctx.runner
        state = batch_ctx.state

        results: list[tuple[int, dict[str, Any], dict[str, Any], float]] = []
        for index, (tool_call, parsed_args) in enumerate(state.parsed_tool_calls):
            result = await runner._handle_tool_call_impl(
                build_tool_call_execution_context(
                    batch_ctx=batch_ctx,
                    index=index,
                    tool_call=tool_call,
                    parsed_args=parsed_args,
                    resolved_outputs=state.resolved_outputs,
                )
            )
            results.append(result)
        return results


__all__ = [
    "ToolLoopConfig",
    "ToolLoopContext",
    "ToolLoopOrchestrator",
    "ToolLoopRuntimeServices",
    "ToolLoopState",
]
