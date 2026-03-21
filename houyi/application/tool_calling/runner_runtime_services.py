"""Remaining host for the tool-loop session builder after runtime collaborator extraction."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from houyi.application.tool_calling.budget import resolve_tool_loop_budget_chars
from houyi.application.tool_calling.context import (
    ToolLoopConfig,
    ToolLoopContext,
    ToolLoopRuntimeServices,
    ToolLoopState,
)
from houyi.application.tool_calling.runner_models import (
    _DEFAULT_TOOL_RESULT_SUMMARY_ENABLED,
    _DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS,
    _DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS,
    _parse_max_parallel_calls,
    _parse_tool_latency_seconds,
    _read_bool_env,
    _read_positive_int_env_or_none,
)
from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


class _ToolLoopSessionBuilder:
    """Build immutable/mutable loop runtime context for one runner invocation."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def build(
        self,
        *,
        adapter: Any,
        messages: list[Any],
        tools: list[dict[str, Any]],
        skills: list[SkillSpec],
        executor: Any,
        max_rounds: int,
        chat_kwargs: dict[str, Any],
        tool_hooks: list[Any],
        allow_tool_replace: bool,
        tool_cache: dict[str, dict[str, Any]] | None,
        llm_cache: dict[str, Any] | None,
    ) -> tuple[ToolLoopContext, list[dict[str, Any]], Any]:
        """Assemble per-run loop config/state/services and initialize execution span."""
        from houyi.infrastructure.config import (
            ENV_TOOLCALL_FAST_PATH,
            ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS,
            ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS,
            ENV_TOOLCALL_RESULT_SUMMARY_ENABLED,
            ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS,
            ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS,
            ENV_TOOLCALL_TIMING,
        )

        timing_enabled = os.getenv(ENV_TOOLCALL_TIMING) == "1"
        skills_by_name = {skill.name: skill for skill in skills}
        all_tool_names = {name for name in skills_by_name if name}
        called_tools: set[str] = set()
        tool_trace: list[dict[str, Any]] = []
        should_parallel = (
            bool(chat_kwargs.get("parallel_tool_calls"))
            and not tool_hooks
            and not allow_tool_replace
        )
        max_parallel_calls = _parse_max_parallel_calls(chat_kwargs)
        fast_path_flag = (os.getenv(ENV_TOOLCALL_FAST_PATH) or "").strip().lower()
        fast_path_enabled = fast_path_flag in {"1", "true", "yes", "on"}
        requested_model_name = str(chat_kwargs.get("model") or "")
        tool_loop_max_message_chars, tool_loop_max_total_chars = resolve_tool_loop_budget_chars(
            adapter,
            _read_positive_int_env_or_none(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS),
            _read_positive_int_env_or_none(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS),
            requested_model_name,
        )
        tool_result_summary_enabled = _read_bool_env(
            ENV_TOOLCALL_RESULT_SUMMARY_ENABLED,
            _DEFAULT_TOOL_RESULT_SUMMARY_ENABLED,
        )
        tool_result_summary_max_chars = (
            _read_positive_int_env_or_none(ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS)
            or _DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS
        )
        tool_result_summary_max_items = (
            _read_positive_int_env_or_none(ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS)
            or _DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS
        )
        tool_outputs: dict[str, Any] = {}
        tool_latency_seconds = _parse_tool_latency_seconds()

        loop_start: float | None = None
        if timing_enabled:
            loop_start = time.perf_counter()
            logger.debug(
                "[ToolCallRunner] start: rounds=%s tools=%s messages=%s",
                max_rounds,
                len(tools),
                len(messages),
            )
            if fast_path_enabled:
                logger.debug("[ToolCallRunner] fast_path=enabled")

        exec_span = self._runner._execution_service._start_execution_span(max_rounds, len(tools))
        tool_loop_config = ToolLoopConfig(
            tool_loop_max_rounds=max_rounds,
            tool_loop_enable_parallel_calls=should_parallel,
            tool_loop_max_parallel_calls=max_parallel_calls,
            tool_loop_enable_fast_path=fast_path_enabled,
            tool_loop_enable_timing=timing_enabled,
            tool_loop_result_summary_enabled=tool_result_summary_enabled,
            tool_loop_result_summary_max_chars=tool_result_summary_max_chars,
            tool_loop_result_summary_max_items=tool_result_summary_max_items,
            tool_loop_max_message_chars=tool_loop_max_message_chars,
            tool_loop_max_total_chars=tool_loop_max_total_chars,
            tool_loop_injected_tool_latency_seconds=tool_latency_seconds,
        )
        tool_loop_state = ToolLoopState(
            tool_loop_messages=messages,
            tool_loop_trace_entries=tool_trace,
            tool_loop_invoked_tool_names=called_tools,
            tool_loop_resolved_outputs_by_tool=tool_outputs,
            tool_loop_started_at_monotonic=loop_start,
        )
        tool_loop_services = ToolLoopRuntimeServices(
            model_adapter=adapter,
            tool_executor=executor,
            available_tool_schemas=tools,
            skill_specs_by_name=skills_by_name,
            available_tool_names=all_tool_names,
            model_request_options=chat_kwargs,
            llm_response_cache=llm_cache,
            tool_result_cache=tool_cache,
            tool_call_hooks=tool_hooks,
            allow_tool_replacement=allow_tool_replace,
        )
        return (
            ToolLoopContext(
                runner=self._runner,
                config=tool_loop_config,
                state=tool_loop_state,
                services=tool_loop_services,
            ),
            tool_trace,
            exec_span,
        )
