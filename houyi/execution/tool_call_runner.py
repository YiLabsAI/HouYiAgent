"""Shared tool-calling loop runner."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from houyi.core.skill import SkillSpec
from houyi.execution.arg_coercion import coerce_args
from houyi.execution.placeholder_resolver import PlaceholderResolver
from houyi.execution.skill_executor import SkillExecutionError
from houyi.execution.tool_result import ToolResultBuilder
from houyi.llm.base import LLMAdapter
from houyi.llm.models import (
    CHARS_PER_TOKEN_BLENDED,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_OUTPUT_RESERVE,
    MODEL_CONTEXT_WINDOWS,
)

if TYPE_CHECKING:
    from houyi.core.skill.consent import ConsentManager
    from houyi.core.skill.hooks import SkillHooksManager
    from houyi.core.skill.metrics import MetricsStore
    from houyi.core.skill.policy import PolicyEnforcer

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_LOOP_MAX_MESSAGE_CHARS = 12_000
_MIN_TOOL_LOOP_MAX_MESSAGE_CHARS = 1_000
_MIN_TOOL_LOOP_MAX_TOTAL_CHARS = 8_000
_AUTO_TOOL_LOOP_INPUT_BUDGET_RATIO = 0.7
_AUTO_TOOL_LOOP_MESSAGE_RATIO = 0.1
_DEFAULT_TOOL_RESULT_SUMMARY_ENABLED = True
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS = 4_000
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS = 50


def _read_positive_int_env_or_none(env_name: str) -> int | None:
    raw = os.getenv(env_name)
    if raw is None:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, ignore and use auto/default budget", env_name, raw)
        return None
    if parsed <= 0:
        logger.warning(
            "Invalid %s=%r (must be > 0), ignore and use auto/default budget", env_name, raw
        )
        return None
    return parsed


def _read_bool_env(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r, fallback to %s", env_name, raw, default)
    return default


def _resolve_tool_loop_budget_chars(
    adapter: Any,
    message_chars_override: int | None,
    total_chars_override: int | None,
    model_name_override: str | None = None,
) -> tuple[int, int]:
    model_name = str(model_name_override or getattr(adapter, "model", "") or "")
    context_window = MODEL_CONTEXT_WINDOWS.get(model_name)
    if context_window is None and model_name:
        model_name_lower = model_name.lower()
        for known_model, known_window in MODEL_CONTEXT_WINDOWS.items():
            known_lower = known_model.lower()
            if known_lower in model_name_lower or model_name_lower in known_lower:
                context_window = known_window
                break
    if context_window is None:
        context_window = DEFAULT_CONTEXT_WINDOW
    input_budget_tokens = max(
        1,
        int(max(0, context_window - DEFAULT_OUTPUT_RESERVE) * _AUTO_TOOL_LOOP_INPUT_BUDGET_RATIO),
    )
    auto_total_chars = max(
        _MIN_TOOL_LOOP_MAX_TOTAL_CHARS,
        int(input_budget_tokens * CHARS_PER_TOKEN_BLENDED),
    )

    total_chars = total_chars_override or auto_total_chars
    if message_chars_override is not None:
        message_chars = message_chars_override
        if total_chars_override is None:
            total_chars = max(total_chars, message_chars * 4)
    else:
        message_chars = max(
            _MIN_TOOL_LOOP_MAX_MESSAGE_CHARS,
            int(total_chars * _AUTO_TOOL_LOOP_MESSAGE_RATIO),
        )
        message_chars = min(message_chars, _DEFAULT_TOOL_LOOP_MAX_MESSAGE_CHARS)

    message_chars = min(message_chars, total_chars)
    return message_chars, total_chars


def _summarize_json_like(
    value: Any, *, max_items: int, max_string_chars: int, depth: int = 0
) -> Any:
    if depth >= 4:
        return "...[truncated-depth]..."
    if isinstance(value, dict):
        items = list(value.items())
        trimmed_items = items[:max_items]
        summarized = {
            str(k): _summarize_json_like(
                v,
                max_items=max_items,
                max_string_chars=max_string_chars,
                depth=depth + 1,
            )
            for k, v in trimmed_items
        }
        if len(items) > max_items:
            summarized["__truncated_keys__"] = len(items) - max_items
        return summarized
    if isinstance(value, list):
        trimmed = [
            _summarize_json_like(
                item,
                max_items=max_items,
                max_string_chars=max_string_chars,
                depth=depth + 1,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            trimmed.append(f"...[{len(value) - max_items} items truncated]...")
        return trimmed
    if isinstance(value, str) and len(value) > max_string_chars:
        return _truncate_middle(value, max_string_chars)
    return value


def _summarize_tool_result_content(
    content: str,
    *,
    max_chars: int,
    max_items: int,
) -> tuple[str, bool]:
    if max_chars <= 0:
        return content, False
    if len(content) <= max_chars:
        return content, False

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return _truncate_middle(content, max_chars), True

    summarized_payload = _summarize_json_like(
        payload,
        max_items=max_items,
        max_string_chars=max(200, max_chars // 4),
    )
    summarized = json.dumps(summarized_payload, ensure_ascii=False, sort_keys=True)
    if len(summarized) > max_chars:
        summarized = _truncate_middle(summarized, max_chars)
    return summarized, summarized != content


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def _message_payload_chars(message: dict[str, Any]) -> int:
    content_len = len(str(message.get("content") or ""))
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return content_len

    args_len = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        args = fn.get("arguments")
        args_len += len(args) if isinstance(args, str) else len(str(args or ""))
    return content_len + args_len


def _truncate_message_for_budget(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    normalized = dict(message)
    normalized["content"] = _truncate_middle(
        LLMAdapter._coerce_message_content_to_text(normalized.get("content")),
        max_chars,
    )
    tool_calls = normalized.get("tool_calls")
    if not isinstance(tool_calls, list):
        return normalized

    fixed_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        fixed = LLMAdapter._sanitize_tool_call(call)
        if fixed is None:
            continue
        fn = fixed.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = _truncate_middle(args, max_chars)
        fixed_calls.append(fixed)
    normalized["tool_calls"] = fixed_calls
    return normalized


def _cap_total_payload(
    messages: list[dict[str, Any]], max_total_chars: int
) -> list[dict[str, Any]]:
    total_chars = sum(_message_payload_chars(msg) for msg in messages)
    if total_chars <= max_total_chars:
        return messages

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    system_chars = sum(_message_payload_chars(msg) for msg in system_messages)
    budget_for_non_system = max(0, max_total_chars - system_chars)

    kept_non_system: list[dict[str, Any]] = []
    used = 0
    for msg in reversed(non_system):
        payload_chars = _message_payload_chars(msg)
        if kept_non_system and used + payload_chars > budget_for_non_system:
            continue
        kept_non_system.append(msg)
        used += payload_chars

    trimmed = system_messages + list(reversed(kept_non_system))
    logger.warning(
        "ToolCallRunner message budget applied: total_payload=%d -> %d, messages=%d -> %d",
        total_chars,
        sum(_message_payload_chars(msg) for msg in trimmed),
        len(messages),
        len(trimmed),
    )
    return trimmed


class _HookCtx(TypedDict):
    tool_name: str | None
    args: dict[str, Any]
    skill: SkillSpec | None
    tool_call_id: str | None


class ToolCallRunner:
    """Run tool-calling loops with hooks, policy enforcement, and trace events."""

    def __init__(
        self,
        trace_manager: Any | None = None,
        skill_hooks_manager: SkillHooksManager | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        consent_manager: ConsentManager | None = None,
        metrics_store: MetricsStore | None = None,
    ) -> None:
        self.trace_manager = trace_manager
        self.skill_hooks_manager = skill_hooks_manager
        self.policy_enforcer = policy_enforcer
        self.consent_manager = consent_manager
        self.metrics_store = metrics_store
        self._consent_cache: dict[str, bool] = {}  # skill_name -> consent_granted
        self._metrics_collectors: dict[str, Any] = {}  # skill_name -> MetricsCollector

    async def _trigger_stop_hook(self, tool_trace: list[dict[str, Any]]) -> None:
        """Trigger the Stop hook at the end of a tool-calling session."""
        if not self.skill_hooks_manager:
            return
        from houyi.core.skill.hooks import HookContext, HookEvent

        stop_ctx = HookContext(
            tool_name="__session__",
            tool_args={"tool_trace_length": len(tool_trace)},
        )
        try:
            await self.skill_hooks_manager.trigger_hook(HookEvent.STOP, stop_ctx)
        except Exception:
            logger.debug("Stop hook error (non-fatal)", exc_info=True)

    def get_skill_metrics(self, skill_name: str) -> Any:
        """Get aggregated metrics for a skill.

        Returns None if no metrics have been collected or no metrics_store is configured.
        """
        if not self.metrics_store:
            return None
        return self.metrics_store.aggregate(skill_name)

    def get_all_skill_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics for all skills.

        Returns empty dict if no metrics_store is configured.
        """
        if not self.metrics_store:
            return {}
        return {
            skill_name: self.metrics_store.aggregate(skill_name)
            for skill_name in self.metrics_store.list_skills()
        }

    def export_metrics_to_trace(self) -> None:
        """Export aggregated metrics to current trace span as OpenTelemetry attributes.

        This should be called at the end of an execution to attach metrics summary
        to the trace for observability.
        """
        if not self.trace_manager or not self.metrics_store:
            return
        span = getattr(self.trace_manager, "current_span", None)
        if span is None:
            return

        from houyi.core.skill.metrics import MetricsExporter

        for skill_name in self.metrics_store.list_skills():
            metrics = self.metrics_store.aggregate(skill_name)
            if metrics:
                attrs = MetricsExporter.to_opentelemetry_attributes(metrics)
                for key, value in attrs.items():
                    with contextlib.suppress(Exception):
                        span.set_attribute(f"{skill_name}.{key}", value)

    async def run(
        self,
        adapter: Any,
        messages: list[Any],
        tools: list[dict[str, Any]],
        skills: list[SkillSpec],
        executor: Any,
        max_rounds: int,
        chat_kwargs: dict[str, Any] | None = None,
        tool_hooks: list[Any] | None = None,
        allow_tool_replace: bool = False,
        tool_cache: dict[str, dict[str, Any]] | None = None,
        llm_cache: dict[str, Any] | None = None,
        preprocessors: list[Any] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Run tool-calling loop until no tool calls are returned.

        Args:
            preprocessors: Optional list of PreprocessorSpec objects.  When
                provided, a :class:`PreprocessorPipeline` executes them
                **before** the first LLM call and injects their outputs into
                the message context.
        """
        from houyi.core.skill.hooks import HookContext, HookEvent

        # --- Preprocessors: deterministic pre-LLM execution (M8) ---
        if preprocessors:
            from houyi.core.skill.preprocessor import PreprocessorPipeline

            pipeline = PreprocessorPipeline(preprocessors)
            try:
                pp_results = await pipeline.run()
                messages = pipeline.inject(messages, pp_results)
                logger.debug(
                    "Preprocessors executed: %d total, %d successful",
                    len(pp_results),
                    sum(1 for r in pp_results if r.success),
                )
            except Exception:
                logger.warning("Preprocessor pipeline error (non-fatal)", exc_info=True)

        # --- SessionStart hook ---
        if self.skill_hooks_manager:
            session_ctx = HookContext(
                tool_name="__session__",
                tool_args={
                    "max_rounds": max_rounds,
                    "tool_count": len(tools),
                    "skill_count": len(skills),
                },
            )
            try:
                await self.skill_hooks_manager.trigger_hook(HookEvent.SESSION_START, session_ctx)
            except Exception:
                logger.debug("SessionStart hook error (non-fatal)", exc_info=True)

        # --- Tool Router: allowed-tools whitelist enforcement (M9) ---
        from houyi.core.skill.tool_router import ToolRouter

        tool_router = ToolRouter(skills, self.policy_enforcer)
        if tool_router.has_restrictions:
            original_count = len(tools)
            tools = tool_router.filter_tools(tools)
            logger.debug(
                "ToolRouter: filtered %d → %d tools",
                original_count,
                len(tools),
            )

        from houyi.config.env_config import (
            ENV_TOOLCALL_FAST_PATH,
            ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS,
            ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS,
            ENV_TOOLCALL_RESULT_SUMMARY_ENABLED,
            ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS,
            ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS,
            ENV_TOOLCALL_TIMING,
            ENV_TOOLCALL_TOOL_LATENCY_MS,
        )

        timing_enabled = os.getenv(ENV_TOOLCALL_TIMING) == "1"
        skills_by_name = {skill.name: skill for skill in skills}
        all_tool_names = {name for name in skills_by_name if name}
        called_tools: set[str] = set()
        tool_trace: list[dict[str, Any]] = []
        tool_hooks = tool_hooks or []
        chat_kwargs = chat_kwargs or {}
        response: Any | None = None
        should_parallel = (
            bool(chat_kwargs.get("parallel_tool_calls"))
            and not tool_hooks
            and not allow_tool_replace
        )
        max_parallel_calls = 5
        if "max_parallel_calls" in chat_kwargs:
            raw_max_parallel_calls = chat_kwargs.get("max_parallel_calls")
            try:
                if raw_max_parallel_calls is None:
                    raise ValueError("max_parallel_calls is None")
                parsed_max_parallel_calls = int(raw_max_parallel_calls)
                if parsed_max_parallel_calls > 0:
                    max_parallel_calls = parsed_max_parallel_calls
                else:
                    logger.warning(
                        "Invalid max_parallel_calls=%s (must be > 0), using default=%s",
                        raw_max_parallel_calls,
                        max_parallel_calls,
                    )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid max_parallel_calls=%s (must be int), using default=%s",
                    raw_max_parallel_calls,
                    max_parallel_calls,
                )
        fast_path_flag = (os.getenv(ENV_TOOLCALL_FAST_PATH) or "").strip().lower()
        fast_path_enabled = fast_path_flag in {"1", "true", "yes", "on"}
        requested_model_name = str(chat_kwargs.get("model") or "")
        tool_loop_max_message_chars, tool_loop_max_total_chars = _resolve_tool_loop_budget_chars(
            adapter,
            _read_positive_int_env_or_none(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS),
            _read_positive_int_env_or_none(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS),
            requested_model_name or None,
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
        tool_latency_seconds: float | None = None
        tool_latency_env = os.getenv(ENV_TOOLCALL_TOOL_LATENCY_MS)
        if tool_latency_env:
            try:
                tool_latency_ms = float(tool_latency_env)
                if tool_latency_ms > 0:
                    tool_latency_seconds = tool_latency_ms / 1000.0
            except ValueError:
                logger.warning(
                    "Invalid HOUYI_TOOLCALL_TOOL_LATENCY_MS=%s",
                    tool_latency_env,
                )

        if timing_enabled:
            loop_start = time.perf_counter()
            logger.info(
                "[ToolCallRunner] start: rounds=%s tools=%s messages=%s",
                max_rounds,
                len(tools),
                len(messages),
            )
            if fast_path_enabled:
                logger.info("[ToolCallRunner] fast_path=enabled")

        # --- OTel: root execution span ---
        _exec_span = self._start_execution_span(max_rounds, len(tools))

        for round_index in range(max_rounds):
            round_start = time.perf_counter() if timing_enabled else 0.0
            chat_start = time.perf_counter() if timing_enabled else 0.0
            normalized_messages = LLMAdapter._sanitize_messages(
                [m for m in messages if isinstance(m, dict)]
            )
            normalized_messages = [
                _truncate_message_for_budget(msg, tool_loop_max_message_chars)
                for msg in normalized_messages
            ]
            chat_messages = _cap_total_payload(normalized_messages, tool_loop_max_total_chars)
            cache_key = self._build_llm_cache_key(
                adapter=adapter,
                messages=chat_messages,
                tools=tools,
                chat_kwargs=chat_kwargs,
            )
            cached_response = None
            if llm_cache is not None and cache_key:
                cached_response = llm_cache.get(cache_key)

            # --- OTel: LLM call span ---
            requested_model = None
            if isinstance(chat_kwargs, dict):
                requested_model = chat_kwargs.get("model")
            _llm_span = self._start_llm_span(adapter, round_index, requested_model=requested_model)
            if cached_response is not None:
                response = self._clone_llm_response(cached_response)
                logger.debug(
                    "[ToolCallRunner] llm_cache_hit round=%s key=%s",
                    round_index + 1,
                    cache_key,
                )
                self._emit_tool_event(
                    "ToolCallLLMCacheHit",
                    {
                        "round": round_index + 1,
                        "cache_key": cache_key,
                    },
                )
                if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                    response.metadata["llm_cache_hit"] = True
                    response.metadata["llm_cache_key"] = cache_key
                if _llm_span is not None:
                    _llm_span.set_attribute("llm.cache_hit", True)
                    _llm_span.cache_hit = True
            else:
                response = await adapter.chat(chat_messages, tools=tools, **chat_kwargs)
                if llm_cache is not None and cache_key:
                    llm_cache[cache_key] = self._clone_llm_response(response)
            if _llm_span is not None:
                usage = getattr(response, "usage", None)
                if isinstance(usage, dict) and usage:
                    _llm_span.set_tokens(
                        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        output_tokens=int(usage.get("completion_tokens", 0) or 0),
                    )
            self._end_span(_llm_span)
            if timing_enabled:
                chat_elapsed = time.perf_counter() - chat_start
                logger.info(
                    "[ToolCallRunner] round=%s chat=%.3fs tool_calls=%s",
                    round_index + 1,
                    chat_elapsed,
                    len(response.tool_calls or []),
                )
            if not response.tool_calls:
                if timing_enabled:
                    logger.info(
                        "[ToolCallRunner] completed: rounds_used=%s total=%.3fs",
                        round_index + 1,
                        time.perf_counter() - loop_start,
                    )
                await self._trigger_stop_hook(tool_trace)
                self._finish_execution_span(_exec_span)
                return response, tool_trace

            assistant_tool_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls,
            }
            response_metadata = getattr(response, "metadata", None)
            if isinstance(response_metadata, dict):
                reasoning_content = response_metadata.get("reasoning_content")
                if isinstance(reasoning_content, str):
                    assistant_tool_message["reasoning_content"] = reasoning_content
            messages.append(assistant_tool_message)

            tool_messages: list[dict[str, Any]] = []
            tool_durations: list[float] = []
            tool_phase_start = time.perf_counter() if timing_enabled else 0.0

            async def _handle_tool_call(
                tool_call: Any,
                index: int,
                parsed_args: dict[str, Any] | None = None,
                resolved_outputs: dict[str, Any] | None = None,
                parallel_group_id: str | None = None,
                round_index_value: int | None = None,
            ) -> tuple[int, dict[str, Any], dict[str, Any], float]:
                tool_start = time.perf_counter() if timing_enabled else 0.0
                if tool_latency_seconds:
                    await asyncio.sleep(tool_latency_seconds)
                tool_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                tool_name = tool_payload.get("name")
                tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                args = (
                    parsed_args
                    if parsed_args is not None
                    else ToolResultBuilder.parse_arguments(tool_payload.get("arguments"))
                )
                if resolved_outputs is not None:
                    args = PlaceholderResolver.resolve(args, resolved_outputs)
                    if tool_name:
                        args = coerce_args(tool_name, args, resolved_outputs)
                requested_tool_name = tool_name
                attempted_tool_name: str | None = None
                skill = skills_by_name.get(tool_name) if tool_name else None

                # Policy enforcement check (SimpleSkill v0.1 §5.2)
                if self.policy_enforcer and tool_name:
                    consent_given = self._consent_cache.get(tool_name, False)
                    decision = self.policy_enforcer.check_invocation(
                        skill_name=tool_name,
                        is_model_initiated=True,  # Tool calls from LLM are model-initiated
                        user_consent_given=consent_given,
                    )

                    if decision.requires_consent and self.consent_manager:
                        # Request consent asynchronously
                        from houyi.core.skill.consent import ConsentRequest, ConsentType

                        policy = self.policy_enforcer.get_policy(tool_name)
                        consent_request = ConsentRequest(
                            consent_type=ConsentType.INVOKE_CONFIRM,
                            skill_name=tool_name,
                            operation=f"invoke tool '{tool_name}'",
                            policy=policy,
                            context={"args": args, "tool_call_id": tool_call_id},
                        )
                        consent_response = await self.consent_manager.request_consent(
                            consent_request
                        )
                        if consent_response.is_granted():
                            self._consent_cache[tool_name] = True
                            decision = self.policy_enforcer.check_invocation(
                                skill_name=tool_name,
                                is_model_initiated=True,
                                user_consent_given=True,
                            )
                        else:
                            # Consent denied - return error result
                            self._emit_tool_event(
                                "ToolUsageBlocked",
                                {
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_name,
                                    "parallel_group_id": parallel_group_id,
                                    "reason": "consent_denied",
                                },
                            )
                            error_result = ToolResultBuilder.build(
                                {
                                    "error": "consent_denied",
                                    "message": f"User denied consent for tool '{tool_name}'",
                                },
                                call_id=tool_call_id,
                                metadata={"tool_name": tool_name, "policy_blocked": True},
                            )
                            trace_entry = {
                                "tool_name": tool_name,
                                "requested_tool_name": requested_tool_name,
                                "tool_call_id": tool_call_id,
                                "round_index": round_index_value,
                                "parallel_group_id": parallel_group_id,
                                "args": args,
                                "result": error_result,
                                "policy_blocked": True,
                                "block_reason": "consent_denied",
                            }
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": ToolResultBuilder.format(error_result),
                            }
                            return index, trace_entry, tool_message, 0.0

                    if not decision.allowed:
                        # Policy denied - return error result
                        self._emit_tool_event(
                            "ToolUsageBlocked",
                            {
                                "tool_call_id": tool_call_id,
                                "tool_name": tool_name,
                                "parallel_group_id": parallel_group_id,
                                "reason": decision.reason or "policy_denied",
                            },
                        )
                        error_result = ToolResultBuilder.build(
                            {
                                "error": "policy_denied",
                                "message": decision.reason
                                or f"Policy denied invocation of tool '{tool_name}'",
                            },
                            call_id=tool_call_id,
                            metadata={"tool_name": tool_name, "policy_blocked": True},
                        )
                        trace_entry = {
                            "tool_name": tool_name,
                            "requested_tool_name": requested_tool_name,
                            "tool_call_id": tool_call_id,
                            "round_index": round_index_value,
                            "parallel_group_id": parallel_group_id,
                            "args": args,
                            "result": error_result,
                            "policy_blocked": True,
                            "block_reason": decision.reason,
                        }
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": ToolResultBuilder.format(error_result),
                        }
                        return index, trace_entry, tool_message, 0.0

                hook_context: _HookCtx = {
                    "tool_name": tool_name,
                    "args": args,
                    "skill": skill,
                    "tool_call_id": tool_call_id,
                }

                # Trigger SimpleSkill PreToolUse hooks
                skill_hook_output: str | None = None
                if self.skill_hooks_manager and tool_name:
                    from houyi.core.skill.hooks import HookContext, HookEvent

                    skill_hook_ctx = HookContext(
                        tool_name=tool_name,
                        tool_args=args,
                        skill=skill,
                        skill_name=skill.name if skill else None,
                        cwd=Path.cwd(),
                        skill_dir=skill.skill_dir if skill else None,
                    )
                    hook_result = await self.skill_hooks_manager.trigger_hook(
                        HookEvent.PRE_TOOL_USE, skill_hook_ctx, tool_name=tool_name
                    )
                    if hook_result.output:
                        skill_hook_output = hook_result.output
                        logger.debug(
                            "[ToolCallRunner] PreToolUse hook output: %s",
                            skill_hook_output[:100] if skill_hook_output else None,
                        )

                for hook in tool_hooks:
                    before_hook = getattr(hook, "before_tool_call", None)
                    if before_hook is None:
                        continue
                    updated = await self._invoke_hook(before_hook, hook_context)
                    if isinstance(updated, dict):
                        if (
                            "tool_name" in updated
                            and updated["tool_name"] != hook_context["tool_name"]
                        ):
                            attempted_tool_name = updated["tool_name"]
                            if allow_tool_replace:
                                hook_context["tool_name"] = updated["tool_name"]
                        if "args" in updated:
                            hook_context["args"] = updated["args"]

                tool_name = hook_context["tool_name"]
                args = hook_context["args"]
                skill = hook_context["skill"]
                cache_key = self._build_tool_cache_key(tool_name, args, skill)
                cached_result = None
                if tool_cache is not None and cache_key:
                    cached_result = tool_cache.get(cache_key)
                cache_hit = cached_result is not None
                # Include skill version per SimpleSkill spec §5.4
                skill_version = getattr(skill, "version", None) if skill else None
                self._emit_tool_event(
                    "ToolUsageStarted",
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "requested_tool_name": requested_tool_name,
                        "parallel_group_id": parallel_group_id,
                        "skill_version": skill_version,
                        "args": args,
                        "cache_hit": cache_hit,
                        "cache_key": cache_key,
                    },
                )
                if cache_hit:
                    assert cached_result is not None
                    result = dict(cached_result)
                    metadata = dict(result.get("metadata") or {})
                    metadata["cache_hit"] = True
                    if cache_key:
                        metadata["cache_key"] = cache_key
                    result["metadata"] = metadata
                    raw_payload = result.get("raw")
                    if isinstance(raw_payload, dict):
                        raw_meta = dict(raw_payload.get("metadata") or {})
                        raw_meta["cache_hit"] = True
                        if cache_key:
                            raw_meta["cache_key"] = cache_key
                        raw_payload["metadata"] = raw_meta
                        result["raw"] = raw_payload
                    logger.debug(
                        "[ToolCallRunner] tool_cache_hit tool=%s call_id=%s key=%s",
                        tool_name,
                        tool_call_id,
                        cache_key,
                    )
                else:
                    # --- OTel: tool execution span ---
                    _tool_span = self._start_tool_span(tool_name, tool_call_id, parallel_group_id)
                    result = await self._execute_tool_call(
                        tool_name=tool_name,
                        args=args,
                        skills_by_name=skills_by_name,
                        executor=executor,
                        tool_call_id=tool_call_id,
                    )
                    if _tool_span is not None:
                        _tool_status = "error" if ToolResultBuilder.is_error(result) else "ok"
                        self._end_span(_tool_span, status=_tool_status)
                    if (
                        tool_cache is not None
                        and cache_key
                        and not ToolResultBuilder.is_error(result)
                    ):
                        tool_cache[cache_key] = result
                raw_result = result.get("raw")
                raw_metadata = raw_result.get("metadata") if isinstance(raw_result, dict) else None
                result_metadata = result.get("metadata")
                tool_reported_cache_hit = (
                    bool(result_metadata.get("cache_hit"))
                    if isinstance(result_metadata, dict)
                    else False
                ) or (
                    bool(raw_metadata.get("cache_hit")) if isinstance(raw_metadata, dict) else False
                )
                cache_hit_for_reporting = cache_hit or tool_reported_cache_hit

                if cache_hit_for_reporting:
                    result_meta = dict(result.get("metadata") or {})
                    result_meta["cache_hit"] = True
                    existing_cache_key = result_meta.get("cache_key")
                    raw_cache_key = (
                        raw_metadata.get("cache_key") if isinstance(raw_metadata, dict) else None
                    )
                    if cache_hit and cache_key:
                        result_meta["cache_key"] = cache_key
                    elif existing_cache_key:
                        result_meta["cache_key"] = existing_cache_key
                    elif raw_cache_key:
                        result_meta["cache_key"] = raw_cache_key
                    result["metadata"] = result_meta
                tool_elapsed = time.perf_counter() - tool_start if timing_enabled else 0.0
                if timing_enabled:
                    logger.info(
                        "[ToolCallRunner] tool=%s call_id=%s elapsed=%.3fs",
                        tool_name,
                        tool_call_id,
                        tool_elapsed,
                    )
                if tool_name:
                    called_tools.add(tool_name)
                if resolved_outputs is not None and tool_name:
                    raw_payload = ToolResultBuilder.coerce_payload(result.get("raw"))
                    resolved_value = raw_payload.get("result", raw_payload)
                    resolved_outputs[tool_name] = resolved_value
                    if tool_call_id:
                        resolved_outputs[tool_call_id] = resolved_value
                    call_index_key = str(index + 1)
                    resolved_outputs[call_index_key] = resolved_value
                # Extract latency from result metadata
                result_metadata = result.get("metadata", {})
                latency_ms = (
                    result_metadata.get("latency_ms") if isinstance(result_metadata, dict) else None
                )

                if ToolResultBuilder.is_error(result):
                    self._emit_tool_event(
                        "ToolUsageError",
                        {
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "requested_tool_name": requested_tool_name,
                            "parallel_group_id": parallel_group_id,
                            "error": result.get("raw"),
                            "latency_ms": latency_ms,
                        },
                    )
                    for hook in tool_hooks:
                        error_hook = getattr(hook, "on_tool_error", None)
                        if error_hook is not None:
                            await self._invoke_hook(error_hook, hook_context, result)
                else:
                    self._emit_tool_event(
                        "ToolUsageFinished",
                        {
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "requested_tool_name": requested_tool_name,
                            "parallel_group_id": parallel_group_id,
                            "result": result.get("raw"),
                            "cache_hit": cache_hit_for_reporting,
                            "cache_key": cache_key,
                            "latency_ms": latency_ms,
                        },
                    )
                    for hook in tool_hooks:
                        after_hook = getattr(hook, "after_tool_call", None)
                        if after_hook is not None:
                            await self._invoke_hook(after_hook, hook_context, result)

                # Trigger SimpleSkill PostToolUse hooks
                if self.skill_hooks_manager and tool_name:
                    from houyi.core.skill.hooks import HookContext, HookEvent

                    skill_hook_ctx = HookContext(
                        tool_name=tool_name,
                        tool_args=args,
                        tool_result=result.get("raw"),
                        skill=skill,
                        skill_name=skill.name if skill else None,
                        cwd=Path.cwd(),
                        skill_dir=skill.skill_dir if skill else None,
                    )
                    post_hook_result = await self.skill_hooks_manager.trigger_hook(
                        HookEvent.POST_TOOL_USE,
                        skill_hook_ctx,
                        tool_name=tool_name,
                    )
                    if post_hook_result.output:
                        logger.debug(
                            "[ToolCallRunner] PostToolUse hook output: %s",
                            post_hook_result.output[:100] if post_hook_result.output else None,
                        )

                trace_entry = {
                    "tool_name": tool_name,
                    "requested_tool_name": requested_tool_name,
                    "tool_call_id": tool_call_id,
                    "round_index": round_index_value,
                    "parallel_group_id": parallel_group_id,
                    "args": args,
                    "result": result,
                    "tool_override": (
                        {
                            "from": requested_tool_name,
                            "to": attempted_tool_name,
                            "allowed": allow_tool_replace,
                            "applied": allow_tool_replace
                            and attempted_tool_name != requested_tool_name,
                        }
                        if attempted_tool_name
                        else None
                    ),
                }
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": ToolResultBuilder.format(result),
                }
                if tool_result_summary_enabled:
                    summarized_content, summarized = _summarize_tool_result_content(
                        tool_message["content"],
                        max_chars=tool_result_summary_max_chars,
                        max_items=tool_result_summary_max_items,
                    )
                    if summarized:
                        tool_message["content"] = summarized_content
                        result_meta = dict(result.get("metadata") or {})
                        result_meta["result_summarized"] = True
                        result_meta["result_summary_max_chars"] = tool_result_summary_max_chars
                        result_meta["result_summary_max_items"] = tool_result_summary_max_items
                        result["metadata"] = result_meta
                return index, trace_entry, tool_message, tool_elapsed

            parsed_tool_calls: list[tuple[Any, dict[str, Any] | None]] = []
            has_placeholders = False
            if fast_path_enabled:
                for tool_call in response.tool_calls:
                    tool_payload = (
                        tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    )
                    args = ToolResultBuilder.parse_arguments(tool_payload.get("arguments"))
                    parsed_tool_calls.append((tool_call, args))
                    if PlaceholderResolver.contains(args):
                        has_placeholders = True
            else:
                parsed_tool_calls = [(tool_call, None) for tool_call in response.tool_calls]

            allow_parallel = should_parallel and not (fast_path_enabled and has_placeholders)
            is_parallel_batch = allow_parallel and len(parsed_tool_calls) > 1
            round_parallel_group_id = f"round_{round_index + 1}" if is_parallel_batch else None
            parallel_semaphore = (
                asyncio.Semaphore(max_parallel_calls) if is_parallel_batch else None
            )

            async def _handle_tool_call_with_limit(
                tool_call: Any,
                index: int,
                parsed_args: dict[str, Any] | None = None,
                parallel_group_id: str | None = None,
                round_index_value: int | None = None,
                semaphore: asyncio.Semaphore | None = parallel_semaphore,
            ) -> tuple[int, dict[str, Any], dict[str, Any], float]:
                if semaphore is None:
                    return await _handle_tool_call(
                        tool_call,
                        index,
                        parsed_args=parsed_args,
                        parallel_group_id=parallel_group_id,
                        round_index_value=round_index_value,
                    )
                async with semaphore:
                    return await _handle_tool_call(
                        tool_call,
                        index,
                        parsed_args=parsed_args,
                        parallel_group_id=parallel_group_id,
                        round_index_value=round_index_value,
                    )

            if is_parallel_batch:
                results = await asyncio.gather(
                    *[
                        _handle_tool_call_with_limit(
                            tool_call,
                            index,
                            parsed_args=parsed_args,
                            parallel_group_id=round_parallel_group_id,
                            round_index_value=round_index + 1,
                        )
                        for index, (tool_call, parsed_args) in enumerate(parsed_tool_calls)
                    ]
                )
                for _, trace_entry, tool_message, tool_elapsed in sorted(
                    results, key=lambda item: item[0]
                ):
                    tool_trace.append(trace_entry)
                    tool_messages.append(tool_message)
                    if timing_enabled:
                        tool_durations.append(tool_elapsed)
            else:
                resolved_outputs = tool_outputs if fast_path_enabled else None
                for index, (tool_call, parsed_args) in enumerate(parsed_tool_calls):
                    _, trace_entry, tool_message, tool_elapsed = await _handle_tool_call(
                        tool_call,
                        index,
                        parsed_args=parsed_args,
                        resolved_outputs=resolved_outputs,
                        parallel_group_id=round_parallel_group_id,
                        round_index_value=round_index + 1,
                    )
                    tool_trace.append(trace_entry)
                    tool_messages.append(tool_message)
                    if timing_enabled:
                        tool_durations.append(tool_elapsed)

            messages.extend(tool_messages)

            if fast_path_enabled:
                should_exit = has_placeholders or (
                    all_tool_names and called_tools >= all_tool_names
                )
                if should_exit:
                    if timing_enabled:
                        logger.info(
                            "[ToolCallRunner] fast_path=early_exit round=%s",
                            round_index + 1,
                        )
                    await self._trigger_stop_hook(tool_trace)
                    self._finish_execution_span(_exec_span)
                    return response, tool_trace
                if timing_enabled:
                    logger.info(
                        "[ToolCallRunner] fast_path=continue round=%s tool_calls=%s max_rounds=%s",
                        round_index + 1,
                        len(parsed_tool_calls),
                        max_rounds,
                    )

            if timing_enabled:
                tool_phase_elapsed = time.perf_counter() - tool_phase_start
                tool_sum = sum(tool_durations)
                tool_max = max(tool_durations, default=0.0)
                logger.info(
                    "[ToolCallRunner] round=%s tools=%.3fs sum=%.3fs max=%.3fs parallel=%s",
                    round_index + 1,
                    tool_phase_elapsed,
                    tool_sum,
                    tool_max,
                    allow_parallel and len(parsed_tool_calls) > 1,
                )
                logger.info(
                    "[ToolCallRunner] round=%s total=%.3fs",
                    round_index + 1,
                    time.perf_counter() - round_start,
                )

        await self._trigger_stop_hook(tool_trace)
        self._finish_execution_span(_exec_span)
        return response, tool_trace

    def _clone_llm_response(self, response: Any) -> Any:
        if hasattr(response, "model_copy"):
            return response.model_copy(deep=True)
        return copy.deepcopy(response)

    def _build_llm_cache_key(
        self,
        adapter: Any,
        messages: list[Any],
        tools: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
    ) -> str | None:
        def _normalize_message(message: Any) -> Any:
            if hasattr(message, "model_dump"):
                return message.model_dump()
            return message

        inner_adapter = getattr(adapter, "_inner", None)
        adapter_model = getattr(adapter, "model", None)
        adapter_base_url = getattr(adapter, "base_url", None)
        if inner_adapter is not None:
            adapter_model = adapter_model or getattr(inner_adapter, "model", None)
            adapter_base_url = adapter_base_url or getattr(inner_adapter, "base_url", None)

        adapter_payload = {
            "class": adapter.__class__.__name__,
            "model": adapter_model,
            "base_url": adapter_base_url,
            "tool_choice": getattr(adapter, "_choice", None),
        }
        payload = {
            "adapter": adapter_payload,
            "messages": [_normalize_message(message) for message in messages],
            "tools": tools,
            "chat_kwargs": chat_kwargs,
        }
        try:
            return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        except TypeError:
            return None

    async def _invoke_hook(self, hook_fn: Any, *args: Any, **kwargs: Any) -> Any:
        result = hook_fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _build_tool_cache_key(
        self,
        tool_name: str | None,
        args: dict[str, Any],
        skill: SkillSpec | None,
    ) -> str | None:
        if not tool_name:
            return None
        version = None
        if skill and isinstance(skill.metadata, dict):
            version = skill.metadata.get("version") or skill.metadata.get("tool_version")
        payload = {"tool": tool_name, "args": args, "version": version}
        try:
            return json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except TypeError:
            return None

    def _emit_tool_event(self, name: str, attributes: dict[str, Any]) -> None:
        if not self.trace_manager:
            return
        span = getattr(self.trace_manager, "current_span", None)
        if span is None:
            return
        with contextlib.suppress(Exception):
            span.add_event(name, attributes)

    # --- Observability span helpers (OTel) ---

    def _start_execution_span(self, max_rounds: int, tool_count: int) -> Any:
        """Create and register root EXECUTION span for the tool-call loop."""
        if not self.trace_manager:
            return None
        try:
            from houyi.observability.trace_manager import Span
            from houyi.observability.types import SpanType

            span = Span(
                name="tool_call_runner.run",
                parent=getattr(self.trace_manager, "current_span", None),
                span_type=SpanType.EXECUTION,
                attributes={
                    "execution.max_rounds": max_rounds,
                    "execution.tool_count": tool_count,
                },
            )
            if self.trace_manager.current_span is None:
                self.trace_manager.root_spans.append(span)
            self.trace_manager.current_span = span
            return span
        except Exception:
            logger.debug("Failed to create execution span", exc_info=True)
            return None

    def _start_llm_span(
        self,
        adapter: Any,
        round_index: int,
        requested_model: Any | None = None,
    ) -> Any:
        """Create child LLM span for an adapter.chat() call."""
        if not self.trace_manager:
            return None
        try:
            from houyi.observability.trace_manager import Span
            from houyi.observability.types import SpanType

            model_name = requested_model or getattr(adapter, "model", None)
            span = Span(
                name="llm.call",
                parent=self.trace_manager.current_span,
                span_type=SpanType.LLM,
                model=model_name,
                attributes={
                    "llm.model": model_name,
                    "llm.round": round_index + 1,
                },
            )
            return span
        except Exception:
            logger.debug("Failed to create LLM span", exc_info=True)
            return None

    def _end_span(self, span: Any, status: str = "ok", description: str | None = None) -> None:
        """End a span safely."""
        if span is None:
            return
        with contextlib.suppress(Exception):
            span.set_status(status, description)
            span.end()

    def _start_tool_span(
        self,
        tool_name: str | None,
        tool_call_id: str | None,
        parallel_group_id: str | None = None,
    ) -> Any:
        """Create child TOOL span for a tool execution."""
        if not self.trace_manager:
            return None
        try:
            from houyi.observability.trace_manager import Span
            from houyi.observability.types import SpanType

            effective_name = tool_name or "unknown"
            span = Span(
                name=f"tool.{effective_name}",
                parent=self.trace_manager.current_span,
                span_type=SpanType.TOOL,
                tool_name=effective_name,
                group_id=parallel_group_id,
                attributes={
                    "tool.name": effective_name,
                    "tool.call_id": tool_call_id,
                },
            )
            return span
        except Exception:
            logger.debug("Failed to create tool span", exc_info=True)
            return None

    def _finish_execution_span(self, span: Any, status: str = "ok") -> None:
        """End root execution span and restore trace_manager.current_span."""
        if span is None or not self.trace_manager:
            return
        with contextlib.suppress(Exception):
            span.set_status(status)
            span.end()
            self.trace_manager.current_span = span.parent

    def _get_metrics_collector(self, skill_name: str) -> Any:
        """Get or create a MetricsCollector for a skill."""
        if skill_name not in self._metrics_collectors:
            from houyi.core.skill.metrics import MetricsCollector

            self._metrics_collectors[skill_name] = MetricsCollector(skill_name)
        return self._metrics_collectors[skill_name]

    def _record_metrics(
        self,
        skill_name: str,
        latency_ms: float,
        success: bool,
        is_timeout: bool = False,
    ) -> None:
        """Record execution metrics for a skill."""
        if not self.metrics_store:
            return
        collector = self._get_metrics_collector(skill_name)
        collector.record_latency(latency_ms)
        if success:
            collector.record_success()
        elif is_timeout:
            collector.record_timeout()
        else:
            collector.record_error()
        # Store metrics snapshot
        self.metrics_store.store(collector.get_metrics())

    async def _execute_tool_call(
        self,
        tool_name: str | None,
        args: dict[str, Any],
        skills_by_name: dict[str, SkillSpec],
        executor: Any,
        tool_call_id: str | None,
    ) -> dict[str, Any]:
        """Execute a single tool call using SkillExecutor."""
        if not tool_name:
            return ToolResultBuilder.build(
                {"error": "tool_name_missing"},
                call_id=tool_call_id,
                metadata={"tool_name": tool_name},
            )

        skill = skills_by_name.get(tool_name)
        if not skill:
            return ToolResultBuilder.build(
                {"error": f"tool_not_found: {tool_name}"},
                call_id=tool_call_id,
                metadata={"tool_name": tool_name},
            )

        start_time = time.time()
        try:
            raw_result = await executor.execute(skill, args)
            latency_ms = (time.time() - start_time) * 1000
            self._record_metrics(tool_name, latency_ms, success=True)
            return ToolResultBuilder.build(
                raw_result,
                call_id=tool_call_id,
                metadata={"tool_name": tool_name, "latency_ms": latency_ms},
            )
        except SkillExecutionError as exc:
            latency_ms = (time.time() - start_time) * 1000
            is_timeout = isinstance(exc.original_error, asyncio.TimeoutError)
            self._record_metrics(tool_name, latency_ms, success=False, is_timeout=is_timeout)
            error_type = "timeout" if is_timeout else "execution_error"
            return ToolResultBuilder.build(
                {
                    "error": "tool_execution_failed",
                    "error_type": error_type,
                    "message": exc.message,
                    "skill_name": exc.skill_name,
                    "cause": str(exc.original_error) if exc.original_error else None,
                    "retry_count": getattr(executor, "max_retries", None),
                    "timeout": getattr(executor, "timeout", None),
                },
                call_id=tool_call_id,
                metadata={"tool_name": tool_name, "latency_ms": latency_ms},
            )
        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            if tool_name:
                self._record_metrics(tool_name, latency_ms, success=False)
            return ToolResultBuilder.build(
                {
                    "error": "tool_execution_failed",
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                    "retry_count": getattr(executor, "max_retries", None),
                    "timeout": getattr(executor, "timeout", None),
                },
                call_id=tool_call_id,
                metadata={"tool_name": tool_name, "latency_ms": latency_ms},
            )
