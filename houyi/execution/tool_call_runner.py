"""Shared tool-calling loop runner."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from houyi.core.skill import SkillSpec
from houyi.execution.skill_executor import SkillExecutionError

if TYPE_CHECKING:
    from houyi.core.skill.consent import ConsentManager
    from houyi.core.skill.hooks import SkillHooksManager
    from houyi.core.skill.metrics import MetricsStore
    from houyi.core.skill.policy import PolicyEnforcer

logger = logging.getLogger(__name__)


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
                    try:
                        span.set_attribute(f"{skill_name}.{key}", value)
                    except Exception:
                        pass

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
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Run tool-calling loop until no tool calls are returned."""
        timing_enabled = os.getenv("HOUYI_TOOLCALL_TIMING") == "1"
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
        fast_path_flag = (os.getenv("HOUYI_TOOLCALL_FAST_PATH") or "").strip().lower()
        fast_path_enabled = fast_path_flag in {"1", "true", "yes", "on"}
        tool_outputs: dict[str, Any] = {}
        tool_latency_seconds: float | None = None
        tool_latency_env = os.getenv("HOUYI_TOOLCALL_TOOL_LATENCY_MS")
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

        for round_index in range(max_rounds):
            round_start = time.perf_counter() if timing_enabled else 0.0
            chat_start = time.perf_counter() if timing_enabled else 0.0
            cache_key = self._build_llm_cache_key(
                adapter=adapter,
                messages=messages,
                tools=tools,
                chat_kwargs=chat_kwargs,
            )
            cached_response = None
            if llm_cache is not None and cache_key:
                cached_response = llm_cache.get(cache_key)

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
            else:
                response = await adapter.chat(messages, tools=tools, **chat_kwargs)
                if llm_cache is not None and cache_key:
                    llm_cache[cache_key] = self._clone_llm_response(response)
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
                return response, tool_trace

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                }
            )

            tool_messages: list[dict[str, Any]] = []
            tool_durations: list[float] = []
            tool_phase_start = time.perf_counter() if timing_enabled else 0.0

            async def _handle_tool_call(
                tool_call: Any,
                index: int,
                parsed_args: dict[str, Any] | None = None,
                resolved_outputs: dict[str, Any] | None = None,
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
                    else self._parse_tool_arguments(tool_payload.get("arguments"))
                )
                if resolved_outputs is not None:
                    args = self._resolve_tool_placeholders(args, resolved_outputs)
                    if tool_name == "get_weather_live":
                        args = self._coerce_weather_live_args(args, resolved_outputs)
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
                                    "reason": "consent_denied",
                                },
                            )
                            error_result = self._build_tool_result(
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
                                "args": args,
                                "result": error_result,
                                "policy_blocked": True,
                                "block_reason": "consent_denied",
                            }
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": self._format_tool_result(error_result),
                            }
                            return index, trace_entry, tool_message, 0.0

                    if not decision.allowed:
                        # Policy denied - return error result
                        self._emit_tool_event(
                            "ToolUsageBlocked",
                            {
                                "tool_call_id": tool_call_id,
                                "tool_name": tool_name,
                                "reason": decision.reason or "policy_denied",
                            },
                        )
                        error_result = self._build_tool_result(
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
                            "args": args,
                            "result": error_result,
                            "policy_blocked": True,
                            "block_reason": decision.reason,
                        }
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": self._format_tool_result(error_result),
                        }
                        return index, trace_entry, tool_message, 0.0

                hook_context = {
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

                tool_name = hook_context.get("tool_name")
                args = hook_context.get("args", {})
                skill = hook_context.get("skill")
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
                        "skill_version": skill_version,
                        "args": args,
                        "cache_hit": cache_hit,
                        "cache_key": cache_key,
                    },
                )
                if cache_hit:
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
                    result = await self._execute_tool_call(
                        tool_name=tool_name,
                        args=args,
                        skills_by_name=skills_by_name,
                        executor=executor,
                        tool_call_id=tool_call_id,
                    )
                    if tool_cache is not None and cache_key and not self._is_tool_error(result):
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
                    raw_payload = self._coerce_tool_payload(result.get("raw"))
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

                if self._is_tool_error(result):
                    self._emit_tool_event(
                        "ToolUsageError",
                        {
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "requested_tool_name": requested_tool_name,
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
                        HookEvent.POST_TOOL_USE, skill_hook_ctx, tool_name=tool_name
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
                    "content": self._format_tool_result(result),
                }
                return index, trace_entry, tool_message, tool_elapsed

            parsed_tool_calls: list[tuple[Any, dict[str, Any] | None]] = []
            has_placeholders = False
            if fast_path_enabled:
                for tool_call in response.tool_calls:
                    tool_payload = (
                        tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    )
                    args = self._parse_tool_arguments(tool_payload.get("arguments"))
                    parsed_tool_calls.append((tool_call, args))
                    if self._contains_tool_placeholders(args):
                        has_placeholders = True
            else:
                parsed_tool_calls = [(tool_call, None) for tool_call in response.tool_calls]

            allow_parallel = should_parallel and not (fast_path_enabled and has_placeholders)

            if allow_parallel and len(parsed_tool_calls) > 1:
                results = await asyncio.gather(
                    *[
                        _handle_tool_call(tool_call, index, parsed_args=parsed_args)
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
        try:
            span.add_event(name, attributes)
        except Exception:
            pass

    def _is_tool_error(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        if result.get("is_error"):
            return True
        raw = result.get("raw")
        return isinstance(raw, dict) and "error" in raw

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
            return self._build_tool_result(
                {"error": "tool_name_missing"},
                call_id=tool_call_id,
                metadata={"tool_name": tool_name},
            )

        skill = skills_by_name.get(tool_name)
        if not skill:
            return self._build_tool_result(
                {"error": f"tool_not_found: {tool_name}"},
                call_id=tool_call_id,
                metadata={"tool_name": tool_name},
            )

        start_time = time.time()
        try:
            raw_result = await executor.execute(skill, args)
            latency_ms = (time.time() - start_time) * 1000
            self._record_metrics(tool_name, latency_ms, success=True)
            return self._build_tool_result(
                raw_result,
                call_id=tool_call_id,
                metadata={"tool_name": tool_name, "latency_ms": latency_ms},
            )
        except SkillExecutionError as exc:
            latency_ms = (time.time() - start_time) * 1000
            is_timeout = isinstance(exc.original_error, asyncio.TimeoutError)
            self._record_metrics(tool_name, latency_ms, success=False, is_timeout=is_timeout)
            error_type = "timeout" if is_timeout else "execution_error"
            return self._build_tool_result(
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
            return self._build_tool_result(
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

    def _parse_tool_arguments(self, raw_args: Any) -> dict[str, Any]:
        """Parse tool arguments from model response."""
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _coerce_tool_payload(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "model_dump"):
            return raw.model_dump()
        return {"result": raw}

    def _contains_tool_placeholders(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.startswith("$tool.") or value.startswith("$call.")
        if isinstance(value, dict):
            return any(self._contains_tool_placeholders(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_tool_placeholders(item) for item in value)
        return False

    def _resolve_tool_placeholders(
        self,
        value: Any,
        resolved_outputs: dict[str, Any],
    ) -> Any:
        if isinstance(value, str):
            placeholder = self._extract_placeholder(value)
            if placeholder is None:
                return value
            root_key, path = placeholder
            payload = resolved_outputs.get(root_key)
            if payload is None:
                return value
            return self._resolve_payload_path(payload, path)
        if isinstance(value, dict):
            return {
                key: self._resolve_tool_placeholders(item, resolved_outputs)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_tool_placeholders(item, resolved_outputs) for item in value]
        return value

    @staticmethod
    def _extract_placeholder(value: str) -> tuple[str, list[str]] | None:
        if value.startswith("$tool."):
            path = value[len("$tool.") :]
        elif value.startswith("$call."):
            path = value[len("$call.") :]
        else:
            return None
        parts = [segment for segment in path.split(".") if segment]
        if not parts:
            return None
        return parts[0], parts[1:]

    @staticmethod
    def _resolve_payload_path(payload: Any, path: list[str]) -> Any:
        current = payload
        for segment in path:
            if isinstance(current, dict) and segment in current:
                current = current[segment]
                continue
            if isinstance(current, list):
                try:
                    index = int(segment)
                except ValueError:
                    return payload
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return payload
        return current

    @staticmethod
    def _coerce_weather_live_args(
        args: dict[str, Any] | None,
        resolved_outputs: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(args, dict):
            return args
        updated = dict(args)
        fallback_date = resolved_outputs.get("get_date")
        if isinstance(fallback_date, str):
            updated["date"] = fallback_date
        location = resolved_outputs.get("get_location")
        if isinstance(location, dict):
            if "lat" in location:
                updated["lat"] = location["lat"]
            if "lon" in location:
                updated["lon"] = location["lon"]
        return updated

    def _build_tool_result(
        self,
        raw: Any,
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build tool result payload with raw + serialized content."""
        if isinstance(raw, dict):
            raw_payload = raw
        elif hasattr(raw, "model_dump"):
            raw_payload = raw.model_dump()
        else:
            raw_payload = {"result": raw}
        is_error = isinstance(raw_payload, dict) and "error" in raw_payload

        return {
            "call_id": call_id,
            "raw": raw_payload,
            "content": self._serialize_tool_payload(raw_payload),
            "is_error": is_error,
            "metadata": metadata or {},
        }

    def _format_tool_result(self, result: dict[str, Any]) -> str:
        """Format tool result for tool message content."""
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        return self._serialize_tool_payload(result)

    def _serialize_tool_payload(self, payload: Any) -> str:
        try:
            return json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except TypeError:
            return json.dumps({"result": str(payload)}, ensure_ascii=True, sort_keys=True)
