"""Runtime services backing the ToolCallRunner facade orchestration flow."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from houyi.application.tool_calling.arg_coercion import coerce_args
from houyi.application.tool_calling.budget import MessageBudget, resolve_tool_loop_budget_chars
from houyi.application.tool_calling.context import (
    ToolLoopConfig,
    ToolLoopContext,
    ToolLoopRuntimeServices,
    ToolLoopState,
)
from houyi.application.tool_calling.placeholder_resolver import PlaceholderResolver
from houyi.application.tool_calling.runner_models import (
    _DEFAULT_TOOL_RESULT_SUMMARY_ENABLED,
    _DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS,
    _DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS,
    _BlockedToolCallPresentationRequest,
    _ExecutedToolCall,
    _HookCtx,
    _parse_max_parallel_calls,
    _parse_tool_latency_seconds,
    _PreparedToolCall,
    _read_bool_env,
    _read_positive_int_env_or_none,
    _ToolCallPreparationRequest,
    _ToolCallPresentationRequest,
)
from houyi.application.tool_calling.tool_results import ToolResultBuilder
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
            logger.info(
                "[ToolCallRunner] start: rounds=%s tools=%s messages=%s",
                max_rounds,
                len(tools),
                len(messages),
            )
            if fast_path_enabled:
                logger.info("[ToolCallRunner] fast_path=enabled")

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


class _PreparedToolCallExecutor:
    """Execute a prepared tool call and normalize runtime reporting metadata."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def execute(
        self,
        *,
        prepared: _PreparedToolCall,
        config: Any,
        state: Any,
        services: Any,
        tool_start: float,
    ) -> _ExecutedToolCall:
        """Execute one prepared tool call and normalize timing/cache metadata."""
        result, cache_hit = await self._runner._execution_service._execute_tool_with_cache(
            tool_name=prepared.tool_name,
            requested_tool_name=prepared.requested_tool_name,
            tool_call_id=prepared.tool_call_id,
            parallel_group_id=state.parallel_group_id,
            args=prepared.args,
            skill=prepared.skill,
            cache_key=prepared.cache_key,
            tool_cache=services.tool_result_cache,
            skills_by_name=services.skill_specs_by_name,
            executor=services.tool_executor,
        )

        raw_result = result.get("raw")
        raw_metadata = raw_result.get("metadata") if isinstance(raw_result, dict) else None
        result_metadata = result.get("metadata")
        tool_reported_cache_hit = (
            bool(result_metadata.get("cache_hit")) if isinstance(result_metadata, dict) else False
        ) or (bool(raw_metadata.get("cache_hit")) if isinstance(raw_metadata, dict) else False)

        result = self._runner._enrich_result_with_cache_metadata(
            result,
            cache_hit,
            prepared.cache_key,
            tool_reported_cache_hit,
        )
        cache_hit_for_reporting = cache_hit or tool_reported_cache_hit
        tool_elapsed = time.perf_counter() - tool_start if config.tool_loop_enable_timing else 0.0
        if config.tool_loop_enable_timing:
            logger.info(
                "[ToolCallRunner] tool=%s call_id=%s elapsed=%.3fs",
                prepared.tool_name,
                prepared.tool_call_id,
                tool_elapsed,
            )

        result_meta = result.get("metadata", {})
        latency_ms = result_meta.get("latency_ms") if isinstance(result_meta, dict) else None
        return _ExecutedToolCall(
            result=result,
            cache_hit_for_reporting=cache_hit_for_reporting,
            tool_elapsed=tool_elapsed,
            latency_ms=latency_ms,
        )


class _ToolCallPreparationService:
    """Prepare tool execution inputs before runtime dispatch."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def prepare(
        self,
        request: _ToolCallPreparationRequest,
    ) -> _PreparedToolCall | tuple[int, dict[str, Any], dict[str, Any], float]:
        """Resolve inputs, apply consent/hook policies, and build prepared call payload."""
        tool_name, tool_call_id, args, skill = self._resolve_tool_call_inputs(
            tool_call=request.tool_call,
            parsed_args=request.parsed_args,
            resolved_outputs=request.resolved_outputs,
            skills_by_name=request.skills_by_name,
        )
        requested_name = tool_name
        consent_rejection = await self._handle_consent_rejection(
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id,
            index=request.index,
            round_index_value=request.round_index_value,
            parallel_group_id=request.parallel_group_id,
            requested_tool_name=requested_name,
        )
        if consent_rejection is not None:
            return consent_rejection

        hook_context, attempted_tool_name = await self._apply_before_tool_hooks(
            tool_name=tool_name,
            args=args,
            skill=skill,
            tool_call_id=tool_call_id,
            tool_hooks=request.tool_hooks,
            allow_tool_replace=request.allow_tool_replace,
        )
        current_tool_name = hook_context["tool_name"]
        current_args = hook_context["args"]
        current_skill = hook_context["skill"]
        return _PreparedToolCall(
            requested_tool_name=requested_name,
            tool_name=current_tool_name,
            tool_call_id=tool_call_id,
            args=current_args,
            skill=current_skill,
            hook_context=hook_context,
            attempted_tool_name=attempted_tool_name,
            cache_key=self._runner._execution_service._build_tool_cache_key(
                current_tool_name,
                current_args,
                current_skill,
            ),
        )

    def _resolve_tool_call_inputs(
        self,
        *,
        tool_call: Any,
        parsed_args: dict[str, Any] | None,
        resolved_outputs: dict[str, Any] | None,
        skills_by_name: dict[str, SkillSpec],
    ) -> tuple[str | None, str | None, dict[str, Any], SkillSpec | None]:
        """Parse tool-call payload and resolve arguments into executable skill inputs."""
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
        skill = skills_by_name.get(tool_name) if tool_name else None
        return tool_name, tool_call_id, args, skill

    async def _handle_consent_rejection(
        self,
        *,
        tool_name: str | None,
        args: dict[str, Any],
        tool_call_id: str | None,
        index: int,
        round_index_value: int | None,
        parallel_group_id: str | None,
        requested_tool_name: str | None,
    ) -> tuple[int, dict[str, Any], dict[str, Any], float] | None:
        if not tool_name:
            return None
        consent_granted, trace_entry, tool_message = await self._check_consent_if_required(
            tool_name,
            args,
            tool_call_id,
        )
        if consent_granted:
            return None
        if trace_entry and tool_message:
            trace_entry["round_index"] = round_index_value
            trace_entry["parallel_group_id"] = parallel_group_id
            trace_entry["requested_tool_name"] = requested_tool_name
        return index, trace_entry or {}, tool_message or {}, 0.0

    async def _check_consent_if_required(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str | None,
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
        if not self._runner.policy_enforcer:
            return True, None, None

        consent_given = self._runner._consent_cache.get(tool_name, False)
        decision = self._runner.policy_enforcer.check_invocation(
            skill_name=tool_name,
            is_model_initiated=True,
            user_consent_given=consent_given,
        )

        if decision.requires_consent and self._runner.consent_manager:
            from houyi.domain.skill.consent import ConsentRequest, ConsentType

            policy = self._runner.policy_enforcer.get_policy(tool_name)
            consent_request = ConsentRequest(
                consent_type=ConsentType.INVOKE_CONFIRM,
                skill_name=tool_name,
                operation=f"invoke tool '{tool_name}'",
                policy=policy,
                context={"args": args, "tool_call_id": tool_call_id},
            )
            consent_response = await self._runner.consent_manager.request_consent(consent_request)
            if consent_response.is_granted():
                self._runner._consent_cache[tool_name] = True
                return True, None, None

            self._runner._emit_tool_usage_blocked(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                reason="consent_denied",
            )
            trace_entry, tool_message = self._runner._build_blocked_tool_trace_and_message(
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                error_code="consent_denied",
                message=f"User denied consent for tool '{tool_name}'",
                block_reason="consent_denied",
            )
            return False, trace_entry, tool_message

        if not decision.allowed:
            policy_reason = decision.reason or "policy_denied"
            self._runner._emit_tool_usage_blocked(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                reason=policy_reason,
            )
            trace_entry, tool_message = self._runner._build_blocked_tool_trace_and_message(
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                error_code="policy_denied",
                message=decision.reason or f"Policy denied invocation of tool '{tool_name}'",
                block_reason=policy_reason,
            )
            return False, trace_entry, tool_message

        return True, None, None

    async def _apply_before_tool_hooks(
        self,
        *,
        tool_name: str | None,
        args: dict[str, Any],
        skill: SkillSpec | None,
        tool_call_id: str | None,
        tool_hooks: list[Any],
        allow_tool_replace: bool,
    ) -> tuple[_HookCtx, str | None]:
        hook_context: _HookCtx = {
            "tool_name": tool_name,
            "args": args,
            "skill": skill,
            "tool_call_id": tool_call_id,
        }
        attempted_tool_name: str | None = None

        if tool_name:
            await self._trigger_pre_tool_use_hook(tool_name, args, skill)

        for hook in tool_hooks:
            before_hook = getattr(hook, "before_tool_call", None)
            if before_hook is None:
                continue
            updated = await self._runner._execution_service._invoke_hook(before_hook, hook_context)
            if not isinstance(updated, dict):
                continue
            if "tool_name" in updated and updated["tool_name"] != hook_context["tool_name"]:
                attempted_tool_name = updated["tool_name"]
                if allow_tool_replace:
                    hook_context["tool_name"] = updated["tool_name"]
            if "args" in updated:
                hook_context["args"] = updated["args"]

        return hook_context, attempted_tool_name

    async def _trigger_pre_tool_use_hook(
        self,
        tool_name: str,
        args: dict[str, Any],
        skill: SkillSpec | None,
    ) -> str | None:
        if not self._runner.skill_hooks_manager:
            return None
        from houyi.domain.skill.hooks import HookContext, HookEvent

        skill_hook_ctx = HookContext(
            tool_name=tool_name,
            tool_args=args,
            skill=skill,
            skill_name=skill.name if skill else None,
            cwd=Path.cwd(),
            skill_dir=skill.skill_dir if skill else None,
        )
        hook_result = await self._runner.skill_hooks_manager.trigger_hook(
            HookEvent.PRE_TOOL_USE,
            skill_hook_ctx,
            tool_name=tool_name,
        )
        if hook_result.output:
            logger.debug(
                "[ToolCallRunner] PreToolUse hook output: %s",
                hook_result.output[:100] if hook_result.output else None,
            )
            return hook_result.output
        return None


class _ToolCallResultPresenter:
    """Build trace entries and tool messages for tool-call outcomes."""

    def build_blocked_trace_and_message(
        self,
        request: _BlockedToolCallPresentationRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        error_result = ToolResultBuilder.build(
            {
                "error": request.error_code,
                "message": request.message,
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": request.tool_name, "policy_blocked": True},
        )
        trace_entry = {
            "tool_name": request.tool_name,
            "requested_tool_name": request.tool_name,
            "tool_call_id": request.tool_call_id,
            "args": request.args,
            "result": error_result,
            "policy_blocked": True,
            "block_reason": request.block_reason,
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": request.tool_call_id,
            "name": request.tool_name,
            "content": ToolResultBuilder.format(error_result),
        }
        return trace_entry, tool_message

    def build_trace_and_message(
        self,
        request: _ToolCallPresentationRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        trace_entry = {
            "tool_name": request.tool_name,
            "requested_tool_name": request.requested_tool_name,
            "tool_call_id": request.tool_call_id,
            "round_index": request.round_index_value,
            "parallel_group_id": request.parallel_group_id,
            "args": request.args,
            "result": request.result,
            "tool_override": (
                {
                    "from": request.requested_tool_name,
                    "to": request.attempted_tool_name,
                    "allowed": request.allow_tool_replace,
                    "applied": request.allow_tool_replace
                    and request.attempted_tool_name != request.requested_tool_name,
                }
                if request.attempted_tool_name
                else None
            ),
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": request.tool_call_id,
            "name": request.tool_name,
            "content": ToolResultBuilder.format(request.result),
        }
        if not request.tool_result_summary_enabled:
            return trace_entry, tool_message

        summarized_content, summarized = MessageBudget.summarize_tool_result(
            str(tool_message["content"]),
            max_chars=request.tool_result_summary_max_chars,
            max_items=request.tool_result_summary_max_items,
        )
        if summarized:
            tool_message["content"] = summarized_content
            result_meta = dict(request.result.get("metadata") or {})
            result_meta["result_summarized"] = True
            result_meta["result_summary_max_chars"] = request.tool_result_summary_max_chars
            result_meta["result_summary_max_items"] = request.tool_result_summary_max_items
            request.result["metadata"] = result_meta
        return trace_entry, tool_message


class _ToolCallEventDispatcher:
    """Dispatch tool-calling lifecycle events and hook callbacks."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def dispatch(self, name: str, attributes: dict[str, Any]) -> None:
        trace_manager = self._runner.trace_manager
        if not trace_manager:
            return
        span = getattr(trace_manager, "current_span", None)
        if span is None:
            return
        with contextlib.suppress(Exception):
            span.add_event(name, attributes)

    def emit_usage_blocked(
        self,
        *,
        tool_call_id: str | None,
        tool_name: str,
        reason: str,
    ) -> None:
        self.dispatch(
            "ToolUsageBlocked",
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "parallel_group_id": None,
                "reason": reason,
            },
        )

    async def emit_usage_outcome(
        self,
        *,
        tool_hooks: list[Any],
        hook_context: _HookCtx,
        result: dict[str, Any],
        tool_call_id: str | None,
        tool_name: str | None,
        requested_tool_name: str | None,
        parallel_group_id: str | None,
        cache_hit_for_reporting: bool,
        cache_key: str | None,
        latency_ms: Any,
    ) -> None:
        if ToolResultBuilder.is_error(result):
            self.dispatch(
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
            hook_name = "on_tool_error"
        else:
            self.dispatch(
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
            hook_name = "after_tool_call"

        for hook in tool_hooks:
            callback = getattr(hook, hook_name, None)
            if callback is not None:
                await self._runner._execution_service._invoke_hook(callback, hook_context, result)


class _ToolCallLifecycleService:
    """Handle session lifecycle hooks, router filtering, and preprocessors."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def run_preprocessors(self, preprocessors: list[Any], messages: list[Any]) -> list[Any]:
        """Run preprocessors before the first LLM call and inject outputs into messages."""
        from houyi.domain.skill.preprocessor import PreprocessorPipeline

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
        return messages

    async def trigger_session_start_hook(
        self,
        max_rounds: int,
        tool_count: int,
        skill_count: int,
    ) -> None:
        """Trigger SessionStart hook with run-level counts."""
        if not self._runner.skill_hooks_manager:
            return
        from houyi.domain.skill.hooks import HookContext, HookEvent

        session_ctx = HookContext(
            tool_name="__session__",
            tool_args={
                "max_rounds": max_rounds,
                "tool_count": tool_count,
                "skill_count": skill_count,
            },
        )
        try:
            await self._runner.skill_hooks_manager.trigger_hook(
                HookEvent.SESSION_START, session_ctx
            )
        except Exception:
            logger.debug("SessionStart hook error (non-fatal)", exc_info=True)

    def apply_tool_router(
        self,
        skills: list[SkillSpec],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter available tools using ToolRouter restrictions."""
        from houyi.domain.skill.tool_router import ToolRouter

        tool_router = ToolRouter(skills, self._runner.policy_enforcer)
        if not tool_router.has_restrictions:
            return tools
        original_count = len(tools)
        filtered_tools = tool_router.filter_tools(tools)
        logger.debug(
            "ToolRouter: filtered %d → %d tools",
            original_count,
            len(filtered_tools),
        )
        return filtered_tools

    async def trigger_post_tool_use_hook(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        skill: SkillSpec | None,
    ) -> None:
        """Trigger PostToolUse hook after tool result is produced."""
        if not self._runner.skill_hooks_manager:
            return
        from houyi.domain.skill.hooks import HookContext, HookEvent

        skill_hook_ctx = HookContext(
            tool_name=tool_name,
            tool_args=args,
            tool_result=result.get("raw"),
            skill=skill,
            skill_name=skill.name if skill else None,
            cwd=Path.cwd(),
            skill_dir=skill.skill_dir if skill else None,
        )
        post_hook_result = await self._runner.skill_hooks_manager.trigger_hook(
            HookEvent.POST_TOOL_USE,
            skill_hook_ctx,
            tool_name=tool_name,
        )
        if post_hook_result.output:
            logger.debug(
                "[ToolCallRunner] PostToolUse hook output: %s",
                post_hook_result.output[:100] if post_hook_result.output else None,
            )

    async def trigger_stop_hook(self, tool_trace: list[dict[str, Any]]) -> None:
        """Trigger Stop hook at the end of a tool-calling session."""
        if not self._runner.skill_hooks_manager:
            return
        from houyi.domain.skill.hooks import HookContext, HookEvent

        stop_ctx = HookContext(
            tool_name="__session__",
            tool_args={"tool_trace_length": len(tool_trace)},
        )
        try:
            await self._runner.skill_hooks_manager.trigger_hook(HookEvent.STOP, stop_ctx)
        except Exception:
            logger.debug("Stop hook error (non-fatal)", exc_info=True)
