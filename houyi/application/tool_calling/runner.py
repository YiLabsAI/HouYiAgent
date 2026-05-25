"""Shared tool-calling loop runner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from houyi.application.tool_calling.context import (
    ToolCallExecutionContext,
)
from houyi.application.tool_calling.event_dispatcher import _ToolCallEventDispatcher
from houyi.application.tool_calling.lifecycle_service import _ToolCallLifecycleService
from houyi.application.tool_calling.orchestrator import ToolLoopOrchestrator
from houyi.application.tool_calling.placeholder_resolver import PlaceholderResolver
from houyi.application.tool_calling.preparation_hook_service import (
    _ToolCallPreparationHookService,
)
from houyi.application.tool_calling.preparation_policy_service import (
    _ToolCallPreparationPolicyService,
)
from houyi.application.tool_calling.preparation_service import _ToolCallPreparationService
from houyi.application.tool_calling.prepared_tool_call_executor import (
    _PreparedToolCallExecutor,
)
from houyi.application.tool_calling.result_presenter import _ToolCallResultPresenter
from houyi.application.tool_calling.runner_execution_service import _ToolCallExecutionService
from houyi.application.tool_calling.runner_models import (
    _BlockedToolCallPresentationRequest,
    _ExecutedToolCall,
    _parse_max_parallel_calls,
    _parse_tool_latency_seconds,
    _PreparedToolCall,
    _ToolCallPreparationRequest,
)
from houyi.application.tool_calling.runner_runtime_services import (
    _ToolLoopSessionBuilder,
)
from houyi.application.tool_calling.tool_results import ToolResultBuilder
from houyi.domain.skill.spec import SkillSpec

if TYPE_CHECKING:
    from houyi.domain.skill.consent import ConsentManager
    from houyi.domain.skill.hooks import SkillHooksManager
    from houyi.domain.skill.metrics import MetricsStore
    from houyi.domain.skill.policy import PolicyEnforcer

logger = logging.getLogger(__name__)


class _ToolCallRunnerBase:
    """Shared state and collaborator wiring for ToolCallRunner."""

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
        self._preparation_hook_service = _ToolCallPreparationHookService(self)
        self._preparation_policy_service = _ToolCallPreparationPolicyService(self)
        self._preparation_service = _ToolCallPreparationService(self)
        self._lifecycle_service = _ToolCallLifecycleService(self)
        self._prepared_tool_call_executor = _PreparedToolCallExecutor(self)
        self._result_presenter = _ToolCallResultPresenter()
        self._event_dispatcher = _ToolCallEventDispatcher(self)
        self._execution_service = _ToolCallExecutionService(self)

    @staticmethod
    def _parse_max_parallel_calls(chat_kwargs: dict[str, Any]) -> int:
        return _parse_max_parallel_calls(chat_kwargs)

    @staticmethod
    def _parse_tool_latency_seconds() -> float | None:
        return _parse_tool_latency_seconds()

    @staticmethod
    def _build_tool_call_preparation_request(
        *,
        tool_call: Any,
        parsed_args: dict[str, Any] | None,
        resolved_outputs: dict[str, Any] | None,
        skills_by_name: dict[str, SkillSpec],
        tool_hooks: list[Any],
        allow_tool_replace: bool,
        index: int,
        round_index_value: int | None,
        parallel_group_id: str | None,
    ) -> _ToolCallPreparationRequest:
        return _ToolCallPreparationRequest(
            tool_call=tool_call,
            parsed_args=parsed_args,
            resolved_outputs=resolved_outputs,
            skills_by_name=skills_by_name,
            tool_hooks=tool_hooks,
            allow_tool_replace=allow_tool_replace,
            index=index,
            round_index_value=round_index_value,
            parallel_group_id=parallel_group_id,
        )


class ToolCallRunner(_ToolCallRunnerBase):
    """Run tool-calling loops with hooks, policy enforcement, and trace events."""

    async def _run_preprocessors(self, preprocessors: list[Any], messages: list[Any]) -> list[Any]:
        """Run preprocessors before the first LLM call and inject outputs into messages."""
        return await self._lifecycle_service.run_preprocessors(preprocessors, messages)

    async def _trigger_session_start_hook(
        self, max_rounds: int, tool_count: int, skill_count: int
    ) -> None:
        await self._lifecycle_service.trigger_session_start_hook(
            max_rounds, tool_count, skill_count
        )

    def _apply_tool_router(
        self, skills: list[SkillSpec], tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self._lifecycle_service.apply_tool_router(skills, tools)

    def _emit_tool_usage_blocked(
        self,
        *,
        tool_call_id: str | None,
        tool_name: str,
        reason: str,
    ) -> None:
        self._event_dispatcher.emit_usage_blocked(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reason=reason,
        )

    def _build_blocked_tool_trace_and_message(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str | None,
        error_code: str,
        message: str,
        block_reason: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._result_presenter.build_blocked_trace_and_message(
            _BlockedToolCallPresentationRequest(
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                error_code=error_code,
                message=message,
                block_reason=block_reason,
            )
        )

    def _get_cached_tool_result(
        self,
        tool_cache: dict[str, dict[str, Any]] | None,
        cache_key: str | None,
    ) -> dict[str, Any] | None:
        return self._execution_service._get_cached_tool_result(tool_cache, cache_key)

    async def _trigger_post_tool_use_hook(
        self, tool_name: str, args: dict[str, Any], result: dict[str, Any], skill: SkillSpec | None
    ) -> None:
        await self._lifecycle_service.trigger_post_tool_use_hook(tool_name, args, result, skill)

    def _build_llm_cache_key(
        self,
        adapter: Any,
        messages: list[Any],
        tools: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
    ) -> str | None:
        return self._execution_service._build_llm_cache_key(adapter, messages, tools, chat_kwargs)

    def _build_tool_cache_key(
        self, tool_name: str | None, args: dict[str, Any], skill: SkillSpec | None
    ) -> str | None:
        return self._execution_service._build_tool_cache_key(tool_name, args, skill)

    async def _call_llm_with_cache(
        self,
        adapter: Any,
        chat_messages: list[Any],
        tools: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
        llm_cache: dict[str, Any] | None,
        round_index: int,
    ) -> tuple[Any, Any | None]:
        return await self._execution_service.execute_llm_with_cache(
            adapter=adapter,
            chat_messages=chat_messages,
            tools=tools,
            chat_kwargs=chat_kwargs,
            llm_cache=llm_cache,
            round_index=round_index,
        )

    @staticmethod
    def _parse_tool_calls_for_fast_path(
        tool_calls: list[Any], fast_path_enabled: bool
    ) -> tuple[list[tuple[Any, dict[str, Any] | None]], bool]:
        parsed_tool_calls: list[tuple[Any, dict[str, Any] | None]] = []
        has_placeholders = False
        if fast_path_enabled:
            for tool_call in tool_calls:
                tool_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                args = ToolResultBuilder.parse_arguments(tool_payload.get("arguments"))
                parsed_tool_calls.append((tool_call, args))
                if PlaceholderResolver.contains(args):
                    has_placeholders = True
        else:
            parsed_tool_calls = [(tool_call, None) for tool_call in tool_calls]
        return parsed_tool_calls, has_placeholders

    def _should_exit_fast_path(
        self,
        fast_path_enabled: bool,
        has_placeholders: bool,
        all_tool_names: set[str],
        called_tools: set[str],
    ) -> bool:
        if not fast_path_enabled:
            return False
        return has_placeholders or (bool(all_tool_names) and called_tools >= all_tool_names)

    def _enrich_result_with_cache_metadata(
        self,
        result: dict[str, Any],
        cache_hit: bool,
        cache_key: str | None,
        tool_reported_cache_hit: bool,
    ) -> dict[str, Any]:
        return self._execution_service.enrich_result_with_cache_metadata(
            result,
            cache_hit,
            cache_key,
            tool_reported_cache_hit,
        )

    async def _trigger_stop_hook(self, tool_trace: list[dict[str, Any]]) -> None:
        """Trigger the Stop hook at the end of a tool-calling session."""
        await self._lifecycle_service.trigger_stop_hook(tool_trace)

    def get_skill_metrics(self, skill_name: str) -> Any:
        """Get aggregated metrics for a skill.

        Returns None if no metrics have been collected or no metrics_store is configured.
        """
        return self._execution_service.get_skill_metrics(skill_name)

    def get_all_skill_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics for all skills.

        Returns empty dict if no metrics_store is configured.
        """
        return self._execution_service.get_all_skill_metrics()

    def export_metrics_to_trace(self) -> None:
        """Export aggregated metrics to current trace span as OpenTelemetry attributes.

        This should be called at the end of an execution to attach metrics summary
        to the trace for observability.
        """
        self._execution_service.export_metrics_to_trace()

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
                provided, a PreprocessorPipeline executes them
                **before** the first LLM call and injects their outputs into
                the message context.
        """

        # --- Preprocessors: deterministic pre-LLM execution (M8) ---
        if preprocessors:
            messages = await self._run_preprocessors(preprocessors, messages)

        # --- SessionStart hook ---
        await self._trigger_session_start_hook(max_rounds, len(tools), len(skills))

        # --- Tool Router: allowed-tools whitelist enforcement (M9) ---
        tools = self._apply_tool_router(skills, tools)
        tool_hooks = tool_hooks or []
        chat_kwargs = chat_kwargs or {}
        loop_ctx, tool_trace, _exec_span = _ToolLoopSessionBuilder(self).build(
            adapter=adapter,
            messages=messages,
            tools=tools,
            skills=skills,
            executor=executor,
            max_rounds=max_rounds,
            chat_kwargs=chat_kwargs,
            tool_hooks=tool_hooks,
            allow_tool_replace=allow_tool_replace,
            tool_cache=tool_cache,
            llm_cache=llm_cache,
        )
        response = await ToolLoopOrchestrator.execute_rounds(loop_ctx)

        return await self._stop_and_return(response, tool_trace, _exec_span)

    async def _stop_and_return(
        self,
        response: Any,
        tool_trace: list[dict[str, Any]],
        exec_span: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Finalize stop hooks/spans and return the final model response plus tool trace."""
        await self._trigger_stop_hook(tool_trace)
        self._execution_service._finish_execution_span(exec_span)
        return response, tool_trace

    async def _handle_tool_call_impl(
        self,
        exec_ctx: ToolCallExecutionContext,
    ) -> tuple[int, dict[str, Any], dict[str, Any], float]:
        return await self._execution_service.handle_tool_call_impl(exec_ctx)

    async def _prepare_tool_call_execution(
        self,
        *,
        request: _ToolCallPreparationRequest,
    ) -> _PreparedToolCall | tuple[int, dict[str, Any], dict[str, Any], float]:
        return await self._preparation_service.prepare(request)

    async def _run_prepared_tool_call(
        self,
        *,
        prepared: _PreparedToolCall,
        config: Any,
        state: Any,
        services: Any,
        tool_start: float,
    ) -> _ExecutedToolCall:
        return await self._prepared_tool_call_executor.execute(
            prepared=prepared,
            config=config,
            state=state,
            services=services,
            tool_start=tool_start,
        )
