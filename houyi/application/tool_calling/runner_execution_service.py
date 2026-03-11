"""Execution/runtime collaborator for ToolCallRunner."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from houyi.application.tool_calling.context import ToolCallExecutionContext
from houyi.application.tool_calling.execution import (
    ToolSkillExecutionRequest,
    ToolSkillExecutionServices,
    execute_tool_skill,
)
from houyi.application.tool_calling.runner_models import _HookCtx, _ToolCallPresentationRequest
from houyi.application.tool_calling.tool_results import ToolResultBuilder
from houyi.domain.skill.spec import SkillSpec

if TYPE_CHECKING:
    from houyi.domain.skill.metrics import MetricsStore

logger = logging.getLogger(__name__)


class _ToolCallExecutionService:
    """Handle execution/cache/span runtime concerns for ToolCallRunner."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    @property
    def trace_manager(self) -> Any | None:
        return self._runner.trace_manager

    @property
    def metrics_store(self) -> MetricsStore | None:
        return self._runner.metrics_store

    @property
    def _event_dispatcher(self) -> Any:
        return self._runner._event_dispatcher

    @property
    def _result_presenter(self) -> Any:
        return self._runner._result_presenter

    @property
    def _metrics_collectors(self) -> dict[str, Any]:
        return self._runner._metrics_collectors

    def _get_cached_tool_result(
        self,
        tool_cache: dict[str, dict[str, Any]] | None,
        cache_key: str | None,
    ) -> dict[str, Any] | None:
        if tool_cache is None or not cache_key:
            return None
        cached_result = tool_cache.get(cache_key)
        if cached_result is None:
            return None

        result = dict(cached_result)
        metadata = dict(result.get("metadata") or {})
        metadata["cache_hit"] = True
        metadata["cache_key"] = cache_key
        result["metadata"] = metadata
        raw_payload = result.get("raw")
        if isinstance(raw_payload, dict):
            raw_meta = dict(raw_payload.get("metadata") or {})
            raw_meta["cache_hit"] = True
            raw_meta["cache_key"] = cache_key
            raw_payload["metadata"] = raw_meta
            result["raw"] = raw_payload
        return result

    async def _emit_tool_usage_outcome(
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
        """Forward final tool outcome events and hook callbacks via dispatcher."""
        await self._event_dispatcher.emit_usage_outcome(
            tool_hooks=tool_hooks,
            hook_context=hook_context,
            result=result,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            requested_tool_name=requested_tool_name,
            parallel_group_id=parallel_group_id,
            cache_hit_for_reporting=cache_hit_for_reporting,
            cache_key=cache_key,
            latency_ms=latency_ms,
        )

    def _build_tool_trace_and_message(
        self,
        *,
        tool_name: str | None,
        requested_tool_name: str | None,
        tool_call_id: str | None,
        round_index_value: int | None,
        parallel_group_id: str | None,
        duration_ms: float | None,
        args: dict[str, Any],
        result: dict[str, Any],
        attempted_tool_name: str | None,
        allow_tool_replace: bool,
        tool_result_summary_enabled: bool,
        tool_result_summary_max_chars: int,
        tool_result_summary_max_items: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._result_presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name=tool_name,
                requested_tool_name=requested_tool_name,
                tool_call_id=tool_call_id,
                round_index_value=round_index_value,
                parallel_group_id=parallel_group_id,
                duration_ms=duration_ms,
                args=args,
                result=result,
                attempted_tool_name=attempted_tool_name,
                allow_tool_replace=allow_tool_replace,
                tool_result_summary_enabled=tool_result_summary_enabled,
                tool_result_summary_max_chars=tool_result_summary_max_chars,
                tool_result_summary_max_items=tool_result_summary_max_items,
            )
        )

    async def _execute_tool_with_cache(
        self,
        *,
        tool_name: str | None,
        requested_tool_name: str | None,
        tool_call_id: str | None,
        parallel_group_id: str | None,
        args: dict[str, Any],
        skill: SkillSpec | None,
        cache_key: str | None,
        tool_cache: dict[str, dict[str, Any]] | None,
        skills_by_name: dict[str, SkillSpec],
        executor: Any,
    ) -> tuple[dict[str, Any], bool]:
        """Execute tool call with cache lookup/write-through and start event emission."""
        cached_result = tool_cache.get(cache_key) if tool_cache is not None and cache_key else None
        cache_hit = cached_result is not None
        self._emit_tool_event(
            "ToolUsageStarted",
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "requested_tool_name": requested_tool_name,
                "parallel_group_id": parallel_group_id,
                "skill_version": getattr(skill, "version", None) if skill else None,
                "args": args,
                "cache_hit": cache_hit,
                "cache_key": cache_key,
            },
        )

        if cache_hit:
            result = self._get_cached_tool_result(tool_cache, cache_key)
            assert result is not None
            logger.debug(
                "[ToolCallRunner] tool_cache_hit tool=%s call_id=%s key=%s",
                tool_name,
                tool_call_id,
                cache_key,
            )
            return result, True

        result = await self._execute_tool_with_span(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            parallel_group_id=parallel_group_id,
            args=args,
            skills_by_name=skills_by_name,
            executor=executor,
        )
        if tool_cache is not None and cache_key and not ToolResultBuilder.is_error(result):
            tool_cache[cache_key] = result
        return result, False

    async def _execute_tool_with_span(
        self,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
        parallel_group_id: str | None,
        args: dict[str, Any],
        skills_by_name: dict[str, SkillSpec],
        executor: Any,
    ) -> dict[str, Any]:
        tool_span = self._start_tool_span(tool_name, tool_call_id, parallel_group_id)
        result = await self._execute_tool_call(
            tool_name=tool_name,
            args=args,
            skills_by_name=skills_by_name,
            executor=executor,
            tool_call_id=tool_call_id,
        )
        if tool_span is not None:
            tool_status = "error" if ToolResultBuilder.is_error(result) else "ok"
            self._end_span(tool_span, status=tool_status)
        return result

    def _update_resolved_outputs(
        self,
        *,
        resolved_outputs: dict[str, Any] | None,
        tool_name: str | None,
        tool_call_id: str | None,
        index: int,
        result: dict[str, Any],
    ) -> None:
        if resolved_outputs is None or not tool_name:
            return
        raw_payload = ToolResultBuilder.coerce_payload(result.get("raw"))
        resolved_value = raw_payload.get("result", raw_payload)
        resolved_outputs[tool_name] = resolved_value
        if tool_call_id:
            resolved_outputs[tool_call_id] = resolved_value
        resolved_outputs[str(index + 1)] = resolved_value

    def _clone_llm_response(self, response: Any) -> Any:
        if hasattr(response, "model_copy"):
            return response.model_copy(deep=True)
        return copy.deepcopy(response)

    async def execute_llm_with_cache(
        self,
        *,
        adapter: Any,
        chat_messages: list[Any],
        tools: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
        llm_cache: dict[str, Any] | None,
        round_index: int,
    ) -> tuple[Any, Any | None]:
        """Execute adapter chat with optional LLM cache and span/token enrichment."""
        cache_key = self._build_llm_cache_key(
            adapter=adapter,
            messages=chat_messages,
            tools=tools,
            chat_kwargs=chat_kwargs,
        )
        cached_response = llm_cache.get(cache_key) if llm_cache is not None and cache_key else None

        requested_model = chat_kwargs.get("model") if isinstance(chat_kwargs, dict) else None
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
        return response, _llm_span

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
        self._event_dispatcher.dispatch(name, attributes)

    def _start_execution_span(self, max_rounds: int, tool_count: int) -> Any:
        """Create and register root EXECUTION span for the tool-call loop."""
        if not self.trace_manager:
            return None
        try:
            from houyi.infrastructure.observability import Span, SpanType

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
            from houyi.infrastructure.observability import Span, SpanType

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
            from houyi.infrastructure.observability import Span, SpanType

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
            from houyi.domain.skill.metrics import MetricsCollector

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
        self.metrics_store.store(collector.get_metrics())

    def enrich_result_with_cache_metadata(
        self,
        result: dict[str, Any],
        cache_hit: bool,
        cache_key: str | None,
        tool_reported_cache_hit: bool,
    ) -> dict[str, Any]:
        """Normalize cache metadata on tool result payload for reporting consistency."""
        cache_hit_for_reporting = cache_hit or tool_reported_cache_hit
        if not cache_hit_for_reporting:
            return result

        result_meta = dict(result.get("metadata") or {})
        result_meta["cache_hit"] = True

        raw_result = result.get("raw")
        raw_metadata = raw_result.get("metadata") if isinstance(raw_result, dict) else None
        existing_cache_key = result_meta.get("cache_key")
        raw_cache_key = raw_metadata.get("cache_key") if isinstance(raw_metadata, dict) else None

        if cache_hit and cache_key:
            result_meta["cache_key"] = cache_key
        elif existing_cache_key:
            result_meta["cache_key"] = existing_cache_key
        elif raw_cache_key:
            result_meta["cache_key"] = raw_cache_key

        result["metadata"] = result_meta
        return result

    def get_skill_metrics(self, skill_name: str) -> Any:
        """Return aggregated metrics for one skill, or None when unavailable."""
        if not self.metrics_store:
            return None
        return self.metrics_store.aggregate(skill_name)

    def get_all_skill_metrics(self) -> dict[str, Any]:
        """Return aggregated metrics for all tracked skills."""
        if not self.metrics_store:
            return {}
        return {
            skill_name: self.metrics_store.aggregate(skill_name)
            for skill_name in self.metrics_store.list_skills()
        }

    def export_metrics_to_trace(self) -> None:
        """Export metrics snapshots into current trace span attributes."""
        if not self.trace_manager or not self.metrics_store:
            return
        span = getattr(self.trace_manager, "current_span", None)
        if span is None:
            return

        from houyi.domain.skill.metrics import MetricsExporter

        for skill_name in self.metrics_store.list_skills():
            metrics = self.metrics_store.aggregate(skill_name)
            if metrics:
                attrs = MetricsExporter.to_opentelemetry_attributes(metrics)
                for key, value in attrs.items():
                    with contextlib.suppress(Exception):
                        span.set_attribute(f"{skill_name}.{key}", value)

    async def handle_tool_call_impl(
        self,
        exec_ctx: ToolCallExecutionContext,
    ) -> tuple[int, dict[str, Any], dict[str, Any], float]:
        """Execute one tool call round, including prepare/execute/hooks/trace outputs."""
        config = exec_ctx.config
        state = exec_ctx.state
        services = exec_ctx.services

        tool_start = time.perf_counter()
        if config.tool_loop_injected_tool_latency_seconds:
            await asyncio.sleep(config.tool_loop_injected_tool_latency_seconds)

        prepared = await self._runner._prepare_tool_call_execution(
            request=self._runner._build_tool_call_preparation_request(
                tool_call=state.tool_call,
                parsed_args=state.parsed_args,
                resolved_outputs=state.resolved_outputs,
                skills_by_name=services.skill_specs_by_name,
                tool_hooks=services.tool_call_hooks,
                allow_tool_replace=services.allow_tool_replacement,
                index=config.index,
                round_index_value=config.round_index_value,
                parallel_group_id=state.parallel_group_id,
            )
        )
        if isinstance(prepared, tuple):
            return prepared

        tool_name = prepared.tool_name
        tool_call_id = prepared.tool_call_id
        args = prepared.args
        skill = prepared.skill
        requested_tool_name = prepared.requested_tool_name
        cache_key = prepared.cache_key

        executed = await self._runner._run_prepared_tool_call(
            prepared=prepared,
            config=exec_ctx.config,
            state=exec_ctx.state,
            services=services,
            tool_start=tool_start,
        )

        result = executed.result
        cache_hit_for_reporting = executed.cache_hit_for_reporting
        tool_elapsed = executed.tool_elapsed
        latency_ms = executed.latency_ms

        if tool_name:
            services.tool_loop_invoked_tool_names.add(tool_name)
        self._update_resolved_outputs(
            resolved_outputs=state.resolved_outputs,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            index=config.index,
            result=result,
        )

        await self._emit_tool_usage_outcome(
            tool_hooks=services.tool_call_hooks,
            hook_context=prepared.hook_context,
            result=result,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            requested_tool_name=requested_tool_name,
            parallel_group_id=state.parallel_group_id,
            cache_hit_for_reporting=cache_hit_for_reporting,
            cache_key=cache_key,
            latency_ms=latency_ms,
        )

        if tool_name:
            await self._runner._trigger_post_tool_use_hook(tool_name, args, result, skill)

        trace_entry, tool_message = self._build_tool_trace_and_message(
            tool_name=tool_name,
            requested_tool_name=requested_tool_name,
            tool_call_id=tool_call_id,
            round_index_value=config.round_index_value,
            parallel_group_id=state.parallel_group_id,
            duration_ms=tool_elapsed * 1000 if tool_elapsed > 0 else None,
            args=args,
            result=result,
            attempted_tool_name=prepared.attempted_tool_name,
            allow_tool_replace=services.allow_tool_replacement,
            tool_result_summary_enabled=config.tool_loop_result_summary_enabled,
            tool_result_summary_max_chars=config.tool_loop_result_summary_max_chars,
            tool_result_summary_max_items=config.tool_loop_result_summary_max_items,
        )

        return config.index, trace_entry, tool_message, tool_elapsed

    async def _execute_tool_call(
        self,
        tool_name: str | None,
        args: dict[str, Any],
        skills_by_name: dict[str, SkillSpec],
        executor: Any,
        tool_call_id: str | None,
    ) -> dict[str, Any]:
        return await execute_tool_skill(
            ToolSkillExecutionRequest(
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
            ),
            ToolSkillExecutionServices(
                skill_specs_by_name=skills_by_name,
                tool_executor=executor,
                record_metrics=self._record_metrics,
            ),
        )
