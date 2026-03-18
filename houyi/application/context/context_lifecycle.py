from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from houyi.application.context.types import CompactionRecord
from houyi.infrastructure.observability import Span, SpanType


@contextlib.contextmanager
def stage_span(parent: Any, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    if not isinstance(parent, Span):
        yield None
        return
    span = Span(
        name=name,
        parent=parent,
        span_type=SpanType.INTERNAL,
        attributes=attributes or {},
    )
    try:
        yield span
    except Exception as exc:
        span.set_status("error", str(exc))
        raise
    else:
        span.set_status("ok")
    finally:
        span.end()


class ContextLifecycleHookService:
    def __init__(
        self,
        *,
        hooks: list[Any] | None = None,
        phase_timeouts: dict[str, float] | None = None,
        before_compress: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        after_compress: Callable[[CompactionRecord], CompactionRecord | dict[str, Any] | None]
        | None = None,
        on_compress_error: Callable[[dict[str, Any]], Any] | None = None,
        on_tool_result: Callable[[list[Any], list[dict[str, Any]] | None], list[Any] | None]
        | None = None,
        on_plan_assembled: Callable[[Any], Any] | None = None,
        on_render: Callable[[list[dict[str, Any]]], list[dict[str, Any]] | None] | None = None,
        on_memory_recall: Callable[[str], str | None] | None = None,
    ) -> None:
        self._before_compress = before_compress
        self._after_compress = after_compress
        self._on_compress_error = on_compress_error
        self._on_tool_result = on_tool_result
        self._on_plan_assembled = on_plan_assembled
        self._on_render = on_render
        self._on_memory_recall = on_memory_recall
        self._hooks = list(hooks or [])
        self._phase_timeouts = {
            key: float(value)
            for key, value in (phase_timeouts or {}).items()
            if value is not None and float(value) > 0
        }
        self._contracts: dict[str, dict[str, str]] = {
            "before_compress": {
                "boundary": "compaction",
                "mode": "mapping",
                "fallback": "keep_original",
                "called_attr": "chat.hooks.before_compress.called",
                "error_attr": "chat.hooks.before_compress.error",
            },
            "after_compress": {
                "boundary": "compaction",
                "mode": "record_update",
                "fallback": "keep_original",
                "called_attr": "chat.hooks.after_compress.called",
                "error_attr": "chat.hooks.after_compress.error",
            },
            "compress_error": {
                "boundary": "compaction",
                "mode": "effect",
                "fallback": "no_op",
                "called_attr": "chat.hooks.on_compress_error.called",
                "error_attr": "chat.hooks.on_compress_error.error",
            },
            "tool_result": {
                "boundary": "tool_loop",
                "mode": "list_replace",
                "fallback": "keep_original",
                "called_attr": "chat.hooks.on_tool_result.called",
                "error_attr": "chat.hooks.on_tool_result.error",
            },
            "plan_assembled": {
                "boundary": "context_runtime",
                "mode": "value_replace",
                "fallback": "keep_original",
                "called_attr": "chat.hooks.on_plan_assembled.called",
                "error_attr": "chat.hooks.on_plan_assembled.error",
            },
            "render": {
                "boundary": "context_runtime",
                "mode": "list_replace",
                "fallback": "keep_original",
                "called_attr": "chat.hooks.on_render.called",
                "error_attr": "chat.hooks.on_render.error",
            },
            "memory_recall": {
                "boundary": "context_runtime",
                "mode": "string_replace",
                "fallback": "keep_original",
                "called_attr": "chat.hooks.on_memory_recall.called",
                "error_attr": "chat.hooks.on_memory_recall.error",
            },
        }

    def describe_contract(self) -> dict[str, dict[str, str]]:
        return {
            phase: {
                key: value
                for key, value in contract.items()
                if key not in {"called_attr", "error_attr"}
            }
            for phase, contract in self._contracts.items()
        }

    def run_before_compress(self, payload: dict[str, Any], *, span: Any) -> dict[str, Any]:
        context = CompressionHookContext(
            phase="before_compress",
            payload=dict(payload),
            span=span,
        )
        return self._run_mapping_hook(
            phase="before_compress",
            handler=self._before_compress,
            payload=payload,
            context=context,
            span=span,
        )

    def run_after_compress(self, record: CompactionRecord, *, span: Any) -> CompactionRecord:
        context = CompressionHookContext(
            phase="after_compress",
            record=record,
            payload=record.model_dump(mode="json"),
            span=span,
        )
        return self._run_record_hook(
            phase="after_compress",
            handler=self._after_compress,
            record=record,
            context=context,
            span=span,
        )

    def run_compress_error(self, *, stage: str, error: Exception, span: Any) -> None:
        contract = self._contracts["compress_error"]
        handlers = self._resolve_phase_handlers("compress_error", self._on_compress_error)
        span.set_attribute(contract["called_attr"], bool(handlers))
        span.set_attribute("chat.compaction.error_stage", stage)
        if not handlers:
            return
        context = CompressionErrorHookContext(
            phase="compress_error",
            payload={"stage": stage, "error": str(error)},
            stage=stage,
            error=error,
            span=span,
        )
        for name, handler in handlers:
            try:
                started_at = time.perf_counter()
                result = (
                    handler({"stage": stage, "error": str(error)})
                    if name == "legacy"
                    else handler(context)
                )
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._record_hook_duration(span, "compress_error", name, duration_ms)
                if self._timed_out("compress_error", duration_ms, span):
                    continue
                if isinstance(result, EffectHookResult):
                    context.metadata.update(result.metadata)
            except Exception as exc:
                span.set_attribute(contract["error_attr"], str(exc))

    def run_tool_result(
        self,
        persisted_tool_messages: list[Any],
        tool_trace: list[dict[str, Any]] | None,
        *,
        span: Any,
    ) -> list[Any]:
        context = ToolResultHookContext(
            phase="tool_result",
            payload=persisted_tool_messages,
            persisted_tool_messages=persisted_tool_messages,
            tool_trace=tool_trace,
            span=span,
        )
        return self._run_list_hook(
            phase="tool_result",
            handler=self._on_tool_result,
            args=(persisted_tool_messages, tool_trace),
            fallback=persisted_tool_messages,
            context=context,
            span=span,
        )

    def run_plan_assembled(self, plan: Any, *, span: Any) -> Any:
        context = PlanHookContext(
            phase="plan_assembled",
            payload=plan,
            plan=plan,
            span=span,
        )
        return self._run_value_hook(
            phase="plan_assembled",
            handler=self._on_plan_assembled,
            payload=plan,
            fallback=plan,
            context=context,
            span=span,
        )

    def run_render(self, messages: list[dict[str, Any]], *, span: Any) -> list[dict[str, Any]]:
        context = RenderHookContext(
            phase="render",
            payload=messages,
            messages=messages,
            span=span,
        )
        return self._run_list_hook(
            phase="render",
            handler=self._on_render,
            args=(messages,),
            fallback=messages,
            context=context,
            span=span,
        )

    def run_memory_recall(self, memory_text: str, *, span: Any) -> str:
        context = MemoryRecallHookContext(
            phase="memory_recall",
            payload=memory_text,
            memory_text=memory_text,
            span=span,
        )
        return self._run_string_hook(
            phase="memory_recall",
            handler=self._on_memory_recall,
            payload=memory_text,
            fallback=memory_text,
            context=context,
            span=span,
        )

    def _run_mapping_hook(
        self,
        *,
        phase: str,
        handler: Callable[[dict[str, Any]], dict[str, Any] | None] | None,
        payload: dict[str, Any],
        context: CompressionHookContext,
        span: Any,
    ) -> dict[str, Any]:
        contract = self._contracts[phase]
        handlers = self._resolve_phase_handlers(phase, handler)
        if not handlers:
            return payload
        span.set_attribute(contract["called_attr"], True)
        current = dict(payload)
        for name, current_handler in handlers:
            try:
                started_at = time.perf_counter()
                result = current_handler(current) if name == "legacy" else current_handler(context)
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._record_hook_duration(span, phase, name, duration_ms)
                if self._timed_out(phase, duration_ms, span):
                    continue
            except Exception as exc:
                span.set_attribute(contract["error_attr"], str(exc))
                continue
            normalized = self._normalize_mapping_result(result, current)
            current = normalized
            context.payload = dict(current)
        return current

    def _run_record_hook(
        self,
        *,
        phase: str,
        handler: Callable[[CompactionRecord], CompactionRecord | dict[str, Any] | None] | None,
        record: CompactionRecord,
        context: CompressionHookContext,
        span: Any,
    ) -> CompactionRecord:
        contract = self._contracts[phase]
        handlers = self._resolve_phase_handlers(phase, handler)
        if not handlers:
            return record
        span.set_attribute(contract["called_attr"], True)
        current = record
        for name, current_handler in handlers:
            try:
                started_at = time.perf_counter()
                result = current_handler(current) if name == "legacy" else current_handler(context)
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._record_hook_duration(span, phase, name, duration_ms)
                if self._timed_out(phase, duration_ms, span):
                    continue
            except Exception as exc:
                span.set_attribute(contract["error_attr"], str(exc))
                continue
            current = self._normalize_record_result(result, current)
            context.record = current
            context.payload = current.model_dump(mode="json")
        return current

    def _run_list_hook(
        self,
        *,
        phase: str,
        handler: Callable[..., list[Any] | None] | None,
        args: tuple[Any, ...],
        fallback: list[Any],
        context: ToolResultHookContext | RenderHookContext,
        span: Any,
    ) -> list[Any]:
        contract = self._contracts[phase]
        handlers = self._resolve_phase_handlers(phase, handler)
        if not handlers:
            return fallback
        span.set_attribute(contract["called_attr"], True)
        current = list(fallback)
        for name, current_handler in handlers:
            try:
                started_at = time.perf_counter()
                result = current_handler(*args) if name == "legacy" else current_handler(context)
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._record_hook_duration(span, phase, name, duration_ms)
                if self._timed_out(phase, duration_ms, span):
                    continue
            except Exception as exc:
                span.set_attribute(contract["error_attr"], str(exc))
                continue
            normalized = self._normalize_list_result(result, current)
            current = normalized
            if isinstance(context, ToolResultHookContext):
                context.persisted_tool_messages = current
            else:
                context.messages = current
            context.payload = current
        return current

    def _run_value_hook(
        self,
        *,
        phase: str,
        handler: Callable[[Any], Any] | None,
        payload: Any,
        fallback: Any,
        context: PlanHookContext,
        span: Any,
    ) -> Any:
        contract = self._contracts[phase]
        handlers = self._resolve_phase_handlers(phase, handler)
        if not handlers:
            return fallback
        span.set_attribute(contract["called_attr"], True)
        current = fallback
        for name, current_handler in handlers:
            try:
                started_at = time.perf_counter()
                result = current_handler(payload) if name == "legacy" else current_handler(context)
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._record_hook_duration(span, phase, name, duration_ms)
                if self._timed_out(phase, duration_ms, span):
                    continue
            except Exception as exc:
                span.set_attribute(contract["error_attr"], str(exc))
                continue
            current = self._normalize_value_result(result, current)
            context.plan = current
            context.payload = current
            payload = current
        return current

    def _run_string_hook(
        self,
        *,
        phase: str,
        handler: Callable[[str], str | None] | None,
        payload: str,
        fallback: str,
        context: MemoryRecallHookContext,
        span: Any,
    ) -> str:
        contract = self._contracts[phase]
        handlers = self._resolve_phase_handlers(phase, handler)
        if not handlers:
            return fallback
        span.set_attribute(contract["called_attr"], True)
        current = fallback
        for name, current_handler in handlers:
            try:
                started_at = time.perf_counter()
                result = current_handler(payload) if name == "legacy" else current_handler(context)
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._record_hook_duration(span, phase, name, duration_ms)
                if self._timed_out(phase, duration_ms, span):
                    continue
            except Exception as exc:
                span.set_attribute(contract["error_attr"], str(exc))
                continue
            current = self._normalize_string_result(result, current)
            context.memory_text = current
            context.payload = current
            payload = current
        return current

    def _resolve_phase_handlers(
        self,
        phase: str,
        legacy_handler: Callable[..., Any] | None,
    ) -> list[tuple[str, Callable[..., Any]]]:
        handlers: list[tuple[int, int, str, Callable[..., Any]]] = []
        if legacy_handler is not None:
            handlers.append((0, 0, "legacy", legacy_handler))
        method_name = self._phase_method_name(phase)
        for index, hook in enumerate(self._hooks, start=1):
            current_handler = getattr(hook, method_name, None)
            if callable(current_handler):
                priority = int(getattr(hook, "priority", 100) or 100)
                hook_name = str(
                    getattr(hook, "name", hook.__class__.__name__) or hook.__class__.__name__
                )
                handlers.append((priority, index, hook_name, current_handler))
        handlers.sort(key=lambda item: (item[0], item[1]))
        return [(name, handler) for _, _, name, handler in handlers]

    @staticmethod
    def _phase_method_name(phase: str) -> str:
        return {
            "before_compress": "before_compress",
            "after_compress": "after_compress",
            "compress_error": "on_compress_error",
            "tool_result": "on_tool_result",
            "plan_assembled": "on_plan_assembled",
            "render": "on_render",
            "memory_recall": "on_memory_recall",
        }[phase]

    def _timed_out(self, phase: str, duration_ms: float, span: Any) -> bool:
        timeout_seconds = self._phase_timeouts.get(phase)
        if timeout_seconds is None:
            return False
        if duration_ms <= timeout_seconds * 1000:
            return False
        span.set_attribute(f"chat.hooks.{phase}.timeout", True)
        return True

    @staticmethod
    def _record_hook_duration(span: Any, phase: str, name: str, duration_ms: float) -> None:
        span.set_attribute(f"chat.hooks.{phase}.last_handler", name)
        span.set_attribute(f"chat.hooks.{phase}.duration_ms", round(duration_ms, 2))

    @staticmethod
    def _normalize_mapping_result(result: Any, fallback: dict[str, Any]) -> dict[str, Any]:
        if isinstance(result, MappingHookResult):
            if result.action != "continue":
                return fallback
            update = result.update or {}
            merged = dict(fallback)
            merged.update(update)
            return merged
        if isinstance(result, dict):
            merged = dict(fallback)
            merged.update(result)
            return merged
        return fallback

    @staticmethod
    def _normalize_record_result(result: Any, fallback: CompactionRecord) -> CompactionRecord:
        if isinstance(result, RecordHookResult):
            if result.action != "continue":
                return fallback
            if isinstance(result.record, CompactionRecord):
                return result.record
            if isinstance(result.update, dict):
                return fallback.model_copy(update=result.update)
            return fallback
        if isinstance(result, CompactionRecord):
            return result
        if isinstance(result, dict):
            return fallback.model_copy(update=result)
        return fallback

    @staticmethod
    def _normalize_list_result(result: Any, fallback: list[Any]) -> list[Any]:
        if isinstance(result, ListHookResult):
            if result.action != "continue":
                return fallback
            return result.items if isinstance(result.items, list) else fallback
        return result if isinstance(result, list) else fallback

    @staticmethod
    def _normalize_value_result(result: Any, fallback: Any) -> Any:
        if isinstance(result, ValueHookResult):
            if result.action != "continue":
                return fallback
            return fallback if result.value is None else result.value
        return fallback if result is None else result

    @staticmethod
    def _normalize_string_result(result: Any, fallback: str) -> str:
        if isinstance(result, StringHookResult):
            if result.action != "continue":
                return fallback
            return result.text if isinstance(result.text, str) else fallback
        return result if isinstance(result, str) else fallback


@dataclass(slots=True, kw_only=True)
class HookContext:
    phase: str
    payload: Any = None
    span: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class CompressionHookContext(HookContext):
    payload: dict[str, Any] | None = None
    record: CompactionRecord | None = None


@dataclass(slots=True, kw_only=True)
class CompressionErrorHookContext(HookContext):
    stage: str
    error: Exception


@dataclass(slots=True, kw_only=True)
class ToolResultHookContext(HookContext):
    persisted_tool_messages: list[Any]
    tool_trace: list[dict[str, Any]] | None


@dataclass(slots=True, kw_only=True)
class PlanHookContext(HookContext):
    plan: Any


@dataclass(slots=True, kw_only=True)
class RenderHookContext(HookContext):
    messages: list[dict[str, Any]]


@dataclass(slots=True, kw_only=True)
class MemoryRecallHookContext(HookContext):
    memory_text: str


@dataclass(slots=True, kw_only=True)
class HookResult:
    value: Any = None
    action: str = "continue"
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stop_pipeline: bool = False
    fallback_to_original: bool = False
    append_observation: str | None = None


@dataclass(slots=True, kw_only=True)
class MappingHookResult(HookResult):
    update: dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class RecordHookResult(HookResult):
    record: CompactionRecord | None = None
    update: dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class EffectHookResult(HookResult):
    pass


@dataclass(slots=True, kw_only=True)
class ListHookResult(HookResult):
    items: list[Any] | None = None


@dataclass(slots=True, kw_only=True)
class ValueHookResult(HookResult):
    pass


@dataclass(slots=True, kw_only=True)
class StringHookResult(HookResult):
    text: str | None = None
