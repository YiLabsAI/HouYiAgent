from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from houyi.application.context.compaction_evaluator import CompactionEvaluator
from houyi.application.context.compaction_policy import (
    build_prune_only_summary,
    partition_messages_for_compaction,
)
from houyi.application.context.compaction_summary import build_compaction_summary
from houyi.application.context.compaction_triggers import (
    build_trigger_payload,
    estimate_post_compaction_utilization,
    resolve_compaction_trigger,
    resolve_utilization,
    resolve_watermarks,
)
from houyi.application.context.context_lifecycle import (
    ContextLifecycleHookService as ChatContextHookService,
)
from houyi.application.context.context_lifecycle import (
    stage_span,
)
from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import CompactionRecord

from .types import Conversation, Message, MessageRole


@dataclass(slots=True)
class CompactionOutcome:
    conversation_snapshot: Conversation
    compaction_event: dict[str, Any] | None = None
    context_state_event: dict[str, Any] | None = None
    record: CompactionRecord | None = None


@dataclass(slots=True)
class SummaryBuildResult:
    text: str
    model: str | None = None
    latency_ms: float | None = None
    mode: str = "heuristic"


@dataclass(slots=True)
class _LegacyContextStateRequest:
    mode: str
    reason: str
    model: str | None = None
    released_units: int = 0
    compacted_at: float | None = None
    compaction_delta: int | None = None
    compacted_message_count: int | None = None


class _LegacyContextStateUpdater:
    request_cls = _LegacyContextStateRequest

    def __init__(self, *, apply_conversation_context_delta: Any) -> None:
        self._apply_conversation_context_delta = apply_conversation_context_delta

    def apply(self, *, conversation: Any, request: _LegacyContextStateRequest) -> Any:
        if request.mode == "release":
            self._apply_conversation_context_delta(
                conversation,
                released_units=request.released_units,
                compacted_at=request.compacted_at,
                compaction_delta=request.compaction_delta,
                compacted_message_count=request.compacted_message_count,
            )
        return type("_LegacyUpdateResult", (), {"event_payload": None})()


class ContextCompressor:
    def __init__(
        self,
        *,
        json_store: Any,
        is_vision_model: Callable[[str | None], bool],
        context_state_updater: Any | None = None,
        apply_conversation_context_delta: Any | None = None,
        repo_intent_detector: Callable[[str], bool] | None = None,
        summary_builder: Callable[..., Any] | None = None,
        hook_service: ChatContextHookService | None = None,
        repo_recent_window: int = 6,
        low_watermark: float = 0.6,
        cooldown_seconds: float = 30.0,
        pressure_threshold: float = 0.7,
        overflow_threshold: float = 0.9,
    ) -> None:
        self._json_store = json_store
        self._is_vision_model = is_vision_model
        if context_state_updater is None:
            if apply_conversation_context_delta is None:
                raise TypeError("context_state_updater is required")
            self._context_state_updater = _LegacyContextStateUpdater(
                apply_conversation_context_delta=apply_conversation_context_delta,
            )
        else:
            self._context_state_updater = context_state_updater
        self._repo_intent_detector = repo_intent_detector or _looks_like_repo_intent
        self._summary_builder = summary_builder or build_compaction_summary
        self._hook_service = hook_service or ChatContextHookService()
        self._repo_recent_window = max(2, int(repo_recent_window))
        self._low_watermark = max(0.0, float(low_watermark))
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._pressure_threshold = max(0.0, float(pressure_threshold))
        self._overflow_threshold = max(self._pressure_threshold, float(overflow_threshold))

    async def compact_for_send(
        self,
        *,
        conversation_id: str,
        conversation_snapshot: Conversation,
        model: str,
        user_content: str,
        conv_lock: Any,
        chat_span: Any,
        trigger_kind: str | None = None,
        recent_window: int | None = None,
        low_watermark: float | None = None,
        pressure_threshold: float | None = None,
        overflow_threshold: float | None = None,
        cooldown_messages: int | None = None,
        cooldown_seconds: float | None = None,
    ) -> CompactionOutcome:
        effective_recent_window = (
            max(2, int(recent_window)) if recent_window is not None else self._repo_recent_window
        )
        (
            effective_low_watermark,
            effective_pressure_threshold,
            effective_overflow_threshold,
        ) = resolve_watermarks(
            default_low_watermark=self._low_watermark,
            default_pressure_threshold=self._pressure_threshold,
            default_overflow_threshold=self._overflow_threshold,
            low_watermark=low_watermark,
            pressure_threshold=pressure_threshold,
            overflow_threshold=overflow_threshold,
        )
        effective_cooldown_messages = (
            max(0, int(cooldown_messages)) if cooldown_messages is not None else 0
        )
        effective_cooldown_seconds = (
            max(0.0, float(cooldown_seconds))
            if cooldown_seconds is not None
            else self._cooldown_seconds
        )
        trigger = self._resolve_trigger(
            conversation=conversation_snapshot,
            model=model,
            user_content=user_content,
            trigger_kind=trigger_kind,
            low_watermark=effective_low_watermark,
            pressure_threshold=effective_pressure_threshold,
            overflow_threshold=effective_overflow_threshold,
        )
        if trigger is None:
            return CompactionOutcome(conversation_snapshot=conversation_snapshot)

        with stage_span(
            chat_span,
            "context.compaction",
            attributes={
                "chat.conversation_id": conversation_id,
                "chat.compaction.trigger": trigger["trigger"],
                "chat.compaction.low_watermark": trigger["low_watermark"],
                "chat.compaction.high_watermark": trigger["high_watermark"],
                "chat.compaction.critical_watermark": trigger["critical_watermark"],
            },
        ) as compaction_span:
            active_parent = compaction_span or chat_span
            active_span = _MirrorSpan(primary=active_parent, secondary=chat_span)

            now = time.time()
            safety = self._evaluate_safety_gate(
                conversation_snapshot=conversation_snapshot,
                now=now,
                recent_window=effective_recent_window,
                cooldown_messages=effective_cooldown_messages,
                cooldown_seconds=effective_cooldown_seconds,
            )
            if safety == "cooldown_active" and str(trigger.get("trigger") or "") in {
                "pre_request_pressure",
                "overflow_recovery",
            }:
                safety = None
            if safety is not None:
                active_span.set_attribute("chat.compaction.triggered", False)
                self._apply_trigger_attrs(active_span, trigger)
                active_span.set_attribute("chat.compaction.safety_gate", safety)
                return CompactionOutcome(conversation_snapshot=conversation_snapshot)

            original_messages = list(conversation_snapshot.messages)
            kept_messages, dropped_messages = partition_messages_for_compaction(
                original_messages,
                protected_message_ids={
                    message.message_id
                    for message in original_messages
                    if isinstance(message.message_id, str)
                    and message.message_id
                    and (
                        message.role == MessageRole.SYSTEM
                        or message.message_id
                        in self._collect_active_pin_source_ids(conversation_snapshot)
                    )
                },
                recent_window=effective_recent_window,
            )
            if not dropped_messages:
                return CompactionOutcome(conversation_snapshot=conversation_snapshot)

            trigger = self._hook_service.run_before_compress(trigger, span=active_span)

            try:
                with stage_span(active_parent, "context.compaction.backup") as backup_span:
                    backup_entry = self._json_store.create_backup(
                        conversation_id,
                        trigger=trigger["trigger"],
                        metadata={
                            "kind": "conversation_snapshot",
                            "reason": trigger["reason"],
                            "state_machine": ["prune", "summarize", "evaluate", "commit"],
                        },
                    )
                    _MirrorSpan(
                        primary=backup_span or active_parent, secondary=chat_span
                    ).set_attribute(
                        "chat.compaction.backup_id",
                        backup_entry["backup_id"],
                    )
            except Exception as exc:
                self._hook_service.run_compress_error(
                    stage="backup",
                    error=exc,
                    span=active_span,
                )
                active_span.set_attribute("chat.compaction.triggered", False)
                self._apply_trigger_attrs(active_span, trigger)
                active_span.set_attribute("chat.compaction.abort_reason", "backup_failed")
                active_span.set_attribute("chat.compaction.backup_error", str(exc))
                return CompactionOutcome(conversation_snapshot=conversation_snapshot)

            next_snapshot = conversation_snapshot.model_copy(deep=True)
            next_snapshot.messages = kept_messages
            estimator = TokenEstimator(model=model)
            evaluator = CompactionEvaluator(estimator)
            before_messages = [
                message.to_llm_message(vision=self._is_vision_model(model))
                for message in dropped_messages
            ]
            source_message_ids = [
                message.message_id
                for message in dropped_messages
                if isinstance(message.message_id, str)
            ]
            protected_message_ids = [
                message.message_id
                for message in kept_messages
                if isinstance(message.message_id, str) and message.message_id
            ]
            fallback_prune_only = False
            summary_result = SummaryBuildResult(text="")
            with stage_span(active_parent, "llm.summarize") as summarize_span:
                summarize_target = _MirrorSpan(
                    primary=summarize_span or active_parent,
                    secondary=chat_span,
                )
                try:
                    summary_result = await self._run_summary_builder(
                        dropped_messages,
                        model=model,
                        chat_span=summarize_target,
                    )
                    summary_text = summary_result.text
                except Exception as exc:
                    fallback_prune_only = True
                    summary_text = build_prune_only_summary(dropped_messages)
                    summary_result = SummaryBuildResult(
                        text=summary_text, mode="fallback_prune_only"
                    )
                    summarize_target.set_attribute(
                        "chat.compaction.summarize_fallback",
                        "prune_only",
                    )
                    summarize_target.set_attribute(
                        "chat.compaction.summarize_error",
                        str(exc),
                    )
                summarize_target.set_attribute(
                    "chat.compaction.summary_source_messages",
                    len(source_message_ids),
                )
                if summary_result.model:
                    summarize_target.set_attribute(
                        "chat.compaction.summary_model",
                        summary_result.model,
                    )
                if summary_result.latency_ms is not None:
                    summarize_target.set_attribute(
                        "chat.compaction.summary_latency_ms",
                        round(float(summary_result.latency_ms), 2),
                    )
            with stage_span(active_parent, "compaction.evaluate") as evaluate_span:
                record = evaluator.evaluate(
                    before_messages=before_messages,
                    summary=summary_text,
                    source_message_ids=source_message_ids,
                    pinned_message_ids=self._collect_pinned_message_ids(
                        conversation=conversation_snapshot,
                        messages=kept_messages,
                    ),
                    trigger=trigger["trigger"],
                    metadata={
                        "kind": "history_trim",
                        "reason": trigger["reason"],
                        "kept_recent_messages": len(kept_messages),
                        "dropped_messages": len(dropped_messages),
                        "utilization_ratio": trigger["utilization_ratio"],
                        "utilization_source": trigger["utilization_source"],
                        "state_machine": ["prune", "summarize", "evaluate", "commit"],
                        "safety_gate": "passed",
                        "summarization_mode": (
                            "fallback_prune_only" if fallback_prune_only else summary_result.mode
                        ),
                        "summary_model": summary_result.model,
                        "summary_latency_ms": (
                            round(float(summary_result.latency_ms), 2)
                            if summary_result.latency_ms is not None
                            else None
                        ),
                    },
                )
                _MirrorSpan(
                    primary=evaluate_span or active_parent,
                    secondary=chat_span,
                ).set_attribute(
                    "chat.compaction.compression_ratio",
                    record.metrics.compression_ratio,
                )
            record.pressure_level = trigger["pressure_level"]
            record.backup_id = str(backup_entry["backup_id"])
            record.pruned_block_ids = list(source_message_ids)
            record.summarized_block_ids = list(source_message_ids)
            record.protected_block_ids = list(protected_message_ids)
            record.active_turn_protected = True
            record.cooldown_applied = False
            record.restore_status = "ready"
            record.metadata["low_watermark"] = trigger["low_watermark"]
            record.metadata["high_watermark"] = trigger["high_watermark"]
            record.metadata["critical_watermark"] = trigger["critical_watermark"]
            record = self._hook_service.run_after_compress(record, span=active_span)

            released_units = max(
                0,
                int(record.metrics.tokens_before or 0) - int(record.metrics.tokens_after or 0),
            )
            post_utilization_ratio = self._estimate_post_compaction_utilization(
                conversation=conversation_snapshot,
                model=model,
                released_units=released_units,
                trigger=trigger,
            )
            record.metadata["post_compaction_utilization_ratio"] = post_utilization_ratio
            record.metadata["target_low_watermark_met"] = post_utilization_ratio <= float(
                trigger["low_watermark"]
            )
            context_state_event: dict[str, Any] | None = None
            try:
                with stage_span(active_parent, "context.compaction.commit") as commit_span:
                    self._json_store.attach_backup_record(
                        record.backup_id, record_id=record.compaction_id
                    )
                    async with conv_lock:
                        conversation = self._json_store.get(conversation_id)
                        if conversation is not None:
                            history = conversation.metadata.get("compaction_history")
                            if not isinstance(history, list):
                                history = []
                            history.append(record.model_dump(mode="json"))
                            conversation.metadata["compaction_history"] = history[-20:]
                            conversation.updated_at = time.time()
                            update_result = self._context_state_updater.apply(
                                conversation=conversation,
                                request=self._context_state_updater.request_cls(
                                    mode="release",
                                    reason="compaction_commit",
                                    model=model,
                                    released_units=released_units,
                                    compacted_at=record.created_at,
                                    compaction_delta=released_units,
                                    compacted_message_count=len(conversation.messages),
                                ),
                            )
                            context_state_event = update_result.event_payload
                            self._json_store.update(conversation)
                    commit_target = _MirrorSpan(
                        primary=commit_span or active_parent,
                        secondary=chat_span,
                    )
                    commit_target.set_attribute(
                        "chat.compaction.blocks_dropped_count",
                        len(record.pruned_block_ids),
                    )
                    commit_target.set_attribute(
                        "chat.compaction.blocks_summarized_count",
                        len(record.summarized_block_ids),
                    )
            except Exception as exc:
                self._hook_service.run_compress_error(
                    stage="commit",
                    error=exc,
                    span=active_span,
                )
                record.restore_status = "restored_after_commit_failure"
                self._json_store.restore_backup(str(record.backup_id))
                active_span.set_attribute("chat.compaction.triggered", False)
                active_span.set_attribute("chat.compaction.trigger", record.trigger)
                active_span.set_attribute("chat.compaction.pressure_level", record.pressure_level)
                active_span.set_attribute("chat.compaction.restore_status", record.restore_status)
                active_span.set_attribute("chat.compaction.commit_error", str(exc))
                return CompactionOutcome(conversation_snapshot=conversation_snapshot)

            self._apply_chat_span(chat_span=active_span, record=record)
            return CompactionOutcome(
                conversation_snapshot=next_snapshot,
                compaction_event={"compaction": record.model_dump(mode="json")},
                context_state_event=context_state_event,
                record=record,
            )

    async def _run_summary_builder(
        self,
        messages: list[Message],
        *,
        model: str,
        chat_span: Any,
    ) -> SummaryBuildResult:
        kwargs: dict[str, Any] = {}
        try:
            parameters = inspect.signature(self._summary_builder).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "model" in parameters:
            kwargs["model"] = model
        if "chat_span" in parameters:
            kwargs["chat_span"] = chat_span
        result = self._summary_builder(messages, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, SummaryBuildResult):
            return result
        return SummaryBuildResult(text=str(result or ""))

    def _resolve_trigger(
        self,
        *,
        conversation: Conversation,
        model: str,
        user_content: str,
        trigger_kind: str | None = None,
        low_watermark: float | None = None,
        pressure_threshold: float | None = None,
        overflow_threshold: float | None = None,
    ) -> dict[str, str | float] | None:
        non_system_messages = [
            message.to_llm_message(vision=self._is_vision_model(model))
            for message in conversation.messages
            if message.role != MessageRole.SYSTEM
        ]
        utilization, utilization_source = self._resolve_utilization(
            conversation=conversation,
            model=model,
            non_system_messages=non_system_messages,
        )
        if self._should_force_repo_compaction(
            user_content=user_content,
            message_count=len(non_system_messages),
            utilization_ratio=utilization,
            utilization_source=utilization_source,
            pressure_threshold=pressure_threshold,
        ):
            return build_trigger_payload(
                trigger="pre_request_pressure",
                pressure_level="elevated",
                reason="repo_intent_recent_window",
                utilization_ratio=utilization,
                utilization_source="repo_intent_override",
                low_watermark=low_watermark or self._low_watermark,
                high_watermark=pressure_threshold or self._pressure_threshold,
                critical_watermark=overflow_threshold or self._overflow_threshold,
            )
        return resolve_compaction_trigger(
            trigger_kind=trigger_kind,
            message_count=len(non_system_messages),
            utilization_ratio=utilization,
            utilization_source=utilization_source,
            default_low_watermark=self._low_watermark,
            default_pressure_threshold=self._pressure_threshold,
            default_overflow_threshold=self._overflow_threshold,
            low_watermark=low_watermark,
            pressure_threshold=pressure_threshold,
            overflow_threshold=overflow_threshold,
        )

    def _should_force_repo_compaction(
        self,
        *,
        user_content: str,
        message_count: int,
        utilization_ratio: float,
        utilization_source: str,
        pressure_threshold: float | None,
    ) -> bool:
        # Repo/code intent acts as a pressure multiplier rather than a blind
        # override. When no persisted context state is available yet, we keep a
        # conservative repo-intent fallback so first compaction still works.
        if not self._repo_intent_detector(user_content):
            return False
        if message_count < self._repo_recent_window:
            return False
        threshold = (
            float(pressure_threshold)
            if pressure_threshold is not None
            else float(self._pressure_threshold)
        )
        if utilization_source == "token_estimate":
            return True
        if utilization_ratio <= 0.0:
            return True
        return utilization_ratio >= max(0.5, threshold)

    @staticmethod
    def _apply_trigger_attrs(chat_span: Any, trigger: dict[str, str | float]) -> None:
        chat_span.set_attribute("chat.compaction.trigger", trigger["trigger"])
        chat_span.set_attribute("chat.compaction.pressure_level", trigger["pressure_level"])
        chat_span.set_attribute("chat.compaction.utilization_ratio", trigger["utilization_ratio"])
        chat_span.set_attribute("chat.compaction.utilization_source", trigger["utilization_source"])
        chat_span.set_attribute("chat.compaction.low_watermark", trigger["low_watermark"])
        chat_span.set_attribute("chat.compaction.high_watermark", trigger["high_watermark"])
        chat_span.set_attribute("chat.compaction.critical_watermark", trigger["critical_watermark"])

    def _estimate_post_compaction_utilization(
        self,
        *,
        conversation: Conversation,
        model: str,
        released_units: int,
        trigger: dict[str, str | float],
    ) -> float:
        state = conversation.conversation_context_state
        non_system_messages = [
            message.to_llm_message(vision=self._is_vision_model(model))
            for message in conversation.messages
            if message.role != MessageRole.SYSTEM
        ]
        estimator = TokenEstimator(model=model)
        return estimate_post_compaction_utilization(
            used_units=getattr(state, "used_units", None),
            max_units=getattr(state, "max_units", None),
            current_tokens=estimator.count_messages(non_system_messages),
            max_input_tokens=max(1, int(estimator.max_input_tokens or 1)),
            released_units=released_units,
        )

    def _resolve_utilization(
        self,
        *,
        conversation: Conversation,
        model: str,
        non_system_messages: list[dict[str, Any]],
    ) -> tuple[float, str]:
        state = conversation.conversation_context_state
        estimator = TokenEstimator(model=model)
        return resolve_utilization(
            used_units=getattr(state, "used_units", None),
            max_units=getattr(state, "max_units", None),
            current_tokens=estimator.count_messages(non_system_messages),
            max_input_tokens=max(1, int(estimator.max_input_tokens or 1)),
        )

    def _evaluate_safety_gate(
        self,
        *,
        conversation_snapshot: Conversation,
        now: float,
        recent_window: int,
        cooldown_messages: int,
        cooldown_seconds: float,
    ) -> str | None:
        if len(conversation_snapshot.messages) <= recent_window:
            return "insufficient_history"
        active_streaming_state = conversation_snapshot.active_streaming_state
        if (
            active_streaming_state is not None
            and str(getattr(active_streaming_state, "status", "")) == "streaming"
            and isinstance(getattr(active_streaming_state, "message_id", None), str)
            and active_streaming_state.message_id
        ):
            return "active_streaming"
        recent_messages = conversation_snapshot.messages[-recent_window:]
        incomplete_turn_gate = self._detect_incomplete_turn_gate(recent_messages)
        if incomplete_turn_gate is not None:
            return incomplete_turn_gate
        state = conversation_snapshot.conversation_context_state
        last_compacted_message_count = getattr(state, "last_compacted_message_count", None)
        if (
            cooldown_messages > 0
            and isinstance(last_compacted_message_count, int)
            and last_compacted_message_count >= 0
            and (len(conversation_snapshot.messages) - last_compacted_message_count)
            < cooldown_messages
        ):
            return "cooldown_active"
        last_compacted_at = getattr(state, "last_compacted_at", None)
        if last_compacted_at is not None and (now - float(last_compacted_at)) < cooldown_seconds:
            return "cooldown_active"
        return None

    def _detect_incomplete_turn_gate(self, messages: list[Message]) -> str | None:
        pending_tool_call_ids: set[str] = set()
        pending_seen = False
        for message in messages:
            if message.role == MessageRole.ASSISTANT:
                if pending_tool_call_ids:
                    return "split_incomplete_turn"
                tool_calls = message.tool_calls if isinstance(message.tool_calls, list) else None
                if tool_calls:
                    extracted_ids = {
                        str(call.get("id"))
                        for call in tool_calls
                        if isinstance(call, dict)
                        and isinstance(call.get("id"), str)
                        and call.get("id")
                    }
                    if extracted_ids:
                        pending_tool_call_ids = extracted_ids
                        pending_seen = False
                continue
            if message.role == MessageRole.TOOL:
                tool_call_id = (
                    str(message.tool_call_id)
                    if isinstance(message.tool_call_id, str) and message.tool_call_id
                    else ""
                )
                if not pending_tool_call_ids or not tool_call_id:
                    return "split_incomplete_turn"
                if tool_call_id not in pending_tool_call_ids:
                    return "split_incomplete_turn"
                pending_tool_call_ids.discard(tool_call_id)
                pending_seen = True
                continue
            if pending_tool_call_ids:
                return "active_tool_loop"
        if pending_tool_call_ids:
            return "active_tool_loop"
        if pending_seen:
            return None
        return None

    def _collect_active_pin_source_ids(self, conversation: Conversation) -> set[str]:
        metadata = conversation.metadata if isinstance(conversation.metadata, dict) else {}
        raw = metadata.get("pinned_contexts")
        if not isinstance(raw, list):
            return set()
        pinned_ids: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "") != "active":
                continue
            source_message_id = item.get("source_message_id")
            if isinstance(source_message_id, str) and source_message_id:
                pinned_ids.add(source_message_id)
        return pinned_ids

    def _collect_pinned_message_ids(
        self,
        *,
        conversation: Conversation,
        messages: list[Message],
    ) -> list[str]:
        active_pin_source_ids = self._collect_active_pin_source_ids(conversation)
        pinned: list[str] = []
        for message in messages:
            if not isinstance(message.message_id, str) or not message.message_id:
                continue
            if message.message_id in active_pin_source_ids:
                pinned.append(message.message_id)
        return pinned

    def _partition_messages_for_compaction(
        self,
        conversation: Conversation,
        messages: list[Message],
        *,
        recent_window: int,
    ) -> tuple[list[Message], list[Message]]:
        active_pin_source_ids = self._collect_active_pin_source_ids(conversation)
        protected_ids = {
            message.message_id
            for message in messages
            if isinstance(message.message_id, str)
            and message.message_id
            and (message.role == MessageRole.SYSTEM or message.message_id in active_pin_source_ids)
        }
        recent_ids = {
            message.message_id
            for message in messages[-recent_window:]
            if isinstance(message.message_id, str) and message.message_id
        }
        kept: list[Message] = []
        dropped: list[Message] = []
        for message in messages:
            message_id = message.message_id if isinstance(message.message_id, str) else ""
            if message_id and (message_id in protected_ids or message_id in recent_ids):
                kept.append(message)
            else:
                dropped.append(message)
        return kept, dropped

    def _apply_chat_span(self, *, chat_span: Any, record: CompactionRecord) -> None:
        chat_span.set_attribute("chat.compaction.triggered", True)
        chat_span.set_attribute("chat.compaction.trigger", record.trigger)
        chat_span.set_attribute("chat.compaction.pressure_level", record.pressure_level)
        utilization_ratio = (
            record.metadata.get("utilization_ratio") if isinstance(record.metadata, dict) else None
        )
        utilization_source = (
            record.metadata.get("utilization_source") if isinstance(record.metadata, dict) else None
        )
        low_watermark = (
            record.metadata.get("low_watermark") if isinstance(record.metadata, dict) else None
        )
        high_watermark = (
            record.metadata.get("high_watermark") if isinstance(record.metadata, dict) else None
        )
        critical_watermark = (
            record.metadata.get("critical_watermark") if isinstance(record.metadata, dict) else None
        )
        post_compaction_utilization = (
            record.metadata.get("post_compaction_utilization_ratio")
            if isinstance(record.metadata, dict)
            else None
        )
        target_low_watermark_met = (
            record.metadata.get("target_low_watermark_met")
            if isinstance(record.metadata, dict)
            else None
        )
        if isinstance(utilization_ratio, (int, float)):
            chat_span.set_attribute("chat.compaction.utilization_ratio", float(utilization_ratio))
        if isinstance(utilization_source, str) and utilization_source:
            chat_span.set_attribute("chat.compaction.utilization_source", utilization_source)
        if isinstance(low_watermark, (int, float)):
            chat_span.set_attribute("chat.compaction.low_watermark", float(low_watermark))
        if isinstance(high_watermark, (int, float)):
            chat_span.set_attribute("chat.compaction.high_watermark", float(high_watermark))
        if isinstance(critical_watermark, (int, float)):
            chat_span.set_attribute("chat.compaction.critical_watermark", float(critical_watermark))
        if isinstance(post_compaction_utilization, (int, float)):
            chat_span.set_attribute(
                "chat.compaction.post_utilization_ratio", float(post_compaction_utilization)
            )
        if isinstance(target_low_watermark_met, bool):
            chat_span.set_attribute(
                "chat.compaction.target_low_watermark_met",
                target_low_watermark_met,
            )
        chat_span.set_attribute("chat.compaction.backup_id", record.backup_id)
        chat_span.set_attribute(
            "chat.compaction.messages_compacted", record.metrics.messages_compacted
        )
        chat_span.set_attribute("chat.compaction.tokens_before", record.metrics.tokens_before)
        chat_span.set_attribute("chat.compaction.tokens_after", record.metrics.tokens_after)
        chat_span.set_attribute(
            "chat.compaction.pin_violation_count", record.metrics.pin_violation_count
        )
        chat_span.set_attribute("chat.compaction.protected_block_ids", record.protected_block_ids)
        chat_span.set_attribute("chat.compaction.restore_status", record.restore_status)
        chat_span.set_attribute(
            "chat.compaction.blocks_dropped_count", len(record.pruned_block_ids)
        )
        chat_span.set_attribute(
            "chat.compaction.blocks_summarized_count",
            len(record.summarized_block_ids),
        )
        chat_span.set_attribute(
            "chat.compaction.recent_messages_kept",
            len(record.protected_block_ids),
        )

    def _build_prune_only_summary(self, messages: list[Message]) -> str:
        if not messages:
            return ""
        first = messages[0]
        last = messages[-1]
        first_id = first.message_id if isinstance(first.message_id, str) else "unknown"
        last_id = last.message_id if isinstance(last.message_id, str) else first_id
        return (
            f"Pruned {len(messages)} earlier messages to recover context budget. "
            f"Range: {first_id}..{last_id}."
        )


class _MirrorSpan:
    def __init__(self, *, primary: Any, secondary: Any | None = None) -> None:
        self._primary = primary
        self._secondary = secondary

    def set_attribute(self, key: str, value: Any) -> None:
        self._primary.set_attribute(key, value)
        if self._secondary is not None and self._secondary is not self._primary:
            self._secondary.set_attribute(key, value)


def _looks_like_repo_intent(user_content: str) -> bool:
    lowered = user_content.lower()
    if "github.com/" in lowered or "gitlab.com/" in lowered:
        return True
    return any(
        token in lowered for token in ("repo", "repository", "readme", "codebase", "project")
    )


def _build_compaction_summary(messages: list[Message]) -> str:
    return build_compaction_summary(messages)
