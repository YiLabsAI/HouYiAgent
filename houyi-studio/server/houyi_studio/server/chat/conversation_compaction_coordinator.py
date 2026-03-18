from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from houyi.application.context.compaction_summary import build_compaction_summary

from .context_compressor import ContextCompressor, SummaryBuildResult
from .json_store import JsonStore
from .types import Message

logger = logging.getLogger(__name__)


class _NullHookSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


class _CaptureSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class ConversationCompactionCoordinator:
    def __init__(
        self,
        *,
        json_store: JsonStore,
        default_model: str,
        is_vision_model: Any,
        apply_conversation_context_delta: Any,
        repo_intent_detector: Any,
        hook_service: Any,
        get_adapter_for_model: Any,
        resolve_summary_model: Any,
        background_tasks: set[asyncio.Task[Any]],
    ) -> None:
        self._json_store = json_store
        self._default_model = default_model
        self._get_adapter_for_model = get_adapter_for_model
        self._resolve_summary_model = resolve_summary_model
        self._background_tasks = background_tasks
        self._context_compressor = ContextCompressor(
            json_store=json_store,
            is_vision_model=is_vision_model,
            apply_conversation_context_delta=apply_conversation_context_delta,
            repo_intent_detector=repo_intent_detector,
            summary_builder=self.build_compaction_summary,
            hook_service=hook_service,
        )

    @property
    def context_compressor(self) -> ContextCompressor:
        return self._context_compressor

    async def build_compaction_summary(
        self,
        messages: list[Message],
        *,
        model: str,
        chat_span: Any | None = None,
    ) -> SummaryBuildResult:
        heuristic = build_compaction_summary(messages)
        summary_model = self._resolve_summary_model(model)
        if not summary_model:
            return SummaryBuildResult(text=heuristic, mode="heuristic")
        adapter = self._get_adapter_for_model(summary_model)
        started_at = time.perf_counter()
        try:
            response = await adapter.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize earlier conversation turns for future context reuse. "
                            "Retain decisions, constraints, key findings, and unresolved questions. "
                            "Be concise and factual."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Compacted messages: {len(messages)}\n\n{heuristic}",
                    },
                ],
                temperature=0.1,
                max_tokens=256,
                model=summary_model,
            )
        except Exception as exc:
            logger.warning(
                "Summary model failed; using heuristic summary: model=%s error=%s",
                summary_model,
                exc,
            )
            return SummaryBuildResult(text=heuristic, mode="heuristic_fallback")
        latency_ms = (time.perf_counter() - started_at) * 1000
        summary_text = str(getattr(response, "content", "") or "").strip()
        if not summary_text:
            return SummaryBuildResult(text=heuristic, mode="heuristic_fallback")
        if chat_span is not None:
            chat_span.set_attribute("chat.compaction.summary_model", summary_model)
            chat_span.set_attribute("chat.compaction.summary_latency_ms", round(latency_ms, 2))
        return SummaryBuildResult(
            text=summary_text,
            model=summary_model,
            latency_ms=latency_ms,
            mode="llm",
        )

    async def run_post_turn_compaction(
        self,
        *,
        conversation_id: str,
        model: str,
    ) -> None:
        conv_lock = await self._json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                return
            snapshot = conversation.model_copy(deep=True)
            resolved_model = conversation.model or model or self._default_model
        await self._context_compressor.compact_for_send(
            conversation_id=conversation_id,
            conversation_snapshot=snapshot,
            model=resolved_model,
            user_content="",
            conv_lock=conv_lock,
            chat_span=_NullHookSpan(),
            trigger_kind="post_turn_background",
        )

    def schedule_post_turn_compaction(
        self,
        *,
        conversation_id: str,
        model: str,
    ) -> None:
        async def _runner() -> None:
            try:
                await self.run_post_turn_compaction(
                    conversation_id=conversation_id,
                    model=model,
                )
            except Exception as exc:
                logger.warning(
                    "Post-turn compaction failed: conversation=%s error=%s",
                    conversation_id,
                    exc,
                )

        task = asyncio.create_task(_runner())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def compact_conversation(self, conversation_id: str) -> dict[str, Any]:
        conv_lock = await self._json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")
            model = conversation.model or self._default_model
            snapshot = conversation.model_copy(deep=True)
        span = _CaptureSpan()
        outcome = await self._context_compressor.compact_for_send(
            conversation_id=conversation_id,
            conversation_snapshot=snapshot,
            model=model,
            user_content="",
            conv_lock=conv_lock,
            chat_span=span,
            trigger_kind="manual",
        )
        return {
            "conversation_id": conversation_id,
            "applied": outcome.record is not None,
            "compaction": outcome.record.model_dump(mode="json")
            if outcome.record is not None
            else None,
            "safety_gate": span.attributes.get("chat.compaction.safety_gate"),
        }
