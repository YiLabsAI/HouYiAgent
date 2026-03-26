from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from houyi.infrastructure.observability import Span

from .types import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistResult:
    persisted: bool
    context_state_event: dict[str, Any] | None = None


class AssistantTurnPersistence:
    """Persists the assistant turn and applies conversation state updates."""

    def __init__(self, *, json_store: Any, context_state_updater: Any) -> None:
        self._json_store = json_store
        self._context_state_updater = context_state_updater

    async def persist(
        self,
        *,
        conversation_id: str,
        conv_lock: Any,
        assistant_msg: Message,
        content_parts: list[str],
        reasoning_parts: list[str],
        persisted_tool_messages: list[Message],
        usage_payload: dict[str, Any] | None,
        finish_reason: str | None,
        budget_metadata: dict[str, Any] | None,
        generation_metadata: dict[str, Any],
        completion_emitted_at: float | None,
        chat_span: Span,
        model: str,
    ) -> PersistResult:
        assistant_msg.content = "".join(content_parts)
        if reasoning_parts:
            assistant_msg.reasoning_content = "".join(reasoning_parts)

        if not (assistant_msg.content or assistant_msg.reasoning_content):
            chat_span.set_status("error", "LLM returned no content")
            logger.warning(
                "Chat response empty (LLM error): conversation=%s, message=%s — not persisted",
                conversation_id,
                assistant_msg.message_id,
            )
            return PersistResult(persisted=False)

        context_state_event: dict[str, Any] | None = None
        async with conv_lock:
            if isinstance(usage_payload, dict) and usage_payload:
                assistant_msg.metadata["usage"] = usage_payload
            if isinstance(finish_reason, str) and finish_reason:
                assistant_msg.metadata["finish_reason"] = finish_reason
            if isinstance(budget_metadata, dict) and budget_metadata:
                assistant_msg.metadata["budget"] = budget_metadata
            assistant_msg.metadata["trace_id"] = chat_span.trace_id

            conversation = self._json_store.get(conversation_id)
            if conversation is not None:
                if completion_emitted_at is not None:
                    generation_metadata["post_stream_persist_ms"] = round(
                        (time.perf_counter() - completion_emitted_at) * 1000,
                        2,
                    )
                assistant_msg.metadata.update(generation_metadata)
                if persisted_tool_messages:
                    conversation.messages.extend(persisted_tool_messages)
                conversation.messages.append(assistant_msg)
                conversation.updated_at = time.time()
                # Persisted assistant/tool messages are the authoritative append
                # point for rolling context state during the main chat stream.
                update_result = self._context_state_updater.apply(
                    conversation=conversation,
                    request=self._context_state_updater.request_cls(
                        mode="append",
                        reason="assistant_persist",
                        model=model,
                        messages=[*persisted_tool_messages, assistant_msg],
                    ),
                )
                context_state_event = update_result.event_payload
                self._json_store.update(conversation)

        chat_span.set_attribute("chat.response_content_len", len(assistant_msg.content))
        chat_span.set_status("ok")
        logger.info(
            "Chat response complete: conversation=%s, message=%s, content_len=%d",
            conversation_id,
            assistant_msg.message_id,
            len(assistant_msg.content),
        )
        return PersistResult(persisted=True, context_state_event=context_state_event)
