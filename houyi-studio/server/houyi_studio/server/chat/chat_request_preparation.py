from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from houyi.infrastructure.observability import Span

from .types import Message, MessageRole, SendMessageRequest


@dataclass
class PreparedSendContext:
    conv_lock: Any
    model: str
    llm_messages: list[dict[str, Any]]
    context_usage: dict[str, Any]
    llm_kwargs: dict[str, Any]
    runtime_profile: Any
    budget_metadata: dict[str, Any] | None = None
    context_state_event: dict[str, Any] | None = None
    compaction_event: dict[str, Any] | None = None
    compaction_state_event: dict[str, Any] | None = None


class ChatRequestPreparation:
    """Builds the persisted and runtime inputs required for one chat exchange."""

    def __init__(
        self,
        *,
        json_store: Any,
        default_model: str,
        default_system_instructions: str,
        conversation_context: Any,
        context_state_updater: Any,
        resolve_llm_kwargs: Callable[..., tuple[dict[str, Any], dict[str, Any] | None]],
        resolve_runtime_profile: Callable[[SendMessageRequest], Any],
        context_compressor: Any,
        build_context_messages: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        self._json_store = json_store
        self._default_model = default_model
        self._default_system_instructions = default_system_instructions
        self._conversation_context = conversation_context
        self._context_state_updater = context_state_updater
        self._resolve_llm_kwargs = resolve_llm_kwargs
        self._resolve_runtime_profile = resolve_runtime_profile
        self._context_compressor = context_compressor
        self._build_context_messages = build_context_messages

    async def prepare(
        self,
        *,
        conversation_id: str,
        request: SendMessageRequest,
        chat_span: Span,
    ) -> PreparedSendContext:
        conv_lock = await self._json_store.lock(conversation_id)
        conversation_snapshot: Any | None = None
        context_state_event: dict[str, Any] | None = None
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            model = (
                (request.model if request.model else None)
                or (conversation.model if conversation.model else None)
                or self._default_model
            )
            sys_instructions = conversation.system_instructions or self._default_system_instructions
            chat_span.set_attribute("chat.model", model)

            user_msg = Message(
                role=MessageRole.USER,
                content=request.content,
                attachments=request.attachments,
            )
            user_input_tokens = self._conversation_context.estimate_units(user_msg, model=model)
            user_msg.metadata["usage"] = {
                "input_tokens": user_input_tokens,
                "prompt_tokens": user_input_tokens,
                "total_tokens": user_input_tokens,
            }
            conversation.messages.append(user_msg)
            conversation.updated_at = time.time()
            update_result = self._context_state_updater.apply(
                conversation=conversation,
                request=self._context_state_updater.request_cls(
                    mode="append",
                    reason="user_append",
                    model=model,
                    messages=[user_msg],
                ),
            )
            context_state_event = update_result.event_payload
            self._json_store.update(conversation)
            llm_kwargs, budget_metadata = self._resolve_llm_kwargs(
                model=model,
                request=request,
                conversation=conversation,
            )
            conversation_snapshot = conversation.model_copy(deep=True)

        if conversation_snapshot is None:
            raise RuntimeError(f"Conversation snapshot unavailable: {conversation_id}")

        runtime_profile = self._resolve_runtime_profile(request)
        chat_span.set_attribute("chat.runtime_profile", runtime_profile.name)
        chat_span.set_attribute("chat.context.keep_n", runtime_profile.keep_n or 0)
        chat_span.set_attribute("chat.context.low_watermark", runtime_profile.low_watermark)
        chat_span.set_attribute(
            "chat.context.compression_threshold", runtime_profile.compression_threshold
        )
        chat_span.set_attribute(
            "chat.context.overflow_threshold", runtime_profile.overflow_threshold
        )
        chat_span.set_attribute("chat.context.cooldown_messages", runtime_profile.cooldown_messages)
        chat_span.set_attribute(
            "chat.tool_result.max_tokens", runtime_profile.tool_result_max_tokens or 0
        )

        compaction_outcome = await self._context_compressor.compact_for_send(
            conversation_id=conversation_id,
            conversation_snapshot=conversation_snapshot,
            model=model,
            user_content=str(request.content or "").strip(),
            conv_lock=conv_lock,
            chat_span=chat_span,
            recent_window=runtime_profile.keep_n,
            low_watermark=runtime_profile.low_watermark,
            pressure_threshold=runtime_profile.compression_threshold,
            overflow_threshold=runtime_profile.overflow_threshold,
            cooldown_messages=runtime_profile.cooldown_messages,
            cooldown_seconds=runtime_profile.cooldown_seconds,
        )
        conversation_snapshot = compaction_outcome.conversation_snapshot
        compaction_event = compaction_outcome.compaction_event
        compaction_state_event = compaction_outcome.context_state_event

        llm_messages, context_usage = self._build_context_messages(
            conversation=conversation_snapshot,
            model=model,
            sys_instructions=sys_instructions,
            span=chat_span,
            input_budget=(
                int(budget_metadata["input_budget"])
                if isinstance(budget_metadata, dict)
                and budget_metadata.get("input_budget") is not None
                else None
            ),
        )

        return PreparedSendContext(
            conv_lock=conv_lock,
            model=model,
            llm_messages=llm_messages,
            context_usage=context_usage,
            llm_kwargs=llm_kwargs,
            runtime_profile=runtime_profile,
            budget_metadata=budget_metadata,
            context_state_event=context_state_event,
            compaction_event=compaction_event,
            compaction_state_event=compaction_state_event,
        )
