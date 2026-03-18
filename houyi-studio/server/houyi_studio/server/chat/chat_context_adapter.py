from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from houyi.adapters.memory import MemoryStore
from houyi.application.context.context_lifecycle import (
    ContextLifecycleHookService as ChatContextHookService,
)
from houyi.application.context.context_request_builder import (
    ContextRequestBuilder,
    ContextRequestSourceInput,
)
from houyi.application.context.context_sources import build_history_message_payloads
from houyi.application.context.token_estimator import TokenEstimator

from .types import Conversation

logger = logging.getLogger(__name__)


class ChatContextAdapter:
    """Adapts chat-facing server objects to the request-context builder.

    Request-scoped context assembly / planning / rendering is owned by
    SDK-neutral objects. This adapter only bridges chat product models,
    memory recall, and span/log projection into that reusable flow.
    """

    def __init__(
        self,
        *,
        memory_store: MemoryStore | None,
        is_vision_model: Callable[[str], bool],
        sanitize_tool_loop_structure: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        hook_service: ChatContextHookService | None = None,
    ) -> None:
        self._hook_service = hook_service or ChatContextHookService()
        self._request_builder = ContextRequestBuilder(
            hook_service=self._hook_service,
            build_history_messages=lambda conversation, model: build_history_message_payloads(
                getattr(conversation, "messages", []) if conversation is not None else [],
                message_to_payload=lambda message: message.to_llm_message(
                    vision=is_vision_model(model)
                ),
            ),
            get_memory_text=lambda span: self._get_memory_text(
                memory_store=memory_store,
                span=span,
            ),
            sanitize_history_messages=sanitize_tool_loop_structure,
        )

    def build_context_messages(
        self,
        *,
        conversation: Any,
        model: str,
        sys_instructions: str,
        span: Any,
        input_budget: int | None = None,
        truncation_log_label: str | None = "chat_send",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        result = self._request_builder.build_from_source(
            ContextRequestSourceInput(
                source=conversation,
                model=model,
                system_instructions=sys_instructions,
                span=span,
                input_budget=input_budget,
                truncation_log_label=truncation_log_label,
            )
        )
        llm_messages = result.llm_messages
        context_usage = result.context_usage

        self._apply_span_usage(
            span, context_usage, len(llm_messages), self._context_window_for_model(model)
        )
        logger.info(
            "Chat context: %d messages, %d tokens used / %d max (%s)",
            len(llm_messages),
            context_usage.get("used_tokens", 0),
            context_usage.get("max_context_tokens", self._context_window_for_model(model)),
            model,
        )
        if (
            input_budget is not None
            and input_budget <= 0
            and getattr(result, "history_message_count", 0) > 0
        ):
            logger.warning(
                "Context input budget is zero; using latest message fallback: model=%s requested_input_budget=%d",
                model,
                input_budget,
            )
        return llm_messages, context_usage

    def get_context_usage(
        self,
        *,
        conversation: Conversation,
        model: str,
        sys_instructions: str,
        truncation_log_label: str | None = None,
    ) -> dict[str, Any] | None:
        return self._request_builder.get_usage_from_source(
            ContextRequestSourceInput(
                source=conversation,
                model=model,
                system_instructions=sys_instructions,
                span=_NullSpan(),
                truncation_log_label=truncation_log_label,
            )
        )

    def _apply_span_usage(
        self,
        span: Any,
        context_usage: dict[str, Any],
        llm_message_count: int,
        fallback_context_window: int,
    ) -> None:
        span.set_attribute("chat.context_tokens_used", context_usage.get("used_tokens", 0))
        span.set_attribute(
            "chat.context_tokens_max",
            context_usage.get("max_context_tokens", fallback_context_window),
        )
        planned_prompt_tokens = context_usage.get("planned_prompt_tokens")
        if planned_prompt_tokens is not None:
            span.set_attribute("chat.context_planned_prompt_tokens", planned_prompt_tokens)
        reserved_output_tokens = context_usage.get("reserved_output_tokens")
        if reserved_output_tokens is not None:
            span.set_attribute("chat.context_reserved_output_tokens", reserved_output_tokens)
        available_input_tokens = context_usage.get("available_input_tokens")
        if available_input_tokens is not None:
            span.set_attribute("chat.context_available_input_tokens", available_input_tokens)
        span.set_attribute("chat.llm_messages_count", llm_message_count)
        span.set_attribute("chat.context.rendered_message_count", llm_message_count)
        block_breakdown = context_usage.get("block_breakdown")
        if isinstance(block_breakdown, dict) and block_breakdown:
            span.set_attribute(
                "chat.context_blocks", json.dumps(block_breakdown, ensure_ascii=False)
            )
        dropped_blocks = context_usage.get("dropped_blocks")
        if isinstance(dropped_blocks, list) and dropped_blocks:
            span.set_attribute(
                "chat.context_dropped_blocks", json.dumps(dropped_blocks, ensure_ascii=False)
            )
        drop_reasons = context_usage.get("drop_reasons")
        if isinstance(drop_reasons, dict) and drop_reasons:
            span.set_attribute(
                "chat.context_drop_reasons", json.dumps(drop_reasons, ensure_ascii=False)
            )
        dropped_block_details = context_usage.get("dropped_block_details")
        if isinstance(dropped_block_details, list) and dropped_block_details:
            span.set_attribute(
                "chat.context_dropped_block_details",
                json.dumps(dropped_block_details, ensure_ascii=False),
            )

    def _context_window_for_model(self, model: str) -> int:
        return TokenEstimator(model=model).context_window

    def _get_memory_text(
        self,
        *,
        memory_store: MemoryStore | None,
        span: Any,
    ) -> str | None:
        if not memory_store:
            return None
        from houyi.adapters.memory.types import MemoryScope

        memory_text = memory_store.as_context_text(MemoryScope.SESSION)
        return self._hook_service.run_memory_recall(memory_text, span=span)


class _NullSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None
