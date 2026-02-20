"""Chat Service: orchestrates SDK components for chat interactions.

Bridges Studio Server with SDK Context Engine, Memory Engine, and LLM Adapter.
Owns the business logic for sending messages, managing context, and streaming.

"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from houyi.context import ContextPlanner, ContextRenderer, TokenEstimator
from houyi.llm.base import LLMAdapter
from houyi.llm.factory import LLMAdapterFactory, _create_vertex_adapter
from houyi.llm.models import DEFAULT_MODEL
from houyi.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.memory import MemoryStore
from houyi.observability.context import TraceContext
from houyi.observability.trace_manager import Span
from houyi.observability.types import SpanType

from .json_store import JsonStore
from .sse_adapter import stream_chat_sse
from .types import EditMessageRequest, Message, MessageRole, SendMessageRequest

logger = logging.getLogger(__name__)

# Vision-capable model patterns.
# Models matching these patterns support image_url in content arrays.
_VISION_PATTERNS = [
    r"gemini-[12345]\.\d",
    r"gemini-(?:flash|pro|flash-lite)",
    r"claude-3",
    r"claude-(?:haiku|sonnet|opus)-4",
    r"gpt-4o",
    r"gpt-4-turbo",
    r"gpt-4\.1",
    r"gpt-4\.5",
    r"gpt-5",
    r"o[134]-",
    r"qwen-vl",
    r"qwen2-vl",
    r"qwen2\.5-vl",
    r"qwen-omni",
    r"deepseek-vl",
    r"glm-4v",
    r"internvl",
    r"pixtral",
    r"llava",
    r"minicpm",
    r"grok-vision",
    r"vision",
]
_VISION_RE = re.compile("|".join(_VISION_PATTERNS), re.IGNORECASE)


def is_vision_model(model_id: str) -> bool:
    """Return True if *model_id* is known to accept image_url content parts."""
    return bool(_VISION_RE.search(model_id))


class ChatService:
    """Orchestrates chat interactions between UI, Server, and SDK layers.

    Responsibilities:
    - Receive user message → persist → build context → call LLM → stream response
    - Manage conversation lifecycle (create, update, delete)
    - Track context usage for UI display

    Concurrency: Uses per-conversation locking via JsonStore.lock() to
    serialize read-modify-write cycles. Safe for concurrent async callers.
    """

    def __init__(
        self,
        json_store: JsonStore,
        memory_store: MemoryStore | None = None,
        default_model: str = "",
        default_system_instructions: str = "",
        settings_store: Any | None = None,
    ):
        """Initialize chat service.

        Args:
            json_store: Conversation persistence store.
            memory_store: Optional memory store for context injection.
            default_model: Default LLM model name.
            default_system_instructions: Default system prompt.
            settings_store: Optional SettingsStore for provider-based model routing.
        """
        self.json_store = json_store
        self.memory_store = memory_store
        self.default_model = default_model or DEFAULT_MODEL
        self.default_system_instructions = default_system_instructions
        self._settings_store = settings_store
        self._default_adapter = LLMAdapterFactory.create()
        self._adapter_cache: dict[str, LLMAdapter] = {}

    def invalidate_adapter_cache(self) -> None:
        """Clear cached adapters. Call when provider settings change."""
        self._adapter_cache.clear()
        logger.info("LLM adapter cache invalidated")

    def _get_adapter_for_model(self, model: str) -> LLMAdapter:
        """Get the LLM adapter for a given model by looking up its provider.

        Routes the model to the correct provider's adapter based on settings.
        Falls back to the default adapter if no provider match is found.
        """
        if not self._settings_store:
            return self._default_adapter

        settings = self._settings_store.get()
        for provider in settings.providers:
            if not provider.enabled:
                continue
            if model in provider.models:
                cache_key = provider.id
                if cache_key in self._adapter_cache:
                    return self._adapter_cache[cache_key]

                # Detect if this is a Vertex AI / Gemini provider
                provider_url = (provider.base_url or "").rstrip("/")
                is_vertex = (
                    "aiplatform.googleapis.com" in provider_url
                    or "vertex" in provider.name.lower()
                    or provider.id.startswith("vertex")
                )
                if is_vertex:
                    adapter = _create_vertex_adapter()
                    logger.info(
                        "Model '%s' routed to Gemini adapter (provider='%s')", model, provider.name
                    )
                elif provider_url and "/v1" in provider_url:
                    adapter = SiliconFlowAdapter(
                        api_key=provider.api_key or None,
                        base_url=provider_url,
                        default_model=model,
                    )
                else:
                    # No usable base_url, fall back to default
                    logger.info(
                        "Model '%s' provider '%s' has no OpenAI-compatible base_url, using default adapter",
                        model,
                        provider.name,
                    )
                    return self._default_adapter
                self._adapter_cache[cache_key] = adapter
                logger.info(
                    "Model '%s' routed to provider '%s' (%s)",
                    model,
                    provider.name,
                    provider.base_url or "default",
                )
                return adapter

        return self._default_adapter

    def get_context_usage(self, conversation_id: str) -> dict[str, Any] | None:
        """Calculate context usage for a conversation without sending a message.

        Returns the same usage dict that the SSE context.usage event provides,
        or None if the conversation has no messages.
        """
        conversation = self.json_store.get(conversation_id)
        if conversation is None:
            return None

        model = conversation.model or self.default_model
        sys_instructions = conversation.system_instructions or self.default_system_instructions

        history_messages = [
            m.to_llm_message() for m in conversation.messages if m.role != MessageRole.SYSTEM
        ]
        if not history_messages:
            return None

        estimator = TokenEstimator(model=model)
        planner = ContextPlanner(
            token_estimator=estimator,
            system_instructions=sys_instructions,
        )

        memory_text = ""
        if self.memory_store:
            from houyi.memory.types import MemoryScope

            memory_text = self.memory_store.as_context_text(MemoryScope.SESSION)

        plan = planner.plan(
            messages=history_messages,
            system_instructions=sys_instructions,
            memory_context=memory_text if memory_text else None,
        )

        return plan.usage.model_dump(mode="json")

    async def send_message(
        self,
        conversation_id: str,
        request: SendMessageRequest,
    ) -> AsyncIterator[str]:
        """Send a user message and stream the assistant response as SSE.

        Flow:
        1. Load conversation from store
        2. Append user message
        3. Build context plan (TokenEstimator → ContextPlanner → ContextRenderer)
        4. Call LLM with rendered messages
        5. Stream response as SSE events
        6. On completion, persist assistant message

        Args:
            conversation_id: Target conversation.
            request: User message and optional overrides.

        Yields:
            SSE-encoded event strings.

        Raises:
            FileNotFoundError: If conversation does not exist.
        """
        # --- Observability: create chat.request root span ---
        chat_span = Span(
            name="chat.request",
            parent=TraceContext.current(),
            span_type=SpanType.NODE,
            attributes={
                "chat.conversation_id": conversation_id,
                "chat.user_content_len": len(request.content),
            },
        )
        chat_token = TraceContext.push(chat_span)

        try:
            # --- Critical section 1: load + append user msg + build context ---
            conv_lock = await self.json_store.lock(conversation_id)
            async with conv_lock:
                # 1. Load conversation
                conversation = self.json_store.get(conversation_id)
                if conversation is None:
                    raise FileNotFoundError(f"Conversation {conversation_id} not found")

                model = (
                    (request.model if request.model else None)
                    or (conversation.model if conversation.model else None)
                    or self.default_model
                )
                sys_instructions = (
                    conversation.system_instructions or self.default_system_instructions
                )
                chat_span.set_attribute("chat.model", model)

                # 2. Append user message (with attachments if any)
                user_msg = Message(
                    role=MessageRole.USER,
                    content=request.content,
                    attachments=request.attachments,
                )
                conversation.messages.append(user_msg)
                conversation.updated_at = time.time()
                self.json_store.update(conversation)

                # 3. Build context plan
                estimator = TokenEstimator(model=model)
                planner = ContextPlanner(
                    token_estimator=estimator,
                    system_instructions=sys_instructions,
                )
                renderer = ContextRenderer()

                # Gather memory context (Phase 1: simple text)
                memory_text = ""
                if self.memory_store:
                    from houyi.memory.types import MemoryScope

                    memory_text = self.memory_store.as_context_text(MemoryScope.SESSION)

                # Build message list for planner (exclude system — planner handles it)
                _vision = is_vision_model(model)
                history_messages = [
                    m.to_llm_message(vision=_vision)
                    for m in conversation.messages
                    if m.role != MessageRole.SYSTEM
                ]

                plan = planner.plan(
                    messages=history_messages,
                    system_instructions=sys_instructions,
                    memory_context=memory_text if memory_text else None,
                )

                llm_messages = renderer.render(plan)

                # Context usage for SSE event
                context_usage = plan.usage.model_dump(mode="json")
                chat_span.set_attribute("chat.context_tokens_used", plan.usage.used_tokens)
                chat_span.set_attribute("chat.context_tokens_max", plan.usage.max_context_tokens)
                chat_span.set_attribute("chat.llm_messages_count", len(llm_messages))

                logger.info(
                    "Chat context: %d messages, %d tokens used / %d max (%s)",
                    len(llm_messages),
                    plan.usage.used_tokens,
                    plan.usage.max_context_tokens,
                    model,
                )
            # --- End critical section 1 (lock released before streaming) ---

            # 4. Create assistant message placeholder
            assistant_msg = Message(role=MessageRole.ASSISTANT, content="")

            # 5. Stream LLM response
            # Priority: request params > conversation params > global defaults
            llm_kwargs: dict[str, Any] = {}
            if request.temperature is not None:
                llm_kwargs["temperature"] = request.temperature
            elif conversation.temperature is not None:
                llm_kwargs["temperature"] = conversation.temperature
            if request.max_tokens is not None:
                llm_kwargs["max_tokens"] = request.max_tokens
            elif conversation.max_tokens is not None:
                llm_kwargs["max_tokens"] = conversation.max_tokens
            if conversation.top_p is not None:
                llm_kwargs["top_p"] = conversation.top_p
            if request.enable_reasoning:
                llm_kwargs["enable_reasoning"] = True

            # --- Observability: create llm.call child span ---
            llm_span = Span(
                name="llm.call",
                parent=chat_span,
                span_type=SpanType.LLM,
                model=model,
                attributes={
                    "llm.model": model,
                    "llm.message_count": len(llm_messages),
                },
            )

            llm_adapter = self._get_adapter_for_model(model)
            llm_stream = llm_adapter.stream_chat(
                messages=llm_messages,
                model=model,
                **llm_kwargs,
            )

            # Wrap with content accumulator for persistence
            content_parts: list[str] = []
            reasoning_parts: list[str] = []

            async def accumulating_stream() -> AsyncIterator[tuple[str, str | None]]:
                async for content_delta, reasoning_delta in llm_stream:
                    if content_delta:
                        content_parts.append(content_delta)
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                    yield content_delta, reasoning_delta

            # 6. Stream SSE events
            async for sse_chunk in stream_chat_sse(
                llm_stream=accumulating_stream(),
                message_id=assistant_msg.message_id,
                model=model,
                context_usage=context_usage,
            ):
                yield sse_chunk

            # End llm.call span with token usage
            if hasattr(llm_adapter, "last_usage") and llm_adapter.last_usage:
                usage = llm_adapter.last_usage
                llm_span.set_tokens(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
            llm_span.set_status("ok")
            llm_span.end()

            # --- Critical section 2: persist assistant message ---
            # Only persist if we actually received content (skip on LLM errors
            # that produced zero content — avoids empty ghost messages).
            assistant_msg.content = "".join(content_parts)
            if reasoning_parts:
                assistant_msg.reasoning_content = "".join(reasoning_parts)

            if assistant_msg.content or assistant_msg.reasoning_content:
                async with conv_lock:
                    # Capture token usage from adapter
                    if hasattr(llm_adapter, "last_usage") and llm_adapter.last_usage:
                        assistant_msg.metadata["usage"] = llm_adapter.last_usage

                    # Re-load to avoid overwriting concurrent changes
                    conversation = self.json_store.get(conversation_id)
                    if conversation is not None:
                        conversation.messages.append(assistant_msg)
                        conversation.updated_at = time.time()
                        self.json_store.update(conversation)

                chat_span.set_attribute("chat.response_content_len", len(assistant_msg.content))
                chat_span.set_status("ok")

                logger.info(
                    "Chat response complete: conversation=%s, message=%s, content_len=%d",
                    conversation_id,
                    assistant_msg.message_id,
                    len(assistant_msg.content),
                )
            else:
                chat_span.set_status("error", "LLM returned no content")
                logger.warning(
                    "Chat response empty (LLM error): conversation=%s, message=%s — not persisted",
                    conversation_id,
                    assistant_msg.message_id,
                )

        except Exception as e:
            chat_span.set_status("error", str(e))
            raise
        finally:
            chat_span.end()
            # GeneratorExit during async streaming may cause cleanup
            # in a different asyncio Context, making ContextVar.reset()
            # fail. This is safe to ignore — the span is already ended.
            with contextlib.suppress(ValueError):
                TraceContext.pop(chat_token)

    async def edit_message(
        self,
        conversation_id: str,
        message_id: str,
        request: EditMessageRequest,
    ) -> Message:
        """Edit a message's content.

        Only user messages can be edited. Updates the message in place
        and persists the conversation.

        Args:
            conversation_id: Target conversation.
            message_id: Message to edit.
            request: New content.

        Returns:
            The updated Message.

        Raises:
            FileNotFoundError: If conversation not found.
            ValueError: If message not found or not a user message.
        """
        conv_lock = await self.json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self.json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            msg = next((m for m in conversation.messages if m.message_id == message_id), None)
            if msg is None:
                raise ValueError(f"Message {message_id} not found")
            if msg.role != MessageRole.USER:
                raise ValueError("Only user messages can be edited")

            msg.content = request.content
            msg.metadata["edited"] = True
            msg.metadata["edited_at"] = time.time()
            conversation.updated_at = time.time()
            self.json_store.update(conversation)
            return msg

    async def delete_message(
        self,
        conversation_id: str,
        message_id: str,
    ) -> None:
        """Delete a single message from a conversation.

        Args:
            conversation_id: Target conversation.
            message_id: Message to delete.

        Raises:
            FileNotFoundError: If conversation not found.
            ValueError: If message not found.
        """
        conv_lock = await self.json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self.json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            original_len = len(conversation.messages)
            conversation.messages = [m for m in conversation.messages if m.message_id != message_id]
            if len(conversation.messages) == original_len:
                raise ValueError(f"Message {message_id} not found")

            conversation.updated_at = time.time()
            self.json_store.update(conversation)

    async def regenerate_message(
        self,
        conversation_id: str,
        message_id: str,
    ) -> AsyncIterator[str]:
        """Regenerate an assistant message.

        Removes the target assistant message and all subsequent messages,
        then re-sends the last user message to get a fresh response.

        Args:
            conversation_id: Target conversation.
            message_id: Assistant message to regenerate.

        Yields:
            SSE-encoded event strings (same as send_message).

        Raises:
            FileNotFoundError: If conversation not found.
            ValueError: If message not found or not an assistant message.
        """
        conv_lock = await self.json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self.json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            # Find the target message index
            msg_idx = next(
                (i for i, m in enumerate(conversation.messages) if m.message_id == message_id),
                None,
            )
            if msg_idx is None:
                raise ValueError(f"Message {message_id} not found")
            if conversation.messages[msg_idx].role != MessageRole.ASSISTANT:
                raise ValueError("Only assistant messages can be regenerated")

            # Find the last user message before this assistant message
            last_user_content = None
            for i in range(msg_idx - 1, -1, -1):
                if conversation.messages[i].role == MessageRole.USER:
                    last_user_content = conversation.messages[i].content
                    break

            if last_user_content is None:
                raise ValueError("No user message found before the assistant message")

            # Remove the assistant message and everything after it
            conversation.messages = conversation.messages[:msg_idx]
            conversation.updated_at = time.time()
            self.json_store.update(conversation)

        # Re-send the last user message (send_message will append it again)
        # We need to remove the last user message too since send_message will re-add it
        conv_lock2 = await self.json_store.lock(conversation_id)
        async with conv_lock2:
            conversation = self.json_store.get(conversation_id)
            if (
                conversation
                and conversation.messages
                and conversation.messages[-1].role == MessageRole.USER
            ):
                conversation.messages.pop()
                conversation.updated_at = time.time()
                self.json_store.update(conversation)

        request = SendMessageRequest(content=last_user_content)
        async for chunk in self.send_message(conversation_id, request):
            yield chunk
