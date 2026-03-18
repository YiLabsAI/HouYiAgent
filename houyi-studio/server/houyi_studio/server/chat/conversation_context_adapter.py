from __future__ import annotations

from collections.abc import Callable

from houyi.application.context.session_context import SessionContextStateManager
from houyi.application.context.session_context_resolver import SessionContextResolver

from .json_store import JsonStore
from .types import Conversation, ConversationContextState, Message, MessageRole


class ConversationContextAdapter:
    """Adapts conversation models to reusable session-context state logic."""

    def __init__(
        self,
        *,
        json_store: JsonStore,
        default_model: str,
        rolling_capacity: int,
        is_vision_model: Callable[[str], bool] | None = None,
    ) -> None:
        self._json_store = json_store
        self._default_model = default_model
        self._is_vision_model = is_vision_model or (lambda _model: False)
        self._state_manager = SessionContextStateManager(
            rolling_capacity=rolling_capacity,
        )
        self._state_resolver = SessionContextResolver(state_manager=self._state_manager)

    def build_initial_state(
        self,
        conversation_id: str,
        *,
        now: float | None = None,
    ) -> ConversationContextState:
        return ConversationContextState.from_session_state(
            self._state_resolver.build_initial_state(conversation_id, now=now),
            conversation_id=conversation_id,
        )

    def estimate_units(self, message: Message, *, model: str) -> int:
        if message.role == MessageRole.SYSTEM:
            return 0
        payload = self._message_payload(message, model=model)
        return self._state_resolver.estimate_units(payload, model=model)

    def recover_state(
        self,
        conversation: Conversation,
        *,
        model: str | None = None,
    ) -> ConversationContextState:
        resolved_model = model or conversation.model or self._default_model
        message_payloads = [
            self._message_payload(message, model=resolved_model)
            for message in conversation.messages
            if message.role != MessageRole.SYSTEM
        ]
        session_state = self._state_resolver.recover_state(
            session_id=conversation.conversation_id,
            message_payloads=message_payloads,
            model=resolved_model,
            previous_state=(
                conversation.conversation_context_state.to_session_state()
                if conversation.conversation_context_state is not None
                else None
            ),
            updated_at=conversation.updated_at,
        )
        return ConversationContextState.from_session_state(
            session_state,
            conversation_id=conversation.conversation_id,
        )

    def ensure_state(
        self,
        conversation: Conversation,
        *,
        model: str | None = None,
        persist: bool = False,
    ) -> ConversationContextState:
        state = conversation.conversation_context_state
        if state is None:
            state = self.recover_state(conversation, model=model)
            conversation.conversation_context_state = state
            if persist:
                self._json_store.update(conversation)
            return state

        normalized_state = self._state_resolver.normalize_state(
            session_id=conversation.conversation_id,
            state=state.to_session_state(),
            updated_at=state.updated_at or conversation.updated_at,
        )
        normalized_state = ConversationContextState.from_session_state(
            normalized_state,
            conversation_id=conversation.conversation_id,
        )
        conversation.conversation_context_state = normalized_state
        if persist and normalized_state != state:
            self._json_store.update(conversation)
        return normalized_state

    def apply_delta(
        self,
        conversation: Conversation,
        *,
        added_units: int = 0,
        released_units: int = 0,
        compacted_at: float | None = None,
        compaction_delta: int | None = None,
        compacted_message_count: int | None = None,
    ) -> ConversationContextState:
        state = self.ensure_state(conversation)
        next_state = self._state_resolver.apply_delta(
            state=state.to_session_state(),
            added_units=added_units,
            released_units=released_units,
            compacted_at=compacted_at,
            compaction_delta=compaction_delta,
            compacted_message_count=compacted_message_count,
        )
        next_state = ConversationContextState.from_session_state(
            next_state,
            conversation_id=conversation.conversation_id,
        )
        conversation.conversation_context_state = next_state
        return next_state

    def apply_appended_messages(
        self,
        conversation: Conversation,
        *,
        messages: list[Message],
        model: str,
    ) -> ConversationContextState:
        added_units = sum(self.estimate_units(message, model=model) for message in messages)
        return self.apply_delta(conversation, added_units=added_units)

    def _message_payload(self, message: Message, *, model: str) -> dict[str, object]:
        try:
            return message.to_llm_message(vision=self._is_vision_model(model))
        except Exception:
            return message.to_llm_message()
