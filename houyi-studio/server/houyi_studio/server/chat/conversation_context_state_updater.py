from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .types import Conversation, ConversationContextState, Message

ContextStateUpdateMode = Literal["append", "release", "recompute"]


@dataclass(frozen=True)
class ContextStateUpdateRequest:
    """Describes one conversation-context state transition.

    The chat server updates rolling context state from multiple sources.
    This request object keeps the transition contract explicit so append,
    compaction, and rewrite flows all reuse the same update path.
    """

    mode: ContextStateUpdateMode
    reason: str
    model: str
    messages: list[Message] | None = None
    released_units: int = 0
    compacted_at: float | None = None
    compaction_delta: int | None = None
    compacted_message_count: int | None = None


@dataclass(frozen=True)
class ContextStateUpdateResult:
    state: ConversationContextState
    event_payload: dict[str, Any]


class ConversationContextStateUpdater:
    """Applies rolling-context updates and builds the matching UI event payload."""

    request_cls = ContextStateUpdateRequest

    def __init__(self, *, conversation_context: Any) -> None:
        self._conversation_context = conversation_context

    def apply(
        self,
        *,
        conversation: Conversation,
        request: ContextStateUpdateRequest,
    ) -> ContextStateUpdateResult:
        # The core state transition lives here so every caller emits the same
        # authoritative conversation_context_state payload to the UI.
        if request.mode == "append":
            next_state = self._conversation_context.apply_appended_messages(
                conversation,
                messages=list(request.messages or []),
                model=request.model,
            )
            source = "append_delta"
        elif request.mode == "release":
            next_state = self._conversation_context.apply_delta(
                conversation,
                released_units=max(0, int(request.released_units)),
                compacted_at=request.compacted_at,
                compaction_delta=request.compaction_delta,
                compacted_message_count=request.compacted_message_count,
            )
            source = "release_delta"
        else:
            next_state = self._conversation_context.recover_state(
                conversation,
                model=request.model,
            )
            conversation.conversation_context_state = next_state
            source = "recompute"

        return ContextStateUpdateResult(
            state=next_state,
            event_payload=self.build_event_payload(
                conversation_id=conversation.conversation_id,
                state=next_state,
                source=source,
                reason=request.reason,
            ),
        )

    @staticmethod
    def build_event_payload(
        *,
        conversation_id: str,
        state: ConversationContextState,
        source: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "conversation_context_state": state.model_dump(mode="json"),
            "source": source,
            "reason": reason,
        }
