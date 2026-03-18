from __future__ import annotations

import time
from typing import Any

from .json_store import JsonStore
from .types import ActiveStreamingState


class ConversationStreamingStateManager:
    def __init__(self, *, json_store: JsonStore) -> None:
        self._json_store = json_store

    async def set_active_streaming_state(
        self,
        *,
        conversation_id: str,
        conv_lock: Any,
        message_id: str,
        request_id: str,
        status: str = "streaming",
        started_at: float | None = None,
    ) -> None:
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                return
            existing = conversation.active_streaming_state
            conversation.active_streaming_state = ActiveStreamingState(
                conversation_id=conversation_id,
                message_id=message_id,
                request_id=request_id,
                status="finishing" if status == "finishing" else "streaming",
                started_at=(
                    existing.started_at
                    if isinstance(existing, ActiveStreamingState)
                    else (started_at if started_at is not None else time.time())
                ),
                updated_at=time.time(),
            )
            conversation.updated_at = time.time()
            self._json_store.update(conversation)

    async def clear_active_streaming_state(
        self,
        *,
        conversation_id: str,
        conv_lock: Any,
        message_id: str,
    ) -> None:
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                return
            current = conversation.active_streaming_state
            if not isinstance(current, ActiveStreamingState):
                return
            if current.message_id != message_id:
                return
            conversation.active_streaming_state = None
            conversation.updated_at = time.time()
            self._json_store.update(conversation)
