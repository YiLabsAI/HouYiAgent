from __future__ import annotations

import time
from typing import Any

from .types import Conversation, Message, MessageRole, PinnedContextRecord, PinStatus


class PinnedContextStore:
    def __init__(self, *, json_store: Any) -> None:
        self._json_store = json_store

    async def pin_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        replace_pin_id: str | None = None,
        title: str | None = None,
    ) -> PinnedContextRecord:
        lock = await self._json_store.lock(conversation_id)
        async with lock:
            conversation = self._require_conversation(conversation_id)
            message = self._require_message(conversation, message_id)
            pins = self._read_pins(conversation)
            if replace_pin_id:
                pins = self._transition_pin(pins, replace_pin_id, PinStatus.SUPERSEDED)
            record = self._build_record(
                conversation_id=conversation_id,
                message=message,
                title=title,
            )
            pins.append(record)
            self._write_pins(conversation, pins)
            self._json_store.update(conversation)
            return record

    async def list_pins(
        self,
        *,
        conversation_id: str,
        include_inactive: bool = False,
    ) -> list[PinnedContextRecord]:
        conversation = self._require_conversation(conversation_id)
        pins = self._read_pins(conversation)
        if include_inactive:
            return pins
        return [pin for pin in pins if pin.status == PinStatus.ACTIVE]

    async def update_pin_status(
        self,
        *,
        conversation_id: str,
        pin_id: str,
        status: PinStatus,
    ) -> PinnedContextRecord:
        lock = await self._json_store.lock(conversation_id)
        async with lock:
            conversation = self._require_conversation(conversation_id)
            pins = self._transition_pin(self._read_pins(conversation), pin_id, status)
            self._write_pins(conversation, pins)
            self._json_store.update(conversation)
            return next(pin for pin in pins if pin.pin_id == pin_id)

    def extract_active_pins(self, conversation: Conversation) -> list[PinnedContextRecord]:
        return [pin for pin in self._read_pins(conversation) if pin.status == PinStatus.ACTIVE]

    def _require_conversation(self, conversation_id: str) -> Conversation:
        conversation = self._json_store.get(conversation_id)
        if conversation is None:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        return conversation

    @staticmethod
    def _require_message(conversation: Conversation, message_id: str) -> Message:
        message = next((msg for msg in conversation.messages if msg.message_id == message_id), None)
        if message is None:
            raise FileNotFoundError(f"Message {message_id} not found")
        return message

    @staticmethod
    def _read_pins(conversation: Conversation) -> list[PinnedContextRecord]:
        raw = conversation.metadata.get("pinned_contexts")
        if not isinstance(raw, list):
            return []
        pins: list[PinnedContextRecord] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                pins.append(PinnedContextRecord(**item))
            except Exception:
                continue
        return pins

    @staticmethod
    def _write_pins(conversation: Conversation, pins: list[PinnedContextRecord]) -> None:
        conversation.metadata["pinned_contexts"] = [pin.model_dump(mode="json") for pin in pins]

    @staticmethod
    def _transition_pin(
        pins: list[PinnedContextRecord],
        pin_id: str,
        status: PinStatus,
    ) -> list[PinnedContextRecord]:
        updated: list[PinnedContextRecord] = []
        found = False
        now = time.time()
        for pin in pins:
            if pin.pin_id == pin_id:
                found = True
                updated.append(pin.model_copy(update={"status": status, "updated_at": now}))
            else:
                updated.append(pin)
        if not found:
            raise FileNotFoundError(f"Pinned context {pin_id} not found")
        return updated

    @staticmethod
    def _build_record(
        *,
        conversation_id: str,
        message: Message,
        title: str | None,
    ) -> PinnedContextRecord:
        role = message.role.value if isinstance(message.role, MessageRole) else str(message.role)
        content = str(message.content or "").strip()
        if not content and message.tool_calls:
            content = "[tool loop turn]"
        if not content:
            content = "[empty]"
        resolved_title = (title or content.splitlines()[0][:80]).strip() or "Pinned context"
        return PinnedContextRecord(
            conversation_id=conversation_id,
            source_message_id=message.message_id,
            title=resolved_title,
            content=content,
            role=role if role in {"user", "assistant", "system", "tool"} else "context",
            token_count=max(1, len(content) // 4),
            metadata={"origin_message_id": message.message_id},
        )
