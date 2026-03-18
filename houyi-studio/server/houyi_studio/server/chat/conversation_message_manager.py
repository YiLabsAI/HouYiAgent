from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable

from .json_store import JsonStore
from .types import EditMessageRequest, Message, MessageRole, SendMessageRequest


class ConversationMessageManager:
    def __init__(
        self,
        *,
        json_store: JsonStore,
        send_message: Callable[[str, SendMessageRequest], AsyncIterator[str]],
    ) -> None:
        self._json_store = json_store
        self._send_message = send_message

    async def edit_message(
        self,
        conversation_id: str,
        message_id: str,
        request: EditMessageRequest,
    ) -> Message:
        conv_lock = await self._json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")
            message = next(
                (item for item in conversation.messages if item.message_id == message_id), None
            )
            if message is None:
                raise ValueError(f"Message {message_id} not found")
            if message.role != MessageRole.USER:
                raise ValueError("Only user messages can be edited")
            message.content = request.content
            message.metadata["edited"] = True
            message.metadata["edited_at"] = time.time()
            conversation.updated_at = time.time()
            self._json_store.update(conversation)
            return message

    async def delete_message(self, conversation_id: str, message_id: str) -> None:
        conv_lock = await self._json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")
            message_index = next(
                (
                    idx
                    for idx, item in enumerate(conversation.messages)
                    if item.message_id == message_id
                ),
                None,
            )
            if message_index is None:
                raise ValueError(f"Message {message_id} not found")
            to_remove_ids = self._resolve_delete_ids(conversation.messages, message_index)
            conversation.messages = [
                item for item in conversation.messages if item.message_id not in to_remove_ids
            ]
            conversation.updated_at = time.time()
            self._json_store.update(conversation)

    async def regenerate_message(
        self,
        conversation_id: str,
        message_id: str,
    ) -> AsyncIterator[str]:
        last_user_content = await self._prepare_regeneration(
            conversation_id=conversation_id,
            message_id=message_id,
        )
        request = SendMessageRequest(content=last_user_content)
        async for chunk in self._send_message(conversation_id, request):
            yield chunk

    async def _prepare_regeneration(self, *, conversation_id: str, message_id: str) -> str:
        conv_lock = await self._json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self._json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")
            message_index = next(
                (
                    idx
                    for idx, item in enumerate(conversation.messages)
                    if item.message_id == message_id
                ),
                None,
            )
            if message_index is None:
                raise ValueError(f"Message {message_id} not found")
            target_message = conversation.messages[message_index]
            if target_message.role != MessageRole.ASSISTANT:
                raise ValueError("Only assistant messages can be regenerated")
            last_user_content = self._find_preceding_user_content(
                conversation.messages,
                before_index=message_index,
            )
            conversation.messages = conversation.messages[:message_index]
            if conversation.messages and conversation.messages[-1].role == MessageRole.USER:
                conversation.messages.pop()
            conversation.updated_at = time.time()
            self._json_store.update(conversation)
            return last_user_content

    def _find_preceding_user_content(self, messages: list[Message], *, before_index: int) -> str:
        for idx in range(before_index - 1, -1, -1):
            if messages[idx].role == MessageRole.USER:
                return messages[idx].content
        raise ValueError("No user message found before the assistant message")

    def _resolve_delete_ids(self, messages: list[Message], message_index: int) -> set[str]:
        target_message = messages[message_index]
        to_remove_ids = {target_message.message_id}
        if target_message.role != MessageRole.ASSISTANT:
            return to_remove_ids
        removed_tool_call_ids: set[str] = set()
        cursor = message_index - 1
        while cursor >= 0 and messages[cursor].role == MessageRole.TOOL:
            tool_message = messages[cursor]
            to_remove_ids.add(tool_message.message_id)
            if tool_message.tool_call_id:
                removed_tool_call_ids.add(str(tool_message.tool_call_id))
            cursor -= 1
        if cursor < 0:
            return to_remove_ids
        carrier = messages[cursor]
        if carrier.role != MessageRole.ASSISTANT or not carrier.tool_calls:
            return to_remove_ids
        carrier_call_ids = {
            str(call.get("id"))
            for call in carrier.tool_calls
            if isinstance(call, dict) and call.get("id")
        }
        if removed_tool_call_ids and not carrier_call_ids.intersection(removed_tool_call_ids):
            return to_remove_ids
        to_remove_ids.add(carrier.message_id)
        return to_remove_ids
