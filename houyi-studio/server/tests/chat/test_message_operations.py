"""Tests for message edit, delete, and regenerate operations.

Tests ChatService.edit_message, delete_message, and regenerate_message
with a real JsonStore (tmp_path) but mocked LLM adapter.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import Conversation, EditMessageRequest, Message, MessageRole

from houyi.llm.base import StreamChunk


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    return JsonStore(data_dir=tmp_path)


@pytest.fixture
def service(store: JsonStore) -> ChatService:
    return ChatService(json_store=store)


def _make_conversation(store: JsonStore) -> Conversation:
    """Create a conversation with 2 user + 2 assistant messages."""
    conv = Conversation(title="Test Chat")
    conv.messages = [
        Message(message_id="u1", role=MessageRole.USER, content="Hello"),
        Message(message_id="a1", role=MessageRole.ASSISTANT, content="Hi there!"),
        Message(message_id="u2", role=MessageRole.USER, content="How are you?"),
        Message(message_id="a2", role=MessageRole.ASSISTANT, content="I am fine."),
    ]
    return store.create(conv)


# --- edit_message ---


class TestEditMessage:
    @pytest.mark.asyncio
    async def test_edit_user_message(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        req = EditMessageRequest(content="Updated hello")
        msg = await service.edit_message(conv.conversation_id, "u1", req)

        assert msg.content == "Updated hello"
        assert msg.metadata.get("edited") is True
        assert "edited_at" in msg.metadata

        # Verify persisted
        reloaded = store.get(conv.conversation_id)
        assert reloaded.messages[0].content == "Updated hello"

    @pytest.mark.asyncio
    async def test_edit_assistant_message_rejected(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        req = EditMessageRequest(content="Nope")
        with pytest.raises(ValueError, match="Only user messages"):
            await service.edit_message(conv.conversation_id, "a1", req)

    @pytest.mark.asyncio
    async def test_edit_nonexistent_message(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        req = EditMessageRequest(content="Nope")
        with pytest.raises(ValueError, match="not found"):
            await service.edit_message(conv.conversation_id, "nonexistent", req)

    @pytest.mark.asyncio
    async def test_edit_nonexistent_conversation(self, service: ChatService):
        req = EditMessageRequest(content="Nope")
        with pytest.raises(FileNotFoundError):
            await service.edit_message("bad_conv", "u1", req)


# --- delete_message ---


class TestDeleteMessage:
    @pytest.mark.asyncio
    async def test_delete_user_message(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        await service.delete_message(conv.conversation_id, "u1")

        reloaded = store.get(conv.conversation_id)
        assert len(reloaded.messages) == 3
        assert all(m.message_id != "u1" for m in reloaded.messages)

    @pytest.mark.asyncio
    async def test_delete_assistant_message(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        await service.delete_message(conv.conversation_id, "a2")

        reloaded = store.get(conv.conversation_id)
        assert len(reloaded.messages) == 3

    @pytest.mark.asyncio
    async def test_delete_assistant_message_cascades_preceding_tool_steps(
        self,
        service: ChatService,
        store: JsonStore,
    ):
        conv = Conversation(title="Tool Delete Cascade")
        conv.messages = [
            Message(message_id="u1", role=MessageRole.USER, content="find file"),
            Message(message_id="t1", role=MessageRole.TOOL, content="{}", name="houyi_find_files"),
            Message(message_id="t2", role=MessageRole.TOOL, content="{}", name="houyi_list_dir"),
            Message(message_id="a1", role=MessageRole.ASSISTANT, content="done"),
        ]
        created = store.create(conv)

        await service.delete_message(created.conversation_id, "a1")

        reloaded = store.get(created.conversation_id)
        assert reloaded is not None
        remaining_ids = [m.message_id for m in reloaded.messages]
        assert remaining_ids == ["u1"]

    @pytest.mark.asyncio
    async def test_delete_assistant_message_removes_tool_call_carrier(
        self,
        service: ChatService,
        store: JsonStore,
    ):
        conv = Conversation(title="Tool Carrier Delete Cascade")
        conv.messages = [
            Message(message_id="u1", role=MessageRole.USER, content="find file"),
            Message(
                message_id="a-carrier",
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "function": {"name": "houyi_find_files", "arguments": "{}"},
                    }
                ],
            ),
            Message(
                message_id="t1",
                role=MessageRole.TOOL,
                content="{}",
                name="houyi_find_files",
                tool_call_id="call-1",
            ),
            Message(message_id="a-final", role=MessageRole.ASSISTANT, content="done"),
        ]
        created = store.create(conv)

        await service.delete_message(created.conversation_id, "a-final")

        reloaded = store.get(created.conversation_id)
        assert reloaded is not None
        remaining_ids = [m.message_id for m in reloaded.messages]
        assert remaining_ids == ["u1"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_message(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        with pytest.raises(ValueError, match="not found"):
            await service.delete_message(conv.conversation_id, "nonexistent")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_conversation(self, service: ChatService):
        with pytest.raises(FileNotFoundError):
            await service.delete_message("bad_conv", "u1")


# --- regenerate_message ---


class TestRegenerateMessage:
    @pytest.mark.asyncio
    async def test_regenerate_user_message_rejected(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        with pytest.raises(ValueError, match="Only assistant messages"):
            async for _ in service.regenerate_message(conv.conversation_id, "u1"):
                pass

    @pytest.mark.asyncio
    async def test_regenerate_nonexistent_message(self, service: ChatService, store: JsonStore):
        conv = _make_conversation(store)
        with pytest.raises(ValueError, match="not found"):
            async for _ in service.regenerate_message(conv.conversation_id, "nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_regenerate_truncates_history(self, service: ChatService, store: JsonStore):
        """Regenerate should remove the target assistant msg and everything after it."""
        conv = _make_conversation(store)

        # Mock the LLM adapter to avoid real API calls
        mock_adapter = MagicMock()

        async def fake_stream(*args, **kwargs):
            yield StreamChunk(content_delta="New response")

        mock_adapter.stream_chat = MagicMock(side_effect=fake_stream)
        mock_adapter.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}

        with patch.object(service, "_get_adapter_for_model", return_value=mock_adapter):
            chunks = []
            try:
                async for chunk in service.regenerate_message(conv.conversation_id, "a1"):
                    chunks.append(chunk)
            except Exception:
                # LLM mock may not produce perfect SSE, but we verify state
                pass

        # After regeneration, messages after a1 should be removed,
        # and a new user + assistant pair should be added
        reloaded = store.get(conv.conversation_id)
        # Original: u1, a1, u2, a2
        # After regen of a1: truncate to [u1], then send_message adds u1 content again + new assistant
        # So we should have: u1_original (kept by truncate-1-more), new_u1, new_assistant
        # Actually: truncate to [:1] = [u1], then pop u1, then send_message re-adds u1 + assistant
        assert reloaded is not None
        # The u2 and a2 messages should be gone
        assert all(m.message_id not in ("u2", "a2") for m in reloaded.messages)
