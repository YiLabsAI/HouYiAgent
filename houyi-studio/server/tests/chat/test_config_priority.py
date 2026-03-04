"""Integration tests for global/conversation/request config priority.

Verifies the three-level priority chain:
  request-level > conversation-level > global default

Tests cover model selection and system_instructions resolution.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import Conversation, Message, MessageRole, SendMessageRequest

from houyi.llm.base import StreamChunk


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    return JsonStore(data_dir=tmp_path)


GLOBAL_MODEL = "global-default-model"
GLOBAL_SYSTEM = "You are a global assistant."
CONV_MODEL = "conv-level-model"
CONV_SYSTEM = "You are a conversation assistant."
REQ_MODEL = "req-level-model"


def _make_conversation(
    store: JsonStore,
    model: str = "",
    system_instructions: str = "",
) -> Conversation:
    """Create a conversation with one user message."""
    conv = Conversation(title="Priority Test", model=model, system_instructions=system_instructions)
    conv.messages = [
        Message(message_id="u1", role=MessageRole.USER, content="Hello"),
    ]
    return store.create(conv)


def _mock_llm_adapter():
    """Create a mock LLM adapter that yields a single chunk."""
    adapter = MagicMock()

    async def fake_stream(*args, **kwargs):
        yield StreamChunk(content_delta="OK")

    adapter.stream_chat = MagicMock(side_effect=fake_stream)
    return adapter


def _patch_adapter(service: ChatService, mock_adapter):
    """Patch _get_adapter_for_model to always return mock_adapter.

    ChatService no longer has a single ``_llm_adapter`` attribute; it routes
    models via ``_get_adapter_for_model()``.  We patch that method so every
    model resolves to our mock.
    """
    return patch.object(service, "_get_adapter_for_model", return_value=mock_adapter)


class TestModelPriority:
    """Verify model selection follows request > conversation > global."""

    @pytest.mark.asyncio
    async def test_global_default_used_when_no_overrides(self, store: JsonStore):
        """When neither request nor conversation specify model, global default is used."""
        conv = _make_conversation(store, model="", system_instructions="")
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test")
            chunks = []
            async for chunk in service.send_message(conv.conversation_id, request):
                chunks.append(chunk)

            # Verify the model passed to stream_chat is the global default
            mock_adapter.stream_chat.assert_called_once()
            call_kwargs = mock_adapter.stream_chat.call_args
            assert (
                call_kwargs.kwargs.get("model") == GLOBAL_MODEL
                or call_kwargs[1].get("model") == GLOBAL_MODEL
            )

    @pytest.mark.asyncio
    async def test_conversation_model_overrides_global(self, store: JsonStore):
        """Conversation-level model overrides global default."""
        conv = _make_conversation(store, model=CONV_MODEL)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test")
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            call_kwargs = mock_adapter.stream_chat.call_args
            used_model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
            assert used_model == CONV_MODEL

    @pytest.mark.asyncio
    async def test_request_model_overrides_conversation(self, store: JsonStore):
        """Request-level model overrides conversation-level model."""
        conv = _make_conversation(store, model=CONV_MODEL)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test", model=REQ_MODEL)
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            call_kwargs = mock_adapter.stream_chat.call_args
            used_model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
            assert used_model == REQ_MODEL


class TestSystemInstructionsPriority:
    """Verify system_instructions follows conversation > global."""

    @pytest.mark.asyncio
    async def test_global_system_used_when_conversation_empty(self, store: JsonStore):
        """When conversation has no system_instructions, global default is used."""
        conv = _make_conversation(store, system_instructions="")
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test")
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            # The system instructions should be passed to ContextPlanner
            # We verify indirectly: the LLM messages should contain the global system prompt
            call_args = mock_adapter.stream_chat.call_args
            messages = call_args.kwargs.get("messages") or call_args[0][0]
            system_msgs = [m for m in messages if m.get("role") == "system"]
            assert len(system_msgs) > 0
            assert GLOBAL_SYSTEM in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_conversation_system_overrides_global(self, store: JsonStore):
        """Conversation-level system_instructions overrides global."""
        conv = _make_conversation(store, system_instructions=CONV_SYSTEM)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test")
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            call_args = mock_adapter.stream_chat.call_args
            messages = call_args.kwargs.get("messages") or call_args[0][0]
            system_msgs = [m for m in messages if m.get("role") == "system"]
            assert len(system_msgs) > 0
            assert CONV_SYSTEM in system_msgs[0]["content"]
            assert GLOBAL_SYSTEM not in system_msgs[0]["content"]


class TestEnableReasoningPassthrough:
    """Verify enable_reasoning is passed through to LLM adapter."""

    @pytest.mark.asyncio
    async def test_enable_reasoning_passed_to_adapter(self, store: JsonStore):
        conv = _make_conversation(store)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test", enable_reasoning=True)
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            call_kwargs = mock_adapter.stream_chat.call_args
            assert call_kwargs.kwargs.get("enable_reasoning") is True

    @pytest.mark.asyncio
    async def test_reasoning_not_passed_when_disabled(self, store: JsonStore):
        conv = _make_conversation(store)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test")
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            call_kwargs = mock_adapter.stream_chat.call_args
            assert "enable_reasoning" not in (call_kwargs.kwargs or {})
