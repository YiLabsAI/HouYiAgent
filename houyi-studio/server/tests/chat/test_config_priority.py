"""Integration tests for global/conversation/request config priority.

Verifies the three-level priority chain:
  request-level > conversation-level > global default

Tests cover model selection and system_instructions resolution.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from houyi_studio.server.chat import chat_service as chat_service_module
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.settings_store import GlobalSettings, ProviderConfig
from houyi_studio.server.chat.types import (
    Conversation,
    ConversationContextState,
    Message,
    MessageRole,
    SendMessageRequest,
)

from houyi.adapters.llm.base import StreamChunk
from houyi.adapters.memory.answerer import AnswerResult


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

    ChatService no longer has a single _llm_adapter attribute; it routes
    models via _get_adapter_for_model().  We patch that method so every
    model resolves to our mock.
    """
    return patch.object(service, "_get_adapter_for_model", return_value=mock_adapter)


class TestModelPriority:
    """Verify model selection follows request > conversation > global."""

    @pytest.mark.asyncio
    async def test_uses_global_model(self, store: JsonStore):
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
    async def test_uses_conversation_model(self, store: JsonStore):
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
    async def test_uses_request_model(self, store: JsonStore):
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
    async def test_uses_global_system(self, store: JsonStore):
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
    async def test_uses_conversation_system(self, store: JsonStore):
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
    async def test_passes_enable_reasoning(self, store: JsonStore):
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
            persisted = store.get(conv.conversation_id)
            assert persisted is not None
            budget = persisted.messages[-1].metadata.get("budget", {})
            assert call_kwargs.kwargs.get("thinking_budget") == budget.get("reasoning_budget")
            assert call_kwargs.kwargs.get("thinking_budget", 0) > 0

    @pytest.mark.asyncio
    async def test_raises_reasoning_max_tokens(self, store: JsonStore):
        conv = _make_conversation(store)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test", enable_reasoning=True, max_tokens=32)
            chunks = []
            async for chunk in service.send_message(conv.conversation_id, request):
                chunks.append(chunk)

            call_kwargs = mock_adapter.stream_chat.call_args
            assert call_kwargs.kwargs.get("enable_reasoning") is True
            assert call_kwargs.kwargs.get("max_tokens") == 512

            complete_chunks = [chunk for chunk in chunks if "event: message.complete" in chunk]
            assert complete_chunks
            assert '"budget": {' in complete_chunks[-1]
            assert '"max_tokens_guardrail_applied": true' in complete_chunks[-1]
            assert '"answer_reserve": 512' in complete_chunks[-1]

            persisted = store.get(conv.conversation_id)
            assert persisted is not None
            assert persisted.messages[-1].metadata["budget"]["max_tokens_guardrail_applied"] is True
            assert persisted.messages[-1].metadata["budget"]["answer_reserve"] == 512

    @pytest.mark.asyncio
    async def test_provider_defaults(self, store: JsonStore):
        conv = _make_conversation(store)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            chunks = []
            request = SendMessageRequest(content="test", enable_reasoning=True)
            async for chunk in service.send_message(conv.conversation_id, request):
                chunks.append(chunk)

            call_kwargs = mock_adapter.stream_chat.call_args
            assert call_kwargs.kwargs.get("enable_reasoning") is True
            assert "max_tokens" not in (call_kwargs.kwargs or {})

            complete_chunks = [chunk for chunk in chunks if "event: message.complete" in chunk]
            assert complete_chunks
            assert '"max_tokens_guardrail_applied": false' in complete_chunks[-1]
            assert '"max_tokens_source": "provider_default"' in complete_chunks[-1]

    @pytest.mark.asyncio
    async def test_reports_budget_decision(self, store: JsonStore):
        conv = _make_conversation(store)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="test", max_tokens=256)
            chunks = []
            async for chunk in service.send_message(conv.conversation_id, request):
                chunks.append(chunk)

            complete_chunks = [chunk for chunk in chunks if "event: message.complete" in chunk]
            assert complete_chunks
            assert '"output_budget": 256' in complete_chunks[-1]
            assert '"input_budget": ' in complete_chunks[-1]
            assert '"max_tokens_to_send": 256' in complete_chunks[-1]

    @pytest.mark.asyncio
    async def test_skips_enable_reasoning(self, store: JsonStore):
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


class TestRepoIntentIsolation:
    @pytest.mark.asyncio
    async def test_preserves_history_low_pressure(self, store: JsonStore):
        conv = Conversation(title="Repo Intent Isolation", model="", system_instructions="")
        conv.messages = [
            Message(message_id=f"m{i}", role=MessageRole.USER, content=f"older message {i}")
            for i in range(1, 8)
        ]
        conv.conversation_context_state = ConversationContextState(
            conversation_id=conv.conversation_id,
            used_units=750,
            max_units=1000,
            state="elevated",
            updated_at=1.0,
        )
        conv = store.create(conv)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="Read README from https://github.com/foo/bar")
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            call_args = mock_adapter.stream_chat.call_args
            messages = call_args.kwargs.get("messages") or call_args[0][0]
            non_system_messages = [m for m in messages if m.get("role") != "system"]
            contents = [str(m.get("content") or "") for m in non_system_messages]

            assert contents == [
                "older message 1",
                "older message 2",
                "older message 3",
                "older message 4",
                "older message 5",
                "older message 6",
                "older message 7",
                "Read README from https://github.com/foo/bar",
            ]
            persisted = store.get(conv.conversation_id)
            assert persisted is not None
            history = persisted.metadata.get("compaction_history")
            assert history is None or history == []

    @pytest.mark.asyncio
    async def test_drop_tool_group(self, store: JsonStore):
        conv = Conversation(title="Repo Intent Tool Isolation", model="", system_instructions="")
        conv.messages = [
            Message(message_id="u1", role=MessageRole.USER, content="older user 1"),
            Message(message_id="u2", role=MessageRole.USER, content="older user 2"),
            Message(message_id="u3", role=MessageRole.USER, content="older user 3"),
            Message(
                message_id="a-tool",
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "demo", "arguments": '{"path":"README.md"}'},
                    }
                ],
            ),
            Message(message_id="u4", role=MessageRole.USER, content="older user 4"),
            Message(message_id="u5", role=MessageRole.USER, content="older user 5"),
        ]
        conv = store.create(conv)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="Read README from https://github.com/foo/bar")
            async for _ in service.send_message(conv.conversation_id, request):
                pass

            call_args = mock_adapter.stream_chat.call_args
            messages = call_args.kwargs.get("messages") or call_args[0][0]
            non_system_messages = [m for m in messages if m.get("role") != "system"]

            assert all(not m.get("tool_calls") for m in non_system_messages)
            assert [str(m.get("content") or "") for m in non_system_messages] == [
                "older user 1",
                "older user 2",
                "older user 3",
                "older user 4",
                "older user 5",
                "Read README from https://github.com/foo/bar",
            ]
            persisted = store.get(conv.conversation_id)
            assert persisted is not None
            history = persisted.metadata.get("compaction_history")
            assert history is None or history == []


class TestProviderRouting:
    def test_uses_vertex_adapter(self, store: JsonStore):
        settings = GlobalSettings(
            providers=[
                ProviderConfig(
                    id="google-ai-custom",
                    name="Google AI",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                    models=["gemini-2.5-pro"],
                    enabled=True,
                )
            ]
        )
        settings_store = SimpleNamespace(get=lambda: settings)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            settings_store=settings_store,
        )

        with patch.object(
            chat_service_module,
            "create_vertex_adapter",
            return_value="vertex-adapter",
        ) as mocked:
            adapter = service._get_adapter_for_model("gemini-2.5-pro")

        assert adapter == "vertex-adapter"
        mocked.assert_called_once()

    def test_uses_generic_openai_compat(self, store: JsonStore):
        settings = GlobalSettings(
            providers=[
                ProviderConfig(
                    id="moonshot-custom",
                    name="Moonshot",
                    api_key="test-key",
                    base_url="https://api.moonshot.ai/v1",
                    models=["moonshotai/Kimi-K2.5"],
                    enabled=True,
                )
            ]
        )
        settings_store = SimpleNamespace(get=lambda: settings)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            settings_store=settings_store,
        )

        class _FakeCompat:
            def __init__(self, *, api_key=None, base_url=None, model=None):
                self.api_key = api_key
                self.base_url = base_url
                self.model = model

        service._model_adapter_resolver._openai_compat_adapter_cls = _FakeCompat

        adapter = service._get_adapter_for_model("moonshotai/Kimi-K2.5")

        assert isinstance(adapter, _FakeCompat)
        assert not isinstance(adapter, chat_service_module.SiliconFlowAdapter)
        assert adapter.base_url == "https://api.moonshot.ai/v1"
        assert adapter.model == "moonshotai/Kimi-K2.5"

    def test_uses_siliconflow_adapter(self, store: JsonStore):
        settings = GlobalSettings(
            providers=[
                ProviderConfig(
                    id="deepseek-custom",
                    name="DeepSeek Mirror",
                    api_key="test-key",
                    base_url="https://api.siliconflow.cn/v1",
                    models=["deepseek-ai/DeepSeek-R1"],
                    enabled=True,
                )
            ]
        )
        settings_store = SimpleNamespace(get=lambda: settings)
        service = ChatService(
            json_store=store,
            default_model=GLOBAL_MODEL,
            settings_store=settings_store,
        )

        adapter = service._get_adapter_for_model("deepseek-ai/DeepSeek-R1")

        assert isinstance(adapter, chat_service_module.SiliconFlowAdapter)
        assert adapter.base_url == "https://api.siliconflow.cn/v1"
        assert adapter.model == "deepseek-ai/DeepSeek-R1"


class TestMemoryAnswerRouting:
    @pytest.mark.asyncio
    async def test_routes_memory_answer(self, store: JsonStore):
        conv = _make_conversation(store, model="", system_instructions="")
        memory_service = MagicMock()
        memory_service.should_use_memory_answer.return_value = True
        memory_service.answer_query = AsyncMock(
            return_value=AnswerResult(
                answer="You said deploy on Friday.",
                abstained=False,
                reason="deterministic_match",
                citations=(),
                facts_used=1,
                prompt_chars=0,
                raw_llm_output="",
            )
        )
        service = ChatService(
            json_store=store,
            memory_service=memory_service,
            default_model=GLOBAL_MODEL,
            default_system_instructions=GLOBAL_SYSTEM,
        )
        mock_adapter = _mock_llm_adapter()

        with _patch_adapter(service, mock_adapter):
            request = SendMessageRequest(content="Do you remember my deploy day?")
            chunks = []
            async for chunk in service.send_message(conv.conversation_id, request):
                chunks.append(chunk)

        mock_adapter.stream_chat.assert_not_called()
        memory_service.answer_query.assert_awaited_once()
        assert any("You said deploy on Friday." in chunk for chunk in chunks)
        assert any('"finish_reason": "memory_answer"' in chunk for chunk in chunks)
