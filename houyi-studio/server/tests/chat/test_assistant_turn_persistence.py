from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from houyi_studio.server.chat.assistant_turn_persistence import AssistantTurnPersistence
from houyi_studio.server.chat.types import Message, MessageRole


class _Lock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Span:
    def __init__(self, *, trace_id: str = "trace-1"):
        self.trace_id = trace_id
        self.attributes: dict[str, object] = {}
        self.status: tuple[str, str | None] | None = None

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: str, message: str | None = None) -> None:
        self.status = (status, message)


class TestAssistantTurnPersistence:
    @pytest.mark.asyncio
    async def test_persist_updates_context(self):
        lock = _Lock()
        conversation = SimpleNamespace(messages=[], updated_at=0.0)
        json_store = SimpleNamespace(get=MagicMock(return_value=conversation), update=MagicMock())
        deps = SimpleNamespace(
            conversation_context=SimpleNamespace(apply_appended_messages=MagicMock()),
        )
        persistence = AssistantTurnPersistence(
            json_store=json_store,
            conversation_context=deps.conversation_context,
        )
        assistant_msg = Message(role=MessageRole.ASSISTANT, content="")
        tool_msg = Message(role=MessageRole.TOOL, content="tool body", name="read_file")
        chat_span = _Span(trace_id="trace-99")

        persisted = await persistence.persist(
            conversation_id="conv-1",
            conv_lock=lock,
            assistant_msg=assistant_msg,
            content_parts=["hello"],
            reasoning_parts=["think"],
            persisted_tool_messages=[tool_msg],
            usage_payload={"total_tokens": 10},
            finish_reason="stop",
            budget_metadata={"input_budget": 20},
            generation_metadata={"first_token_ms": 12.3},
            completion_emitted_at=time.perf_counter(),
            chat_span=chat_span,
            model="model-1",
        )

        assert persisted is True
        assert assistant_msg.content == "hello"
        assert assistant_msg.reasoning_content == "think"
        assert assistant_msg.metadata["usage"] == {"total_tokens": 10}
        assert assistant_msg.metadata["finish_reason"] == "stop"
        assert assistant_msg.metadata["budget"] == {"input_budget": 20}
        assert assistant_msg.metadata["trace_id"] == "trace-99"
        assert assistant_msg.metadata["post_stream_persist_ms"] >= 0
        assert conversation.messages == [tool_msg, assistant_msg]
        deps.conversation_context.apply_appended_messages.assert_called_once_with(
            conversation,
            messages=[tool_msg, assistant_msg],
            model="model-1",
        )
        assert chat_span.status == ("ok", None)

    @pytest.mark.asyncio
    async def test_persist_reasoning_only(self):
        lock = _Lock()
        conversation = SimpleNamespace(messages=[], updated_at=0.0)
        json_store = SimpleNamespace(get=MagicMock(return_value=conversation), update=MagicMock())
        deps = SimpleNamespace(
            conversation_context=SimpleNamespace(apply_appended_messages=MagicMock()),
        )
        persistence = AssistantTurnPersistence(
            json_store=json_store,
            conversation_context=deps.conversation_context,
        )
        assistant_msg = Message(role=MessageRole.ASSISTANT, content="")
        chat_span = _Span(trace_id="trace-100")

        persisted = await persistence.persist(
            conversation_id="conv-1",
            conv_lock=lock,
            assistant_msg=assistant_msg,
            content_parts=[],
            reasoning_parts=["think only"],
            persisted_tool_messages=[],
            usage_payload={"total_tokens": 8},
            finish_reason="stop",
            budget_metadata=None,
            generation_metadata={"final_stream_status": "reasoning_only"},
            completion_emitted_at=time.perf_counter(),
            chat_span=chat_span,
            model="model-1",
        )

        assert persisted is True
        assert assistant_msg.content == ""
        assert assistant_msg.reasoning_content == "think only"
        assert assistant_msg.metadata["usage"] == {"total_tokens": 8}
        assert assistant_msg.metadata["finish_reason"] == "stop"
        assert assistant_msg.metadata["final_stream_status"] == "reasoning_only"
        assert conversation.messages == [assistant_msg]
        deps.conversation_context.apply_appended_messages.assert_called_once_with(
            conversation,
            messages=[assistant_msg],
            model="model-1",
        )
        assert chat_span.status == ("ok", None)

    @pytest.mark.asyncio
    async def test_persist_skips_empty(self):
        lock = _Lock()
        deps = SimpleNamespace(
            json_store=SimpleNamespace(get=MagicMock(), update=MagicMock()),
            conversation_context=SimpleNamespace(
                estimate_units=MagicMock(),
                apply_appended_messages=MagicMock(),
            ),
        )
        persistence = AssistantTurnPersistence(
            json_store=deps.json_store,
            conversation_context=deps.conversation_context,
        )
        chat_span = _Span()

        persisted = await persistence.persist(
            conversation_id="conv-1",
            conv_lock=lock,
            assistant_msg=Message(role=MessageRole.ASSISTANT, content=""),
            content_parts=[],
            reasoning_parts=[],
            persisted_tool_messages=[],
            usage_payload=None,
            finish_reason=None,
            budget_metadata=None,
            generation_metadata={},
            completion_emitted_at=None,
            chat_span=chat_span,
            model="model-1",
        )

        assert persisted is False
        assert chat_span.status == ("error", "LLM returned no content")
        deps.json_store.update.assert_not_called()
