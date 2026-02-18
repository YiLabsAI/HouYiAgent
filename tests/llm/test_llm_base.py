"""Tests for llm/base.py"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from houyi.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMAdapter,
    LLMMessage,
    LLMResponse,
    MessageRole,
)


def test_message_role_enum():
    """Test MessageRole enum values."""
    assert MessageRole.SYSTEM == "system"
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.TOOL == "tool"


def test_llm_message_creation():
    """Test LLMMessage creation."""
    msg = LLMMessage(role=MessageRole.USER, content="Hello, world!")

    assert msg.role == MessageRole.USER
    assert msg.content == "Hello, world!"
    assert msg.name is None
    assert msg.tool_calls is None


def test_llm_message_with_name():
    """Test LLMMessage with name."""
    msg = LLMMessage(role=MessageRole.ASSISTANT, content="Response", name="assistant_1")

    assert msg.name == "assistant_1"


def test_llm_message_with_tool_calls():
    """Test LLMMessage with tool calls."""
    msg = LLMMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "test"}'},
            }
        ],
    )

    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["function"]["name"] == "search"


def test_llm_response_creation():
    """Test LLMResponse creation."""
    response = LLMResponse(
        content="Test response",
        model="gpt-4",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="stop",
    )

    assert response.content == "Test response"
    assert response.model == "gpt-4"
    assert response.usage["total_tokens"] == 30
    assert response.finish_reason == "stop"


def test_llm_response_default_values():
    """Test LLMResponse with default values."""
    response = LLMResponse(content="Test", model="gpt-3.5-turbo", finish_reason="stop")

    assert response.usage == {}
    assert response.tool_calls == []
    assert response.metadata == {}


def test_llm_adapter_abstract():
    """Test LLMAdapter is abstract."""
    # LLMAdapter should not be instantiable directly
    with pytest.raises(TypeError):
        LLMAdapter()


def test_llm_message_serialization():
    """Test LLMMessage can be serialized."""
    msg = LLMMessage(role=MessageRole.USER, content="Test message")

    # Pydantic models can be converted to dict
    msg_dict = msg.model_dump()
    assert msg_dict["role"] == "user"
    assert msg_dict["content"] == "Test message"


def test_llm_response_serialization():
    """Test LLMResponse can be serialized."""
    response = LLMResponse(
        content="Response", model="gpt-4", finish_reason="stop", usage={"total_tokens": 100}
    )

    response_dict = response.model_dump()
    assert response_dict["content"] == "Response"
    assert response_dict["model"] == "gpt-4"
    assert response_dict["usage"]["total_tokens"] == 100
    assert response_dict["finish_reason"] == "stop"


def test_message_role_string_values():
    """Test MessageRole string values."""
    roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL]
    role_strings = ["system", "user", "assistant", "tool"]

    for role, role_str in zip(roles, role_strings):
        assert role.value == role_str


def test_llm_response_with_tool_calls():
    """Test LLMResponse with tool calls."""
    response = LLMResponse(
        content="",
        model="gpt-4",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "test"}'},
            }
        ],
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "search"
    assert response.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Tests for the abstract LLMAdapter base class behaviour (stream_completion)
# ---------------------------------------------------------------------------


class StubAdapter(LLMAdapter):
    """Minimal concrete adapter for testing base class stream_completion."""

    def __init__(self):
        self.received_messages: list[list[dict]] = []

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content="stub", tool_calls=[], finish_reason="stop", usage={}, model="stub"
        )

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        normalized = self._normalize_messages(messages)
        self.received_messages.append(normalized)
        last_user = ""
        for msg in reversed(normalized):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        for word in last_user.split():
            yield (word, None)


class TestLLMAdapterBaseStreamCompletion:
    """Test the base class stream_completion convenience wrapper."""

    @pytest.mark.asyncio
    async def test_stream_completion_delegates_to_stream_chat(self):
        """stream_completion wraps prompt as user message and calls stream_chat."""
        adapter = StubAdapter()
        chunks = []
        async for content, reasoning in adapter.stream_completion("Hello world"):
            chunks.append((content, reasoning))

        assert len(chunks) == 2
        assert chunks[0] == ("Hello", None)
        assert chunks[1] == ("world", None)
        assert len(adapter.received_messages) == 1
        assert adapter.received_messages[0][0]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_completion_empty_prompt(self):
        adapter = StubAdapter()
        chunks = []
        async for content, _reasoning in adapter.stream_completion(""):
            chunks.append(content)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_normalize_messages_with_llm_message(self):
        """_normalize_messages converts LLMMessage to dict."""
        adapter = StubAdapter()

        messages = [
            LLMMessage(role=MessageRole.USER, content="hello"),
            {"role": "assistant", "content": "hi"},
        ]
        chunks = []
        async for content, _reasoning in adapter.stream_chat(messages):
            chunks.append(content)
        assert len(chunks) > 0
