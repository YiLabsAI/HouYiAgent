"""Tests for llm/base.py"""

import pytest

from houyi.llm.base import LLMAdapter, LLMMessage, LLMResponse, MessageRole


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
