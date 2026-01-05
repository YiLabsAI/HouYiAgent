"""Tests for LLM adapters."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.llm.base import LLMMessage, LLMResponse, MessageRole

# Try to import adapters, skip tests if dependencies not available
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    from houyi.llm.openai_adapter import OpenAIAdapter

    OPENAI_AVAILABLE = True
except (ImportError, ValueError):
    OpenAIAdapter = None  # type: ignore

try:
    from houyi.llm.anthropic_adapter import AnthropicAdapter

    ANTHROPIC_AVAILABLE = True
except (ImportError, ValueError):
    AnthropicAdapter = None  # type: ignore


class TestLLMMessage:
    """Test LLMMessage."""

    def test_message_creation(self) -> None:
        """Test creating a message."""
        msg = LLMMessage(role=MessageRole.USER, content="Hello, world!")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.name is None
        assert msg.tool_calls is None

    def test_message_with_tool_calls(self) -> None:
        """Test message with tool calls."""
        msg = LLMMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[{"id": "call_1", "type": "function"}],
        )

        assert msg.role == MessageRole.ASSISTANT
        assert len(msg.tool_calls) == 1


class TestLLMResponse:
    """Test LLMResponse."""

    def test_response_creation(self) -> None:
        """Test creating a response."""
        response = LLMResponse(
            content="Test response", finish_reason="stop", model="gpt-4", usage={"total_tokens": 10}
        )

        assert response.content == "Test response"
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 10

    def test_from_openai(self) -> None:
        """Test creating response from OpenAI format."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert response.content == "Hello!"
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 8

    def test_from_anthropic(self) -> None:
        """Test creating response from Anthropic format."""
        # Mock Anthropic response
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello from Claude!"
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-3-opus"

        response = LLMResponse.from_anthropic(mock_response)

        assert response.content == "Hello from Claude!"
        assert response.finish_reason == "end_turn"
        assert response.model == "claude-3-opus"


class TestOpenAIAdapter:
    """Test OpenAIAdapter."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    def test_adapter_init_with_api_key(self) -> None:
        """Test adapter initialization with API key."""
        adapter = OpenAIAdapter(api_key="test-key", model="gpt-3.5-turbo")

        assert adapter.api_key == "test-key"
        assert adapter.model == "gpt-3.5-turbo"

    def test_adapter_init_without_api_key(self) -> None:
        """Test adapter initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not provided"):
                OpenAIAdapter()

    def test_adapter_init_from_env(self) -> None:
        """Test adapter initialization from environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            adapter = OpenAIAdapter()
            assert adapter.api_key == "env-key"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    async def test_chat(self) -> None:
        """Test chat method."""
        adapter = OpenAIAdapter(api_key="test-key")

        # Mock the client's chat.completions.create method
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage(role=MessageRole.USER, content="Hello")]
        response = await adapter.chat(messages)

        assert response.content == "Test response"
        assert response.finish_reason == "stop"
        assert adapter.client.chat.completions.create.called

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    async def test_chat_with_tools(self) -> None:
        """Test chat with tools."""
        adapter = OpenAIAdapter(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_tool_call = MagicMock()
        mock_tool_call.model_dump.return_value = {"id": "call_1", "type": "function"}
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4"

        adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage(role=MessageRole.USER, content="Use a tool")]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        response = await adapter.chat(messages, tools=tools)

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    def test_normalize_messages(self) -> None:
        """Test message normalization."""
        adapter = OpenAIAdapter(api_key="test-key")

        # Test with LLMMessage objects
        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
            LLMMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        normalized = adapter._normalize_messages(messages)

        assert len(normalized) == 2
        assert normalized[0]["role"] == "user"
        assert normalized[0]["content"] == "Hello"

        # Test with dict messages
        dict_messages = [{"role": "user", "content": "Test"}]
        normalized_dict = adapter._normalize_messages(dict_messages)
        assert len(normalized_dict) == 1


class TestAnthropicAdapter:
    """Test AnthropicAdapter."""

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic package not installed")
    def test_adapter_init_with_api_key(self) -> None:
        """Test adapter initialization with API key."""
        adapter = AnthropicAdapter(api_key="test-key", model="claude-3-opus")

        assert adapter.api_key == "test-key"
        assert adapter.model == "claude-3-opus"

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic package not installed")
    def test_adapter_init_without_api_key(self) -> None:
        """Test adapter initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key not provided"):
                AnthropicAdapter()

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic package not installed")
    def test_adapter_init_from_env(self) -> None:
        """Test adapter initialization from environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            adapter = AnthropicAdapter()
            assert adapter.api_key == "env-key"

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic package not installed")
    def test_normalize_messages(self) -> None:
        """Test message normalization."""
        adapter = AnthropicAdapter(api_key="test-key")

        # Test with LLMMessage objects
        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
            LLMMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ]
        normalized = adapter._normalize_messages(messages)

        assert len(normalized) == 2
        assert normalized[0]["role"] == "user"
        assert normalized[0]["content"] == "Hello"
