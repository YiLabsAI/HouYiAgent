"""Tests for LLM adapters."""

from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.adapters.llm.base import LLMMessage, LLMResponse, MessageRole

# Try to import adapters, skip tests if dependencies not available
try:
    from houyi.adapters.llm.openai_adapter import OpenAIAdapter

    OPENAI_AVAILABLE = True
except (ImportError, ValueError):
    OpenAIAdapter = None
    OPENAI_AVAILABLE = False

try:
    from houyi.adapters.llm.anthropic_adapter import AnthropicAdapter

    ANTHROPIC_AVAILABLE = True
except (ImportError, ValueError):
    AnthropicAdapter = None
    ANTHROPIC_AVAILABLE = False


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

    def test_openai_empty_choices(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 5
        mock_response.model = "deepseek-ai/DeepSeek-R1"

        response = LLMResponse.from_openai(mock_response)

        assert response.content == ""
        assert response.tool_calls == []
        assert response.finish_reason == "error"
        assert response.model == "deepseek-ai/DeepSeek-R1"
        assert response.usage["total_tokens"] == 5
        assert response.metadata["response_shape"] == "empty_choices"

    def test_raw_empty_choices(self) -> None:
        response = LLMResponse.from_raw_dict(
            {
                "model": "deepseek-ai/DeepSeek-V3.2",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
                "error": {"message": "provider returned no choices"},
            }
        )

        assert response.content == ""
        assert response.tool_calls == []
        assert response.finish_reason == "error"
        assert response.model == "deepseek-ai/DeepSeek-V3.2"
        assert response.metadata["response_shape"] == "empty_choices"
        assert response.metadata["provider_error"] == {"message": "provider returned no choices"}

    def test_openai_preserves_reasoning(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.reasoning_content = "I should inspect files first"
        mock_response.choices[0].message.tool_calls = [
            MagicMock(
                model_dump=MagicMock(
                    return_value={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "houyi_grep",
                            "arguments": '{"query":"tool loop"}',
                        },
                    }
                )
            )
        ]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert response.metadata["reasoning_content"] == "I should inspect files first"
        assert response.tool_calls[0]["function"]["name"] == "houyi_grep"

    def test_compatible_function_call(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.function_call = type(
            "LegacyFunctionCall",
            (),
            {
                "name": "houyi_web_search",
                "arguments": '{"query":"deepseek tool loop"}',
            },
        )
        mock_response.choices[0].finish_reason = "function_call"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "houyi_web_search"
        assert response.tool_calls[0]["function"]["arguments"] == {"query": "deepseek tool loop"}

    def test_raw_compatible_function_call(self) -> None:
        response = LLMResponse.from_raw_dict(
            {
                "model": "deepseek-ai/DeepSeek-R1",
                "choices": [
                    {
                        "finish_reason": "function_call",
                        "message": {
                            "content": "",
                            "function_call": {
                                "name": "houyi_web_search",
                                "arguments": '{"query":"siliconflow deepseek"}',
                            },
                        },
                    }
                ],
            }
        )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "houyi_web_search"
        assert response.tool_calls[0]["function"]["arguments"] == {"query": "siliconflow deepseek"}

    def test_raw_token_wrapped_toolcall(self) -> None:
        response = LLMResponse.from_raw_dict(
            {
                "model": "deepseek-ai/DeepSeek-V3.2",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "<|tool_calls_section_begin|><|tool_call_begin|>functions.houyi_list_dir:20"
                                '<|tool_call_argument_begin|>{"path": ".houyi/skills", "max_results": 50}'
                                "<|tool_call_end|><|tool_calls_section_end|>"
                            )
                        },
                    }
                ],
            }
        )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "functions.houyi_list_dir"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "path": ".houyi/skills",
            "max_results": 50,
        }

    def test_raw_xml_toolcall(self) -> None:
        response = LLMResponse.from_raw_dict(
            {
                "model": "deepseek-ai/DeepSeek-R1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "<tool_call>houyi_read_file"
                                "<arg_key>path</arg_key><arg_value>houyi/application/workflow/orchestration/plan.py</arg_value>"
                                "<arg_key>start_line</arg_key><arg_value>1</arg_value>"
                                "<arg_key>end_line</arg_key><arg_value>150</arg_value>"
                                "</tool_call>"
                            )
                        },
                    }
                ],
            }
        )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "houyi_read_file"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "path": "houyi/application/workflow/orchestration/plan.py",
            "start_line": 1,
            "end_line": 150,
        }

    def test_raw_multiple_xml_toolcalls(self) -> None:
        response = LLMResponse.from_raw_dict(
            {
                "model": "deepseek-ai/DeepSeek-V3.2",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "<tool_call>houyi_read_file"
                                "<arg_key>path</arg_key><arg_value>./houyi/domain/skill/schema.py</arg_value>"
                                "</tool_call>"
                                "<tool_call>houyi_read_file"
                                "<arg_key>path</arg_key><arg_value>./houyi/domain/skill/spec.py</arg_value>"
                                "</tool_call>"
                            )
                        },
                    }
                ],
            }
        )

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0]["function"]["name"] == "houyi_read_file"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "path": "./houyi/domain/skill/schema.py"
        }
        assert response.tool_calls[1]["function"]["name"] == "houyi_read_file"
        assert response.tool_calls[1]["function"]["arguments"] == {
            "path": "./houyi/domain/skill/spec.py"
        }

    def test_raw_bracket_toolcall(self) -> None:
        response = LLMResponse.from_raw_dict(
            {
                "model": "deepseek-ai/DeepSeek-V3.2",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "[tool:houyi_shell_exec] "
                                '{"command": "find /Users/von/workspace/HouYiAgent -name \\"*.md\\"", '
                                '"cwd": "/Users/von/workspace/HouYiAgent", "timeout_seconds": 30}'
                            )
                        },
                    }
                ],
            }
        )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "houyi_shell_exec"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "command": 'find /Users/von/workspace/HouYiAgent -name "*.md"',
            "cwd": "/Users/von/workspace/HouYiAgent",
            "timeout_seconds": 30,
        }

    def test_bracket_envelope_not_toolcall(self) -> None:
        response = LLMResponse.from_raw_dict(
            {
                "model": "deepseek-ai/DeepSeek-V3.2",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "[tool:houyi_shell_exec] "
                                '{"data": {"command": "find /Users/von/workspace/HouYiAgent -name \\"readme.md\\""}, '
                                '"message": "", "success": true}'
                            )
                        },
                    }
                ],
            }
        )

        assert response.tool_calls == []

    def test_parses_embedded_toolcall(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.reasoning_content = (
            "[tool call]\n\n"
            "<\uff5cDSML\uff5cfunction_calls>\n"
            '<\uff5cDSML\uff5cinvoke name="houyi_web_search">\n'
            '<\uff5cDSML\uff5cparameter name="query" string="true">Articles authored by Von Gosling on InfoQ in 2025</\uff5cDSML\uff5cparameter>\n'
            '<\uff5cDSML\uff5cparameter name="search_engine" string="true">bing</\uff5cDSML\uff5cparameter>\n'
            '<\uff5cDSML\uff5cparameter name="max_results" string="false">10</\uff5cDSML\uff5cparameter>\n'
            "</\uff5cDSML\uff5cinvoke>\n"
            "</\uff5cDSML\uff5cfunction_calls>"
        )
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "houyi_web_search"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "query": "Articles authored by Von Gosling on InfoQ in 2025",
            "search_engine": "bing",
            "max_results": 10,
        }

    def test_parses_token_wrapped_toolcall(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.houyi_list_dir:20"
            '<|tool_call_argument_begin|>{"path": ".houyi/skills", "max_results": 50}'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "functions.houyi_list_dir"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "path": ".houyi/skills",
            "max_results": 50,
        }

    def test_parses_xml_toolcall(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<tool_call>houyi_read_file"
            "<arg_key>path</arg_key><arg_value>houyi/application/workflow/orchestration/plan.py</arg_value>"
            "<arg_key>start_line</arg_key><arg_value>1</arg_value>"
            "<arg_key>end_line</arg_key><arg_value>150</arg_value>"
            "</tool_call>"
        )
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "houyi_read_file"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "path": "houyi/application/workflow/orchestration/plan.py",
            "start_line": 1,
            "end_line": 150,
        }

    def test_parses_multiple_xml_toolcalls(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<tool_call>houyi_read_file"
            "<arg_key>path</arg_key><arg_value>./houyi/domain/skill/schema.py</arg_value>"
            "</tool_call>"
            "<tool_call>houyi_read_file"
            "<arg_key>path</arg_key><arg_value>./houyi/domain/skill/spec.py</arg_value>"
            "</tool_call>"
        )
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0]["function"]["arguments"] == {
            "path": "./houyi/domain/skill/schema.py"
        }
        assert response.tool_calls[1]["function"]["arguments"] == {
            "path": "./houyi/domain/skill/spec.py"
        }

    def test_parses_bracket_toolcall(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "[tool:houyi_shell_exec] "
            '{"command": "find /Users/von/workspace/HouYiAgent -name \\"*.md\\"", '
            '"cwd": "/Users/von/workspace/HouYiAgent", "timeout_seconds": 30}'
        )
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-4"

        response = LLMResponse.from_openai(mock_response)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "houyi_shell_exec"
        assert response.tool_calls[0]["function"]["arguments"] == {
            "command": 'find /Users/von/workspace/HouYiAgent -name "*.md"',
            "cwd": "/Users/von/workspace/HouYiAgent",
            "timeout_seconds": 30,
        }

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

    def _fake_adapter(self, *, model: str = "gpt-3.5-turbo") -> OpenAIAdapter:
        """Build adapter with fake openai SDK (avoids heavy real import)."""
        pytest.importorskip("openai")
        client = MagicMock()
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=client)
        with patch.dict("sys.modules", {"openai": fake_openai}):
            return OpenAIAdapter(api_key="test-key", model=model)

    def test_adapter_with_apikey(self) -> None:
        """Test adapter initialization with API key."""
        adapter = self._fake_adapter()

        assert adapter.api_key == "test-key"
        assert adapter.model == "gpt-3.5-turbo"

    def test_adapter_without_apikey(self) -> None:
        """Test adapter initialization without API key raises error."""
        pytest.importorskip("openai")
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            with pytest.raises(ValueError, match="OpenAI API key not provided"):
                OpenAIAdapter()

    def test_adapter_from_env(self) -> None:
        """Test adapter initialization from environment variable."""
        pytest.importorskip("openai")
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock()
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            adapter = OpenAIAdapter()
            assert adapter.api_key == "env-key"

    @pytest.mark.asyncio
    async def test_chat(self) -> None:
        """Test chat method."""
        adapter = self._fake_adapter()

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
    async def test_chat_with_tools(self) -> None:
        """Test chat with tools."""
        adapter = self._fake_adapter()

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

    def test_normalize_messages(self) -> None:
        """Test message normalization."""
        pytest.importorskip("openai")
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

    def test_adapter_with_apikey(self) -> None:
        """Test adapter initialization with API key."""
        pytest.importorskip("anthropic")
        fake_anthropic = ModuleType("anthropic")
        fake_anthropic.AsyncAnthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            adapter = AnthropicAdapter(api_key="test-key", model="claude-3-opus")

        assert adapter.api_key == "test-key"
        assert adapter.model == "claude-3-opus"

    def test_adapter_without_apikey(self) -> None:
        """Test adapter initialization without API key raises error."""
        pytest.importorskip("anthropic")
        fake_anthropic = ModuleType("anthropic")
        fake_anthropic.AsyncAnthropic = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict("sys.modules", {"anthropic": fake_anthropic}),
        ):
            with pytest.raises(ValueError, match="Anthropic API key not provided"):
                AnthropicAdapter()

    def test_adapter_from_env(self) -> None:
        """Test adapter initialization from environment variable."""
        pytest.importorskip("anthropic")
        fake_anthropic = ModuleType("anthropic")
        fake_anthropic.AsyncAnthropic = MagicMock()
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}),
            patch.dict("sys.modules", {"anthropic": fake_anthropic}),
        ):
            adapter = AnthropicAdapter()
            assert adapter.api_key == "env-key"

    def test_normalize_messages(self) -> None:
        """Test message normalization."""
        pytest.importorskip("anthropic")
        fake_anthropic = ModuleType("anthropic")
        fake_anthropic.AsyncAnthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
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
