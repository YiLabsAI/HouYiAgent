"""Covers Gemini adapter conversion, auth-mode setup, and client error wrapping."""

from __future__ import annotations

import base64
import builtins
import os
import types
from unittest.mock import patch

import pytest

from houyi.adapters.llm import vertex_gemini_adapter as vertex_gemini_adapter_module
from houyi.adapters.llm.vertex_gemini_adapter import (
    GoogleVertexGeminiAdapter,
    _build_http_options,
    _build_proxy_http_options,
)


def _build_adapter() -> GoogleVertexGeminiAdapter:
    # Bypass __init__ to avoid optional runtime dependency.
    adapter = GoogleVertexGeminiAdapter.__new__(GoogleVertexGeminiAdapter)
    adapter.model = "test-model"
    adapter._auth_mode = "developer_api"
    adapter._proxy_url = None
    return adapter


class _FakeTypesPart:
    @staticmethod
    def from_text(*, text):
        return {"text": text}

    def __init__(self, **kwargs) -> None:
        self.payload = kwargs


class _FakeTypesFunctionCall:
    def __init__(self, *, id="", name, args) -> None:
        self.id = id
        self.name = name
        self.args = args


class _FakeTypesFunctionResponse:
    def __init__(self, *, id="", name, response) -> None:
        self.id = id
        self.name = name
        self.response = response


class _FakeTypesContent:
    def __init__(self, role=None, parts=None) -> None:
        self.role = role
        self.parts = parts


def _fake_vertex_types(**overrides):
    base = {
        "Content": _FakeTypesContent,
        "Part": _FakeTypesPart,
        "FunctionCall": _FakeTypesFunctionCall,
        "FunctionResponse": _FakeTypesFunctionResponse,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_convert_tools_schema() -> None:
    """Should convert OpenAI tool schema to Vertex function declarations."""

    adapter = _build_adapter()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    converted = adapter._convert_tools(tools)
    assert converted
    declarations = converted[0]["function_declarations"]
    assert declarations[0]["name"] == "weather"


def test_tool_choice_maps() -> None:
    """Should map tool_choice to Vertex function calling config."""

    adapter = _build_adapter()
    assert adapter._convert_tool_choice("required")["function_calling_config"]["mode"] == "ANY"
    assert adapter._convert_tool_choice("none")["function_calling_config"]["mode"] == "NONE"
    assert adapter._convert_tool_choice("auto")["function_calling_config"]["mode"] == "AUTO"


def test_normalize_extracts_tools() -> None:
    """Normalize should extract text and tool calls from response parts."""

    class _FunctionCall:
        def __init__(self) -> None:
            self.name = "weather"
            self.args = {"city": "Tokyo"}

    class _Part:
        def __init__(self) -> None:
            self.text = "hi"
            self.function_call = _FunctionCall()

    class _Content:
        def __init__(self) -> None:
            self.parts = [_Part()]

    class _Candidate:
        def __init__(self) -> None:
            self.content = _Content()

    class _Response:
        def __init__(self) -> None:
            self.candidates = [_Candidate()]
            self.usage_metadata = None

    adapter = _build_adapter()
    result = adapter._normalize_response(_Response())
    assert result.content == "hi"
    assert result.tool_calls[0]["function"]["name"] == "weather"


def test_response_with_token_usage() -> None:
    class _Part:
        def __init__(self) -> None:
            self.text = "hi"
            self.function_call = None

    class _Content:
        def __init__(self) -> None:
            self.parts = [_Part()]

    class _Candidate:
        def __init__(self) -> None:
            self.content = _Content()

    class _UsageMetadata:
        def __init__(self) -> None:
            self.prompt_token_count = 7
            self.candidates_token_count = 5
            self.total_token_count = 16
            self.thoughts_token_count = 3
            self.cached_content_token_count = 2
            self.prompt_tokens_details = [{"modality": "TEXT", "token_count": 7}]

    class _Response:
        def __init__(self) -> None:
            self.candidates = [_Candidate()]
            self.usage_metadata = _UsageMetadata()

    adapter = _build_adapter()
    result = adapter._normalize_response(_Response())

    assert result.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 16,
        "thinking_tokens": 3,
        "cache_read_input_tokens": 2,
    }


def test_normalize_keeps_signature() -> None:
    class _FunctionCall:
        def __init__(self) -> None:
            self.name = "houyi_shell_exec"
            self.args = {"command": "pwd"}

    class _Part:
        def __init__(self) -> None:
            self.text = ""
            self.function_call = _FunctionCall()
            self.thought_signature = b"sig-123"

    class _Content:
        def __init__(self) -> None:
            self.parts = [_Part()]

    class _Candidate:
        def __init__(self) -> None:
            self.content = _Content()

    class _Response:
        def __init__(self) -> None:
            self.candidates = [_Candidate()]
            self.usage_metadata = None

    adapter = _build_adapter()
    result = adapter._normalize_response(_Response())

    assert result.tool_calls[0]["function"]["name"] == "houyi_shell_exec"
    assert result.tool_calls[0]["function"]["thought_signature"] == base64.b64encode(
        b"sig-123"
    ).decode("ascii")


def test_normalize_uses_text() -> None:
    """Normalize should fall back to response text when candidate parts are absent."""

    class _Response:
        def __init__(self) -> None:
            self.candidates = []
            self.usage_metadata = None
            self.text = '{"scores": [9, 1]}'

    adapter = _build_adapter()
    result = adapter._normalize_response(_Response())

    assert result.content == '{"scores": [9, 1]}'
    assert result.tool_calls == []


def test_normalize_uses_parsed() -> None:
    """Normalize should stringify parsed structured output when present."""

    class _Response:
        def __init__(self) -> None:
            self.candidates = []
            self.usage_metadata = None
            self.parsed = {"city": "Tokyo", "score": 9}
            self.text = "ignored"

    adapter = _build_adapter()
    result = adapter._normalize_response(_Response())

    assert result.content == '{"city": "Tokyo", "score": 9}'
    assert result.tool_calls == []


def test_normalize_skips_text() -> None:
    class _FunctionCall:
        def __init__(self) -> None:
            self.id = "call_1"
            self.name = "weather"
            self.args = {"city": "Tokyo"}

    class _Part:
        def __init__(self) -> None:
            self.text = ""
            self.function_call = _FunctionCall()

    class _Content:
        def __init__(self) -> None:
            self.parts = [_Part()]

    class _Candidate:
        def __init__(self) -> None:
            self.content = _Content()
            self.text = None

    class _Response:
        def __init__(self) -> None:
            self.candidates = [_Candidate()]
            self.usage_metadata = None

        @property
        def text(self) -> str:
            raise AssertionError("response.text should not be accessed when candidates are present")

    adapter = _build_adapter()
    result = adapter._normalize_response(_Response())

    assert result.content == ""
    assert result.tool_calls[0]["id"] == "call_1"
    assert result.tool_calls[0]["function"]["name"] == "weather"


def test_extract_prefers_text() -> None:
    adapter = _build_adapter()

    class _Candidate:
        text = "direct text"
        content = None

    assert adapter._extract_candidate_content(_Candidate()) == "direct text"


def test_prepare_separates_system() -> None:
    adapter = _build_adapter()

    class _Part:
        @staticmethod
        def from_text(*, text):
            return {"text": text}

    class _Content:
        def __init__(self, role=None, parts=None) -> None:
            self.role = role
            self.parts = parts

    fake_types = types.SimpleNamespace(Content=_Content, Part=_Part)
    system_instruction, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )

    assert system_instruction == "be concise"
    assert len(contents) == 2
    assert contents[0].role == "user"
    assert contents[0].parts == [{"text": "hello"}]
    assert contents[1].role == "model"


def test_prepare_keeps_calls() -> None:
    adapter = _build_adapter()
    captured_parts: list[dict[str, object]] = []

    class _CapturingPart(_FakeTypesPart):
        def __init__(self, **kwargs) -> None:
            captured_parts.append(kwargs)
            super().__init__(**kwargs)

    fake_types = _fake_vertex_types(Part=_CapturingPart)
    system_instruction, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ],
            }
        ],
    )

    assert system_instruction is None
    assert len(contents) == 1
    assert contents[0].role == "model"
    assert len(contents[0].parts) == 1
    assert contents[0].parts[0].payload["function_call"].id == "call_1"
    assert contents[0].parts[0].payload["function_call"].name == "get_weather"
    assert contents[0].parts[0].payload["function_call"].args == {"city": "Tokyo"}
    assert len(captured_parts) == 1
    assert captured_parts[0]["function_call"].id == "call_1"


def test_prepare_roundtrips_signature() -> None:
    adapter = _build_adapter()
    captured_parts: list[dict[str, object]] = []

    class _CapturingPart(_FakeTypesPart):
        def __init__(self, **kwargs) -> None:
            captured_parts.append(kwargs)
            super().__init__(**kwargs)

    fake_types = _fake_vertex_types(Part=_CapturingPart)
    _, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "houyi_shell_exec",
                            "arguments": '{"command":"pwd"}',
                            "thought_signature": base64.b64encode(b"sig-xyz").decode("ascii"),
                        },
                    }
                ],
            }
        ],
    )

    assert len(contents) == 1
    assert contents[0].role == "model"
    assert len(captured_parts) == 1
    assert captured_parts[0]["thought_signature"] == b"sig-xyz"
    assert captured_parts[0]["function_call"].id == "call_1"
    assert captured_parts[0]["function_call"].name == "houyi_shell_exec"


def test_prepare_keeps_results() -> None:
    adapter = _build_adapter()
    fake_types = _fake_vertex_types()
    system_instruction, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "call_1",
                "content": '{"temperature":21}',
            }
        ],
    )

    assert system_instruction is None
    assert contents == []


def test_prepare_groups_results() -> None:
    adapter = _build_adapter()
    fake_types = _fake_vertex_types()
    _, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": '{"path":"/tmp"}'},
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "call_1",
                "content": '{"temperature":21}',
            },
            {
                "role": "tool",
                "name": "list_dir",
                "tool_call_id": "call_2",
                "content": '{"files":["a.txt"]}',
            },
        ],
    )

    assert len(contents) == 2
    assert contents[0].role == "model"
    assert len(contents[0].parts) == 2
    assert contents[1].role == "user"
    assert [part.payload["function_response"].id for part in contents[1].parts] == [
        "call_1",
        "call_2",
    ]
    assert contents[1].parts[0].payload["function_response"].response == {"temperature": 21}
    assert contents[1].parts[1].payload["function_response"].response == {"files": ["a.txt"]}


def test_prepare_recovers_name() -> None:
    adapter = _build_adapter()
    fake_types = _fake_vertex_types()
    _, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature":21}',
            },
        ],
    )

    assert len(contents) == 2
    assert contents[0].role == "model"
    assert contents[1].role == "user"
    assert len(contents[1].parts) == 1
    assert contents[1].parts[0].payload["function_response"].id == "call_1"
    assert contents[1].parts[0].payload["function_response"].name == "get_weather"
    assert contents[1].parts[0].payload["function_response"].response == {"temperature": 21}


def test_prepare_drops_group() -> None:
    adapter = _build_adapter()
    fake_types = _fake_vertex_types()
    _, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": '{"path":"/tmp"}'},
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "call_1",
                "content": '{"temperature":21}',
            },
        ],
    )

    assert len(contents) == 1
    assert contents[0].role == "model"
    assert len(contents[0].parts) == 2


def test_prepare_keeps_group() -> None:
    adapter = _build_adapter()
    fake_types = _fake_vertex_types()
    _, contents = adapter._prepare_contents_and_system(
        fake_types,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "old_call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                    },
                    {
                        "id": "old_call_2",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": '{"path":"/tmp"}'},
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "old_call_1",
                "content": '{"temperature":21}',
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "new_call_1",
                        "type": "function",
                        "function": {"name": "pwd", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "pwd",
                "tool_call_id": "new_call_1",
                "content": '{"cwd":"/tmp"}',
            },
        ],
    )

    assert len(contents) == 3
    assert contents[0].role == "model"
    assert len(contents[0].parts) == 2
    assert contents[1].role == "model"
    assert len(contents[1].parts) == 1
    assert contents[2].role == "user"
    assert len(contents[2].parts) == 1
    assert contents[2].parts[0].payload["function_response"].id == "new_call_1"
    assert contents[2].parts[0].payload["function_response"].name == "pwd"


def test_config_sets_system() -> None:
    adapter = _build_adapter()

    class _Config:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(GenerateContentConfig=_Config)
    config = adapter._build_generate_config(
        fake_types,
        temperature=0.1,
        max_tokens=8,
        tools=None,
        system_instruction="be concise",
        extra_kwargs={"response_mime_type": "application/json"},
    )

    assert config.kwargs["temperature"] == 0.1
    assert config.kwargs["max_output_tokens"] == 8
    assert config.kwargs["system_instruction"] == "be concise"
    assert config.kwargs["response_mime_type"] == "application/json"
    assert config.kwargs["tool_config"]["function_calling_config"]["mode"] == "NONE"


def test_config_ignores_model() -> None:
    adapter = _build_adapter()

    class _Config:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(GenerateContentConfig=_Config)
    config = adapter._build_generate_config(
        fake_types,
        temperature=0.1,
        max_tokens=8,
        tools=None,
        system_instruction=None,
        extra_kwargs={"model": "gemini-3.1-pro-preview"},
    )
    assert "model" not in config.kwargs


def test_config_ignores_parallel() -> None:
    adapter = _build_adapter()

    class _Config:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(GenerateContentConfig=_Config)
    config = adapter._build_generate_config(
        fake_types,
        temperature=0.1,
        max_tokens=8,
        tools=None,
        system_instruction=None,
        extra_kwargs={"parallel_tool_calls": True},
    )
    assert "parallel_tool_calls" not in config.kwargs


def test_ignores_openai_usage_flags() -> None:
    adapter = _build_adapter()

    class _Config:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(GenerateContentConfig=_Config)
    config = adapter._build_generate_config(
        fake_types,
        temperature=0.1,
        max_tokens=8,
        tools=None,
        system_instruction=None,
        extra_kwargs={
            "include_stream_usage": False,
            "stream_options": {"include_usage": True},
        },
    )
    assert "include_stream_usage" not in config.kwargs
    assert "stream_options" not in config.kwargs


@pytest.mark.asyncio
async def test_stream_skips_text() -> None:
    adapter = _build_adapter()

    class _Part:
        def __init__(self) -> None:
            self.text = ""
            self.function_call = types.SimpleNamespace(
                id="", name="weather", args={"city": "Tokyo"}
            )
            self.thought_signature = None

    class _CandidateContent:
        def __init__(self) -> None:
            self.parts = [_Part()]

    class _Candidate:
        def __init__(self) -> None:
            self.content = _CandidateContent()

    class _Chunk:
        def __init__(self) -> None:
            self.usage_metadata = None
            self.candidates = [_Candidate()]

        @property
        def text(self) -> str:
            raise AssertionError("chunk.text should not be accessed when parts are available")

    class _Stream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return _Chunk()

    captured_configs: list[object] = []

    class _GenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.tool_config = types.SimpleNamespace(
                function_calling_config=types.SimpleNamespace(
                    mode=kwargs.get("tool_config", {})
                    .get("function_calling_config", {})
                    .get("mode")
                )
            )

    fake_types = types.SimpleNamespace(
        Content=_FakeTypesContent,
        Part=_FakeTypesPart,
        GenerateContentConfig=_GenerateContentConfig,
    )

    class _Models:
        async def generate_content_stream(self, *, model, contents, config):
            _ = (model, contents)
            captured_configs.append(config)
            return _Stream()

    adapter._client = types.SimpleNamespace(aio=types.SimpleNamespace(models=_Models()))

    chunks = []
    with patch.dict("sys.modules", {"google.genai": types.SimpleNamespace(types=fake_types)}):
        async for chunk in adapter.stream_chat([{"role": "user", "content": "hello"}], tools=None):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].content_delta == ""
    assert chunks[0].tool_calls_delta is not None
    assert chunks[0].tool_calls_delta[0]["id"] == "gemini_call_0_weather"
    config = captured_configs[0]
    assert config.tool_config.function_calling_config.mode == "NONE"


def test_choice_dict_maps() -> None:
    adapter = _build_adapter()
    assert (
        adapter._convert_tool_choice({"type": "function"})["function_calling_config"]["mode"]
        == "ANY"
    )


def test_convert_tools_empty() -> None:
    """Empty or invalid tools should return empty list."""

    adapter = _build_adapter()
    assert adapter._convert_tools([]) == []
    assert adapter._convert_tools([{"type": "noop"}]) == []


def test_from_env_default(monkeypatch) -> None:
    """from_env should read configuration from environment variables."""

    created: dict[str, object] = {}

    def _fake_init(
        self, *, model, api_key=None, project=None, location="us-central1", credentials_path=None
    ) -> None:
        created["project"] = project
        created["location"] = location
        created["model"] = model
        created["api_key"] = api_key

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "loc")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(GoogleVertexGeminiAdapter, "__init__", _fake_init, raising=True)

    from houyi.infrastructure.config.env_config import EnvConfig

    EnvConfig._reset()
    try:
        GoogleVertexGeminiAdapter.from_env()
    finally:
        EnvConfig._reset()
    assert created["project"] == "proj"
    assert created["location"] == "loc"
    assert created["model"] == "gemini-test"


def test_from_env_key(monkeypatch) -> None:
    """from_env should use GOOGLE_API_KEY when set."""

    created: dict[str, object] = {}

    def _fake_init(
        self, *, model, api_key=None, project=None, location="us-central1", credentials_path=None
    ) -> None:
        created["api_key"] = api_key
        created["model"] = model

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(GoogleVertexGeminiAdapter, "__init__", _fake_init, raising=True)

    from houyi.infrastructure.config.env_config import EnvConfig

    EnvConfig._reset()
    try:
        GoogleVertexGeminiAdapter.from_env()
    finally:
        EnvConfig._reset()
    assert created["api_key"] == "test-key"


def test_from_env_auth(monkeypatch) -> None:
    """from_env should raise when neither API key nor project is set."""

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    from houyi.infrastructure.config.env_config import EnvConfig

    EnvConfig._reset()
    try:
        with pytest.raises(ValueError, match="Either GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY"):
            GoogleVertexGeminiAdapter.from_env()
    finally:
        EnvConfig._reset()


@pytest.mark.asyncio
async def test_stream_chat() -> None:
    """stream_chat should yield content from streaming response."""
    adapter = _build_adapter()

    class _Chunk:
        def __init__(self, text: str, usage_metadata=None) -> None:
            self.text = text
            self.usage_metadata = usage_metadata

    class _UsageMetadata:
        def __init__(self) -> None:
            self.prompt_token_count = 4
            self.candidates_token_count = 2
            self.total_token_count = 6
            self.thoughts_token_count = 3
            self.cached_content_token_count = 1
            self.prompt_tokens_details = [{"modality": "TEXT", "token_count": 4}]

    async def _fake_stream(model=None, contents=None, config=None):
        yield _Chunk("hello")
        yield _Chunk(" world", usage_metadata=_UsageMetadata())

    class _Models:
        async def generate_content_stream(self, model=None, contents=None, config=None):
            return _fake_stream(model, contents, config)

    class _Aio:
        def __init__(self) -> None:
            self.models = _Models()

    adapter._client = type("FakeClient", (), {"aio": _Aio()})()

    fake_types = types.SimpleNamespace(
        Content=_FakeTypesContent,
        Part=_FakeTypesPart,
        GenerateContentConfig=lambda **kw: types.SimpleNamespace(**kw),
    )
    fake_genai = types.SimpleNamespace(types=fake_types)
    with patch.dict("sys.modules", {"google.genai": fake_genai}):
        chunks = []
        async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}]):
            chunks.append((chunk.content_delta, chunk.reasoning_delta))

    assert chunks == [("hello", None), (" world", None)]
    assert adapter.last_usage == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
        "thinking_tokens": 3,
        "cache_read_input_tokens": 1,
    }
    assert adapter.last_finish_reason == "stop"


@pytest.mark.asyncio
async def test_no_visible_output(caplog) -> None:
    adapter = _build_adapter()

    class _UsageMetadata:
        def __init__(self) -> None:
            self.prompt_token_count = 4
            self.candidates_token_count = 0
            self.total_token_count = 4
            self.thoughts_token_count = 0
            self.cached_content_token_count = 0
            self.prompt_tokens_details = None

    class _Candidate:
        def __init__(self) -> None:
            self.finish_reason = "STOP"
            self.content = types.SimpleNamespace(parts=[])

    class _Chunk:
        def __init__(self) -> None:
            self.candidates = [_Candidate()]
            self.usage_metadata = _UsageMetadata()

    async def _fake_stream(model=None, contents=None, config=None):
        _ = (model, contents, config)
        yield _Chunk()

    class _Models:
        async def generate_content_stream(self, model=None, contents=None, config=None):
            return _fake_stream(model, contents, config)

    class _Aio:
        def __init__(self) -> None:
            self.models = _Models()

    adapter._client = type("FakeClient", (), {"aio": _Aio()})()

    fake_types = types.SimpleNamespace(
        Content=_FakeTypesContent,
        Part=_FakeTypesPart,
        GenerateContentConfig=lambda **kw: types.SimpleNamespace(**kw),
    )
    fake_genai = types.SimpleNamespace(types=fake_types)
    with patch.dict("sys.modules", {"google.genai": fake_genai}):
        with caplog.at_level("WARNING"):
            chunks = [
                chunk async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}])
            ]

    assert chunks == []
    assert adapter.last_finish_reason == "stop"
    assert "Gemini stream completed without visible output" in caplog.text


def test_init_sets_env(monkeypatch) -> None:
    created = {}

    class _Client:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

    class _HttpRetryOptions:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class _HttpOptions:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    fake_types = types.SimpleNamespace(
        HttpRetryOptions=_HttpRetryOptions,
        HttpOptions=_HttpOptions,
    )
    fake_genai = types.SimpleNamespace(Client=_Client, types=fake_types)
    fake_google = types.SimpleNamespace(genai=fake_genai)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "preexisting.json")
    monkeypatch.setattr(vertex_gemini_adapter_module, "detect_proxy", lambda: None)

    with patch.dict("sys.modules", {"google": fake_google, "google.genai": fake_genai}):
        adapter = GoogleVertexGeminiAdapter(
            model="gemini-test",
            project="proj",
            location="asia-east1",
            credentials_path="new-creds.json",
        )

    assert adapter._auth_mode == "vertex_ai"
    assert created["vertexai"] is True
    assert created["project"] == "proj"
    assert created["location"] == "asia-east1"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "preexisting.json"
    retry_options = created["http_options"].retry_options
    assert retry_options.attempts == 5
    assert retry_options.http_status_codes == [408, 429, 500, 502, 503, 504]


def test_http_options_retry() -> None:
    class _HttpRetryOptions:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class _HttpOptions:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    fake_types = types.SimpleNamespace(
        HttpRetryOptions=_HttpRetryOptions,
        HttpOptions=_HttpOptions,
    )

    options = _build_http_options(fake_types, proxy_url=None)

    assert options.retry_options.attempts == 5
    assert options.retry_options.initial_delay == 1.0
    assert options.retry_options.max_delay == 60.0
    assert options.retry_options.exp_base == 2.0
    assert options.retry_options.jitter == 1.0
    assert options.retry_options.http_status_codes == [408, 429, 500, 502, 503, 504]


def test_init_raises() -> None:
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google":
            raise ImportError("missing google")

    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(
            ImportError,
            match=r"Google GenAI client not installed\. Install with: pip install google\-genai",
        ):
            GoogleVertexGeminiAdapter(model="gemini-test", api_key="test-key")


@pytest.mark.asyncio
async def test_chat_wraps_errors() -> None:
    class _Part:
        @staticmethod
        def from_text(*, text):
            return {"text": text}

    class _Content:
        def __init__(self, role=None, parts=None) -> None:
            self.role = role
            self.parts = parts

    class _Config:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(Content=_Content, Part=_Part, GenerateContentConfig=_Config)
    adapter = _build_adapter()

    class _Models:
        async def generate_content(self, **kwargs):
            raise Exception("401 UNAUTHENTICATED")

    adapter._client = type("FakeClient", (), {"aio": type("Aio", (), {"models": _Models()})()})()

    with patch.dict("sys.modules", {"google.genai": types.SimpleNamespace(types=fake_types)}):
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            await adapter.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_chat_rate_limit_errors() -> None:
    class _Part:
        @staticmethod
        def from_text(*, text):
            return {"text": text}

    class _Content:
        def __init__(self, role=None, parts=None) -> None:
            self.role = role
            self.parts = parts

    class _Config:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(Content=_Content, Part=_Part, GenerateContentConfig=_Config)
    adapter = _build_adapter()

    class _Models:
        async def generate_content(self, **kwargs):
            raise Exception("429 RESOURCE_EXHAUSTED")

    adapter._client = type("FakeClient", (), {"aio": type("Aio", (), {"models": _Models()})()})()

    with patch.dict("sys.modules", {"google.genai": types.SimpleNamespace(types=fake_types)}):
        with pytest.raises(RuntimeError, match="retry"):
            await adapter.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_stream_wraps_errors() -> None:
    class _Part:
        @staticmethod
        def from_text(*, text):
            return {"text": text}

    class _Content:
        def __init__(self, role=None, parts=None) -> None:
            self.role = role
            self.parts = parts

    class _Config:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(Content=_Content, Part=_Part, GenerateContentConfig=_Config)
    adapter = _build_adapter()

    class _Models:
        async def generate_content_stream(self, **kwargs):
            raise Exception("404 NOT_FOUND")

    adapter._client = type("FakeClient", (), {"aio": type("Aio", (), {"models": _Models()})()})()

    with patch.dict("sys.modules", {"google.genai": types.SimpleNamespace(types=fake_types)}):
        with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
            async for _ in adapter.stream_chat([{"role": "user", "content": "hi"}]):
                pass


class TestBuildProxyHttpOptions:
    """Tests for _build_proxy_http_options."""

    def test_creates_clients_proxy(self) -> None:
        opts = _build_proxy_http_options("http://127.0.0.1:7890")
        assert "httpx_client" in opts
        assert "httpx_async_client" in opts

    def test_clients_are_httpx(self) -> None:
        import httpx

        opts = _build_proxy_http_options("http://127.0.0.1:8118")
        assert isinstance(opts["httpx_client"], httpx.Client)
        assert isinstance(opts["httpx_async_client"], httpx.AsyncClient)


class TestWrapClientErrorRegion:
    """_wrap_client_error should explain account-level region restriction."""

    def test_region_error_vpn(self) -> None:
        adapter = _build_adapter()
        adapter._proxy_url = "http://127.0.0.1:8118"
        exc = adapter._wrap_client_error(
            Exception("400 FAILED_PRECONDITION: User location is not supported")
        )
        msg = str(exc)
        assert "VPN/proxy/datacenter" in msg
        assert "Vertex AI" in msg

    def test_region_error_vertex(self) -> None:
        adapter = _build_adapter()
        adapter._auth_mode = "vertex_ai"
        adapter._proxy_url = None
        exc = adapter._wrap_client_error(
            Exception("400 FAILED_PRECONDITION: User location is not supported")
        )
        msg = str(exc)
        assert "GOOGLE_CLOUD_LOCATION" in msg

    def test_401_error_creds(self) -> None:
        adapter = _build_adapter()
        exc = adapter._wrap_client_error(Exception("401 UNAUTHENTICATED"))
        msg = str(exc)
        assert "GOOGLE_API_KEY" in msg
        assert "AIza" in msg

    def test_403_error_perms(self) -> None:
        adapter = _build_adapter()
        exc = adapter._wrap_client_error(Exception("403 PERMISSION_DENIED"))
        assert "aiplatform.user role" in str(exc)

    def test_404_error_model(self) -> None:
        adapter = _build_adapter()
        exc = adapter._wrap_client_error(Exception("404 NOT_FOUND"))
        assert "GEMINI_MODEL" in str(exc)

    def test_other_error_context(self) -> None:
        adapter = _build_adapter()
        exc = adapter._wrap_client_error(Exception("socket closed"))
        msg = str(exc)
        assert "developer_api" in msg
        assert "test-model" in msg
        assert "socket closed" in msg
