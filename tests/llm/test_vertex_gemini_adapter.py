"""Covers Gemini SDK adapter conversion, auth-mode setup, and SDK error wrapping."""

from __future__ import annotations

import builtins
import os
import types
from unittest.mock import patch

import pytest

from houyi.adapters.llm.vertex_gemini_adapter import (
    GoogleVertexGeminiAdapter,
    _build_proxy_http_options,
)


def _build_adapter() -> GoogleVertexGeminiAdapter:
    # Bypass __init__ to avoid optional runtime dependency.
    adapter = GoogleVertexGeminiAdapter.__new__(GoogleVertexGeminiAdapter)
    adapter.model = "test-model"
    adapter._auth_mode = "developer_api"
    adapter._proxy_url = None
    return adapter


def test_vertex_gemini_tool_schema_conversion() -> None:
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


def test_vertex_gemini_tool_choice_conversion() -> None:
    """Should map tool_choice to Vertex function calling config."""

    adapter = _build_adapter()
    assert adapter._convert_tool_choice("required")["function_calling_config"]["mode"] == "ANY"
    assert adapter._convert_tool_choice("none")["function_calling_config"]["mode"] == "NONE"
    assert adapter._convert_tool_choice("auto")["function_calling_config"]["mode"] == "AUTO"


def test_vertex_gemini_normalize_response_with_tool_calls() -> None:
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


def test_vertex_gemini_normalize_response_without_parts_uses_text() -> None:
    """Normalize should fall back to response text when candidate parts are absent."""

    class _Content:
        def __init__(self) -> None:
            self.parts = None

    class _Candidate:
        def __init__(self) -> None:
            self.content = _Content()
            self.text = None

    class _Response:
        def __init__(self) -> None:
            self.candidates = [_Candidate()]
            self.usage_metadata = None
            self.text = '{"scores": [9, 1]}'

    adapter = _build_adapter()
    result = adapter._normalize_response(_Response())

    assert result.content == '{"scores": [9, 1]}'
    assert result.tool_calls == []


def test_vertex_gemini_normalize_response_uses_parsed_object_string() -> None:
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


def test_vertex_gemini_extract_candidate_content_prefers_candidate_text() -> None:
    adapter = _build_adapter()

    class _Candidate:
        text = "direct text"
        content = None

    assert adapter._extract_candidate_content(_Candidate()) == "direct text"


def test_vertex_gemini_prepare_contents_and_system_separates_system_message() -> None:
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


def test_vertex_gemini_build_generate_config_sets_system_instruction() -> None:
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


def test_vertex_gemini_tool_choice_dict_maps_to_any() -> None:
    adapter = _build_adapter()
    assert (
        adapter._convert_tool_choice({"type": "function"})["function_calling_config"]["mode"]
        == "ANY"
    )


def test_vertex_gemini_convert_tools_empty() -> None:
    """Empty or invalid tools should return empty list."""

    adapter = _build_adapter()
    assert adapter._convert_tools([]) == []
    assert adapter._convert_tools([{"type": "noop"}]) == []


def test_vertex_gemini_from_env(monkeypatch) -> None:
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


def test_vertex_gemini_from_env_api_key(monkeypatch) -> None:
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


def test_vertex_gemini_from_env_requires_auth(monkeypatch) -> None:
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
async def test_vertex_gemini_chat_builds_config() -> None:
    """chat should build config and pass tools/tool_choice to generate_content."""
    pytest.importorskip("google.genai")

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

    class _UsageMetadata:
        def __init__(self) -> None:
            self.prompt_token_count = 10
            self.candidates_token_count = 5
            self.total_token_count = 15

    class _Response:
        def __init__(self) -> None:
            self.candidates = [_Candidate()]
            self.usage_metadata = _UsageMetadata()

    captured: dict[str, object] = {}

    class _Models:
        async def generate_content(self, model=None, contents=None, config=None):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return _Response()

    class _Aio:
        def __init__(self) -> None:
            self.models = _Models()

    adapter = _build_adapter()
    adapter._client = type("FakeClient", (), {"aio": _Aio()})()

    result = await adapter.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "weather", "parameters": {}}}],
        tool_choice="required",
        max_tokens=5,
        response_mime_type="application/json",
    )
    assert result.tool_calls
    config = captured["config"]
    assert config.max_output_tokens == 5
    assert config.tool_config.function_calling_config.mode == "ANY"
    assert config.response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_vertex_gemini_stream_chat() -> None:
    """stream_chat should yield content from streaming response."""
    pytest.importorskip("google.genai")

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

    chunks = []
    async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}]):
        chunks.append((chunk.content_delta, chunk.reasoning_delta))

    assert chunks == [("hello", None), (" world", None)]
    assert adapter.last_usage == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    assert adapter.last_finish_reason == "stop"


def test_vertex_gemini_init_vertex_ai_mode_sets_credentials_env(monkeypatch) -> None:
    created = {}

    class _Client:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

    fake_genai = types.SimpleNamespace(Client=_Client)
    fake_google = types.SimpleNamespace(genai=fake_genai)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "preexisting.json")
    monkeypatch.setattr(
        "houyi.adapters.llm.vertex_gemini_adapter.detect_proxy",
        lambda: None,
    )

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


def test_vertex_gemini_init_raises_import_error_when_sdk_missing():
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google":
            raise ImportError("missing google")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(ImportError, match="Google GenAI SDK not installed"):
            GoogleVertexGeminiAdapter(model="gemini-test", api_key="test-key")


@pytest.mark.asyncio
async def test_vertex_gemini_chat_wraps_sdk_errors() -> None:
    pytest.importorskip("google.genai")

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
async def test_vertex_gemini_stream_wraps_sdk_errors() -> None:
    pytest.importorskip("google.genai")

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

    def test_creates_httpx_clients_with_proxy(self):
        opts = _build_proxy_http_options("http://127.0.0.1:7890")
        assert "httpx_client" in opts
        assert "httpx_async_client" in opts

    def test_clients_are_httpx_instances(self):
        import httpx

        opts = _build_proxy_http_options("http://127.0.0.1:8118")
        assert isinstance(opts["httpx_client"], httpx.Client)
        assert isinstance(opts["httpx_async_client"], httpx.AsyncClient)


class TestWrapSdkErrorRegion:
    """_wrap_sdk_error should explain account-level region restriction."""

    def test_region_error_explains_vpn_restriction(self):
        adapter = _build_adapter()
        adapter._proxy_url = "http://127.0.0.1:8118"
        exc = adapter._wrap_sdk_error(
            Exception("400 FAILED_PRECONDITION: User location is not supported")
        )
        msg = str(exc)
        assert "VPN/proxy/datacenter" in msg
        assert "Vertex AI" in msg

    def test_region_error_vertex_ai_mode(self):
        adapter = _build_adapter()
        adapter._auth_mode = "vertex_ai"
        adapter._proxy_url = None
        exc = adapter._wrap_sdk_error(
            Exception("400 FAILED_PRECONDITION: User location is not supported")
        )
        msg = str(exc)
        assert "GOOGLE_CLOUD_LOCATION" in msg

    def test_401_error_explains_developer_api_credentials(self):
        adapter = _build_adapter()
        exc = adapter._wrap_sdk_error(Exception("401 UNAUTHENTICATED"))
        msg = str(exc)
        assert "GOOGLE_API_KEY" in msg
        assert "AIza" in msg

    def test_403_error_explains_permissions(self):
        adapter = _build_adapter()
        exc = adapter._wrap_sdk_error(Exception("403 PERMISSION_DENIED"))
        assert "aiplatform.user role" in str(exc)

    def test_404_error_explains_model_config(self):
        adapter = _build_adapter()
        exc = adapter._wrap_sdk_error(Exception("404 NOT_FOUND"))
        assert "GEMINI_MODEL" in str(exc)

    def test_other_error_preserves_auth_mode_and_model_context(self):
        adapter = _build_adapter()
        exc = adapter._wrap_sdk_error(Exception("socket closed"))
        msg = str(exc)
        assert "developer_api" in msg
        assert "test-model" in msg
        assert "socket closed" in msg
