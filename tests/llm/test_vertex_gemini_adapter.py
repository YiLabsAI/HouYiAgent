"""Unit tests for GoogleVertexGeminiAdapter conversions."""

from __future__ import annotations

import pytest

from houyi.llm.vertex_gemini_adapter import (
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

    from houyi.config.env_config import EnvConfig

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

    from houyi.config.env_config import EnvConfig

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

    from houyi.config.env_config import EnvConfig

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
    )
    assert result.tool_calls
    config = captured["config"]
    assert config.max_output_tokens == 5
    assert config.tool_config.function_calling_config.mode == "ANY"


@pytest.mark.asyncio
async def test_vertex_gemini_stream_chat() -> None:
    """stream_chat should yield content from streaming response."""
    pytest.importorskip("google.genai")

    adapter = _build_adapter()

    class _Chunk:
        def __init__(self, text: str) -> None:
            self.text = text

    async def _fake_stream(model=None, contents=None, config=None):
        for text in ["hello", " world"]:
            yield _Chunk(text)

    class _Models:
        async def generate_content_stream(self, model=None, contents=None, config=None):
            return _fake_stream(model, contents, config)

    class _Aio:
        def __init__(self) -> None:
            self.models = _Models()

    adapter._client = type("FakeClient", (), {"aio": _Aio()})()

    chunks = []
    async for content, reasoning in adapter.stream_chat([{"role": "user", "content": "hi"}]):
        chunks.append((content, reasoning))

    assert chunks == [("hello", None), (" world", None)]


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
