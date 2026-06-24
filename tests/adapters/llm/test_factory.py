from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import houyi.adapters.llm.factory as factory_module
import houyi.adapters.llm.siliconflow_adapter as sf_module
from houyi.adapters.llm.factory import (
    LLMAdapterFactory,
    _create_dashscope_adapter,
    _create_deepseek_adapter,
    _create_vertex_adapter,
)
from houyi.adapters.llm.models import (
    PROVIDER_DASHSCOPE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GOOGLE_AI,
    PROVIDER_OPENAI_COMPAT,
    PROVIDER_SILICONFLOW,
    PROVIDER_VERTEX,
)
from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.infrastructure.config.env_config import (
    ENV_DASHSCOPE_API_KEY,
    ENV_DASHSCOPE_BASE_URL,
    ENV_DASHSCOPE_MODEL,
    ENV_DEEPSEEK_API_KEY,
    ENV_DEEPSEEK_BASE_URL,
    ENV_DEEPSEEK_MODEL,
    ENV_DEFAULT_LLM_PROVIDER,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    ENV_QWEN_MODEL,
    ENV_SILICONFLOW_MODEL,
    ENV_TOOLCALL_MODEL,
    EnvConfig,
)


@pytest.fixture(autouse=True, scope="module")
def _reset_env_config():
    """Reset EnvConfig singleton once per module for faster test execution.

    Using module scope avoids repeated resets between tests in the same file,
    saving ~50-100ms per test while still ensuring isolation across modules.
    """
    EnvConfig._reset()
    yield
    EnvConfig._reset()


class TestLLMAdapterFactory:
    """Test LLMAdapterFactory.create."""

    def test_default_siliconflow(self):
        """Default provider routes to SiliconFlowAdapter."""
        fake_adapter = MagicMock(spec=SiliconFlowAdapter)
        with patch.object(sf_module, "SiliconFlowAdapter", return_value=fake_adapter):
            with patch.dict(
                os.environ, {ENV_DEFAULT_LLM_PROVIDER: PROVIDER_SILICONFLOW}, clear=False
            ):
                adapter = LLMAdapterFactory.create()
        assert isinstance(adapter, MagicMock)

    def test_vertex(self):
        """Vertex provider routes through _create_vertex_adapter."""
        fake_adapter = SimpleNamespace(stream_completion=lambda: None, stream_chat=lambda: None)
        with patch.object(factory_module, "_create_vertex_adapter", return_value=fake_adapter):
            adapter = LLMAdapterFactory.create(PROVIDER_VERTEX)
        assert hasattr(adapter, "stream_completion")
        assert hasattr(adapter, "stream_chat")

    def test_deepseek(self):
        """deepseek provider creates OpenAI-compatible adapter."""
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock()
        with (
            patch.dict(os.environ, {ENV_DEEPSEEK_API_KEY: "test-key"}),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            adapter = LLMAdapterFactory.create(PROVIDER_DEEPSEEK)
            assert hasattr(adapter, "chat")

    def test_openai_compat(self):
        """openai_compat provider creates OpenAI-compatible adapter."""
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock()
        with (
            patch.dict(os.environ, {ENV_OPENAI_API_KEY: "test-key"}),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            adapter = LLMAdapterFactory.create(PROVIDER_OPENAI_COMPAT)
            assert hasattr(adapter, "chat")

    def test_dashscope(self):
        """dashscope provider creates OpenAI-compatible adapter."""
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock()
        with (
            patch.dict(os.environ, {ENV_DASHSCOPE_API_KEY: "bailian-key"}, clear=False),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            EnvConfig._reset()
            adapter = LLMAdapterFactory.create(PROVIDER_DASHSCOPE)
            EnvConfig._reset()
        assert hasattr(adapter, "chat")

    def test_unknown_falls_back(self):
        fake_adapter = MagicMock(spec=SiliconFlowAdapter)
        with patch.object(sf_module, "SiliconFlowAdapter", return_value=fake_adapter):
            adapter = LLMAdapterFactory.create("unknown_provider")
        assert isinstance(adapter, MagicMock)

    def test_none_uses_env(self):
        fake_adapter = SimpleNamespace(stream_completion=lambda: None, stream_chat=lambda: None)
        with (
            patch.dict(os.environ, {ENV_DEFAULT_LLM_PROVIDER: PROVIDER_VERTEX}),
            patch.object(factory_module, "_create_vertex_adapter", return_value=fake_adapter),
        ):
            adapter = LLMAdapterFactory.create()
        assert hasattr(adapter, "stream_completion")

    def test_google_routes_vertex(self):
        fake_adapter = SimpleNamespace(stream_chat=lambda: None, stream_completion=lambda: None)
        with patch.object(factory_module, "_create_vertex_adapter", return_value=fake_adapter):
            adapter = LLMAdapterFactory.create(PROVIDER_GOOGLE_AI)
        assert hasattr(adapter, "stream_chat")


def test_vertex_uses_httpx(monkeypatch):
    fake_env = SimpleNamespace(
        google_project="proj",
        google_api_key=None,
        gemini_model="gemini-test",
        google_credentials_path=None,
        google_project_id="proj",
        google_location="us-central1",
    )
    monkeypatch.setattr(EnvConfig, "get", lambda: fake_env)

    original_import = __import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "houyi.adapters.llm.vertex_gemini_adapter":
            raise ImportError("missing google-genai")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_fake_import):
        adapter = _create_vertex_adapter()

    assert adapter.__class__.__name__ == "VertexAIAdapter"


def test_vertex_rejects_env(monkeypatch):
    class _BadAdapter:
        @staticmethod
        def from_env():
            raise ValueError("bad config")

    fake_env = SimpleNamespace(
        google_project="proj",
        google_api_key=None,
        gemini_model="gemini-test",
        google_credentials_path=None,
        google_project_id="proj",
        google_location="us-central1",
    )
    monkeypatch.setattr(EnvConfig, "get", lambda: fake_env)
    fake_module = ModuleType("houyi.adapters.llm.vertex_gemini_adapter")
    fake_module.GoogleVertexGeminiAdapter = _BadAdapter

    with patch.dict(sys.modules, {"houyi.adapters.llm.vertex_gemini_adapter": fake_module}):
        adapter = _create_vertex_adapter()

    assert adapter.__class__.__name__ == "VertexAIAdapter"


def test_vertex_forces_httpx(monkeypatch):
    fake_env = SimpleNamespace(
        google_project="proj",
        google_api_key="api-key",
        gemini_model="gemini-test",
        google_credentials_path=None,
        google_project_id="proj",
        google_location="us-central1",
    )
    monkeypatch.setattr(EnvConfig, "get", lambda: fake_env)

    with patch.dict(os.environ, {"HOUYI_VERTEX_ADAPTER": "httpx"}, clear=False):
        adapter = _create_vertex_adapter()

    assert adapter.__class__.__name__ == "VertexAIAdapter"


def test_vertex_forces_genai(monkeypatch):
    class _FakeGeminiAdapter:
        @staticmethod
        def from_env():
            return "forced-genai"

    fake_env = SimpleNamespace(
        google_project=None,
        google_api_key=None,
        gemini_model="gemini-test",
        google_credentials_path=None,
        google_project_id=None,
        google_location="us-central1",
    )
    monkeypatch.setattr(EnvConfig, "get", lambda: fake_env)
    fake_module = ModuleType("houyi.adapters.llm.vertex_gemini_adapter")
    fake_module.GoogleVertexGeminiAdapter = _FakeGeminiAdapter

    with (
        patch.dict(os.environ, {"HOUYI_VERTEX_ADAPTER": "genai"}, clear=False),
        patch.dict(sys.modules, {"houyi.adapters.llm.vertex_gemini_adapter": fake_module}),
    ):
        adapter = _create_vertex_adapter()

    assert adapter == "forced-genai"


def test_dashscope_uses_keyed_env(monkeypatch):
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(
            os.environ,
            {
                ENV_DASHSCOPE_API_KEY: "bailian-key",
                ENV_DASHSCOPE_BASE_URL: "https://dashscope.example/compatible-mode/v1",
                ENV_DASHSCOPE_MODEL: "glm-5.1",
            },
            clear=False,
        ),
    ):
        EnvConfig._reset()
        adapter = _create_dashscope_adapter()
        EnvConfig._reset()

    assert adapter.api_key == "bailian-key"
    assert adapter.base_url == "https://dashscope.example/compatible-mode/v1"
    assert adapter.model == "glm-5.1"


def test_dashscope_missing_key_raises(monkeypatch):
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(os.environ, {ENV_DASHSCOPE_API_KEY: ""}, clear=False),
    ):
        EnvConfig._reset()
        with pytest.raises(ValueError):
            _create_dashscope_adapter()
        EnvConfig._reset()


def test_dashscope_override_wins(monkeypatch):
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(
            os.environ,
            {
                ENV_DASHSCOPE_API_KEY: "env-key",
                ENV_DASHSCOPE_BASE_URL: "https://env.example/v1",
                ENV_DASHSCOPE_MODEL: "glm-5.1",
            },
            clear=False,
        ),
    ):
        EnvConfig._reset()
        adapter = _create_dashscope_adapter(
            model="qwen3.7-max",
            api_key="explicit-key",
            base_url="https://explicit.example/v1",
        )
        EnvConfig._reset()

    assert adapter.api_key == "explicit-key"
    assert adapter.base_url == "https://explicit.example/v1"
    assert adapter.model == "qwen3.7-max"


def test_dashscope_explicit_key(monkeypatch):
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(os.environ, {ENV_DASHSCOPE_API_KEY: ""}, clear=False),
    ):
        EnvConfig._reset()
        adapter = _create_dashscope_adapter(api_key="explicit-key")
        EnvConfig._reset()

    assert adapter.api_key == "explicit-key"


def test_create_siliconflow_model_override(monkeypatch):
    captured = {}

    def _fake_ctor(*, api_key=None, base_url=None, default_model=None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["default_model"] = default_model
        return MagicMock(spec=SiliconFlowAdapter)

    with patch.object(sf_module, "SiliconFlowAdapter", side_effect=_fake_ctor):
        LLMAdapterFactory.create(
            PROVIDER_SILICONFLOW,
            model="custom-model",
            api_key="custom-key",
            base_url="https://custom.example/v1",
        )

    assert captured == {
        "api_key": "custom-key",
        "base_url": "https://custom.example/v1",
        "default_model": "custom-model",
    }


def test_deepseek_prefers_env(monkeypatch):
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(
            os.environ,
            {
                ENV_DEEPSEEK_API_KEY: "deepseek-key",
                ENV_OPENAI_API_KEY: "openai-key",
                ENV_DEEPSEEK_BASE_URL: "https://deepseek.example",
                ENV_OPENAI_BASE_URL: "https://openai.example",
                ENV_SILICONFLOW_MODEL: "deepseek-ai/DeepSeek-V3.2",
                ENV_DEEPSEEK_MODEL: "deepseek-chat",
            },
            clear=True,
        ),
    ):
        adapter = _create_deepseek_adapter()

    assert adapter.api_key == "deepseek-key"
    assert adapter.base_url == "https://deepseek.example"
    assert adapter.model == "deepseek-ai/DeepSeek-V3.2"


def test_deepseek_accepts_qwen_alias(monkeypatch):
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(
            os.environ,
            {
                ENV_DEEPSEEK_API_KEY: "deepseek-key",
                ENV_DEEPSEEK_BASE_URL: "https://deepseek.example",
                ENV_QWEN_MODEL: "Qwen/Qwen3-32B",
            },
            clear=True,
        ),
    ):
        adapter = _create_deepseek_adapter()

    assert adapter.model == "Qwen/Qwen3-32B"


def test_deepseek_uses_openai(monkeypatch):
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(
            os.environ,
            {
                ENV_DEEPSEEK_API_KEY: "",
                ENV_OPENAI_API_KEY: "openai-key",
                ENV_DEEPSEEK_BASE_URL: "",
                ENV_OPENAI_BASE_URL: "https://openai.example",
                ENV_DEEPSEEK_MODEL: "",
                ENV_TOOLCALL_MODEL: "toolcall-model",
            },
            clear=False,
        ),
    ):
        adapter = _create_deepseek_adapter()

    assert adapter.api_key == "openai-key"
    assert adapter.base_url == "https://openai.example"
    assert adapter.model == "toolcall-model"
