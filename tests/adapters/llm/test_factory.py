from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from houyi.adapters.llm.factory import (
    LLMAdapterFactory,
    _create_deepseek_adapter,
    _create_vertex_adapter,
)
from houyi.adapters.llm.models import (
    PROVIDER_DEEPSEEK,
    PROVIDER_GOOGLE_AI,
    PROVIDER_OPENAI_COMPAT,
    PROVIDER_SILICONFLOW,
    PROVIDER_VERTEX,
)
from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.infrastructure.config.env_config import (
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


@pytest.fixture(autouse=True)
def _reset_env_config():
    """Reset EnvConfig singleton before/after each test so env patches take effect."""
    EnvConfig._reset()
    yield
    EnvConfig._reset()


class TestLLMAdapterFactory:
    """Test LLMAdapterFactory.create."""

    def test_default_siliconflow(self):
        with patch.dict(os.environ, {ENV_DEFAULT_LLM_PROVIDER: PROVIDER_SILICONFLOW}, clear=False):
            adapter = LLMAdapterFactory.create()
            assert isinstance(adapter, SiliconFlowAdapter)

    def test_vertex(self):
        """Vertex provider creates an adapter (type depends on available SDK + env)."""
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

    def test_unknown_falls_back(self):
        adapter = LLMAdapterFactory.create("unknown_provider")
        assert isinstance(adapter, SiliconFlowAdapter)

    def test_none_uses_env(self):
        with patch.dict(os.environ, {ENV_DEFAULT_LLM_PROVIDER: PROVIDER_VERTEX}):
            adapter = LLMAdapterFactory.create()
            assert hasattr(adapter, "stream_completion")

    def test_google_routes_vertex(self):
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
