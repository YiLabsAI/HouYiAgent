"""Unit tests for houyi.adapters.llm.factory — LLMAdapterFactory."""

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

    def test_google_ai_alias_uses_vertex_path(self):
        adapter = LLMAdapterFactory.create(PROVIDER_GOOGLE_AI)
        assert hasattr(adapter, "stream_chat")


def test_create_vertex_adapter_falls_back_when_google_sdk_missing(monkeypatch):
    fake_env = SimpleNamespace(
        google_project="proj",
        google_api_key=None,
        gemini_model="gemini-test",
        google_credentials_path=None,
        google_project_id="proj",
        google_location="us-central1",
    )
    monkeypatch.setattr(
        "houyi.infrastructure.config.env_config.EnvConfig.get",
        lambda: fake_env,
    )

    original_import = __import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "houyi.adapters.llm.vertex_gemini_adapter":
            raise ImportError("missing google-genai")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_fake_import):
        adapter = _create_vertex_adapter()

    assert adapter.__class__.__name__ == "VertexAIAdapter"


def test_create_vertex_adapter_falls_back_when_google_adapter_rejects_env(monkeypatch):
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
    monkeypatch.setattr(
        "houyi.infrastructure.config.env_config.EnvConfig.get",
        lambda: fake_env,
    )
    fake_module = ModuleType("houyi.adapters.llm.vertex_gemini_adapter")
    fake_module.GoogleVertexGeminiAdapter = _BadAdapter

    with patch.dict(sys.modules, {"houyi.adapters.llm.vertex_gemini_adapter": fake_module}):
        adapter = _create_vertex_adapter()

    assert adapter.__class__.__name__ == "VertexAIAdapter"


def test_create_deepseek_adapter_prefers_deepseek_env_over_openai_fallbacks(monkeypatch):
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
                ENV_DEEPSEEK_MODEL: "deepseek-chat",
                ENV_TOOLCALL_MODEL: "toolcall-model",
            },
            clear=False,
        ),
    ):
        adapter = _create_deepseek_adapter()

    assert adapter.api_key == "deepseek-key"
    assert adapter.base_url == "https://deepseek.example"
    assert adapter.model == "deepseek-chat"


def test_create_deepseek_adapter_uses_openai_and_toolcall_fallbacks(monkeypatch):
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
