"""Unit tests for houyi.adapters.llm.factory — LLMAdapterFactory."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from houyi.adapters.llm.factory import LLMAdapterFactory
from houyi.adapters.llm.models import (
    PROVIDER_DEEPSEEK,
    PROVIDER_OPENAI_COMPAT,
    PROVIDER_SILICONFLOW,
    PROVIDER_VERTEX,
)
from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.infrastructure.config.env_config import (
    ENV_DEEPSEEK_API_KEY,
    ENV_DEFAULT_LLM_PROVIDER,
    ENV_OPENAI_API_KEY,
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
