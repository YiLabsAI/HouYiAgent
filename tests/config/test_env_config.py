"""Unit tests for houyi.config.env_config — EnvConfig singleton.

Tests cover:
- Singleton behavior
- Default values
- Env var overrides for all properties
- reload() re-reads env vars
- summary() masks API keys
- Google env vars follow google-genai SDK conventions
"""

from __future__ import annotations

import os
from unittest.mock import patch

from houyi.config.env_config import EnvConfig


class TestEnvConfigSingleton:
    """Singleton must return the same instance and support reset."""

    def setup_method(self):
        EnvConfig._reset()

    def teardown_method(self):
        EnvConfig._reset()

    def test_get_returns_same_instance(self):
        a = EnvConfig.get()
        b = EnvConfig.get()
        assert a is b

    def test_reset_clears_singleton(self):
        a = EnvConfig.get()
        EnvConfig._reset()
        b = EnvConfig.get()
        assert a is not b


class TestEnvConfigDefaults:
    """Default values when no env vars are set."""

    def setup_method(self):
        EnvConfig._reset()

    def teardown_method(self):
        EnvConfig._reset()

    def test_siliconflow_api_key_none_by_default(self):
        env_clean = {k: v for k, v in os.environ.items() if k != "SILICONFLOW_API_KEY"}
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.siliconflow_api_key is None

    def test_siliconflow_base_url_default(self):
        env_clean = {k: v for k, v in os.environ.items() if k != "SILICONFLOW_BASE_URL"}
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.siliconflow_base_url == "https://api.siliconflow.cn/v1"

    def test_default_llm_provider_is_siliconflow(self):
        env_clean = {k: v for k, v in os.environ.items() if k != "DEFAULT_LLM_PROVIDER"}
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.default_llm_provider == "siliconflow"

    def test_google_location_default(self):
        env_clean = {k: v for k, v in os.environ.items() if k != "GOOGLE_CLOUD_LOCATION"}
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_location == "us-central1"

    def test_rag_knowledge_dir_default(self):
        env_clean = {k: v for k, v in os.environ.items() if k != "RAG_KNOWLEDGE_DIR"}
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.rag_knowledge_dir == "knowledge/"

    def test_embedding_defaults(self):
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.embedding_provider == "local"
            assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_google_api_key_none_by_default(self):
        env_clean = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_api_key is None


class TestEnvConfigOverrides:
    """Env var overrides should take effect."""

    def setup_method(self):
        EnvConfig._reset()

    def teardown_method(self):
        EnvConfig._reset()

    def test_siliconflow_api_key_override(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "sk-test-123"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.siliconflow_api_key == "sk-test-123"

    def test_siliconflow_base_url_override(self):
        with patch.dict(os.environ, {"SILICONFLOW_BASE_URL": "https://custom.api/v1"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.siliconflow_base_url == "https://custom.api/v1"

    def test_rag_knowledge_dir_override(self):
        with patch.dict(os.environ, {"RAG_KNOWLEDGE_DIR": "/data/kb/"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.rag_knowledge_dir == "/data/kb/"

    def test_default_llm_provider_override(self):
        with patch.dict(os.environ, {"DEFAULT_LLM_PROVIDER": "vertex"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.default_llm_provider == "vertex"

    def test_embedding_provider_override(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "gemini"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.embedding_provider == "gemini"

    def test_embedding_model_override(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "text-embedding-004"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.embedding_model == "text-embedding-004"


class TestEnvConfigGoogle:
    """Google env vars follow google-genai SDK conventions."""

    def setup_method(self):
        EnvConfig._reset()

    def teardown_method(self):
        EnvConfig._reset()

    def test_google_cloud_project(self):
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "my-project"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_project == "my-project"
            # Backward compat alias
            assert cfg.google_project_id == "my-project"

    def test_google_cloud_location(self):
        with patch.dict(os.environ, {"GOOGLE_CLOUD_LOCATION": "europe-west1"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_location == "europe-west1"

    def test_google_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIza-test-key"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_api_key == "AIza-test-key"

    def test_gemini_model(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.0-flash"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.gemini_model == "gemini-2.0-flash"


class TestEnvConfigReload:
    """reload() must re-read env vars."""

    def setup_method(self):
        EnvConfig._reset()

    def teardown_method(self):
        EnvConfig._reset()

    def test_reload_picks_up_new_values(self):
        cfg = EnvConfig.get()

        with patch.dict(os.environ, {"RAG_KNOWLEDGE_DIR": "/new/path/"}):
            cfg.reload()
            assert cfg.rag_knowledge_dir == "/new/path/"

        cfg.reload()


class TestEnvConfigSummary:
    """summary() must mask API keys."""

    def setup_method(self):
        EnvConfig._reset()

    def teardown_method(self):
        EnvConfig._reset()

    def test_summary_masks_api_key(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "sk-abcdef123456"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            s = cfg.summary()
            assert "abcdef" not in s["siliconflow_api_key"]
            assert s["siliconflow_api_key"].startswith("sk-a")
            assert s["siliconflow_api_key"].endswith("3456")

    def test_summary_masks_google_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyDabcdef123456"}):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            s = cfg.summary()
            assert "abcdef" not in s["google_api_key"]
            assert s["google_api_key"].startswith("AIza")

    def test_summary_shows_not_set_for_missing(self):
        env_clean = {k: v for k, v in os.environ.items() if k != "SILICONFLOW_API_KEY"}
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            s = cfg.summary()
            assert s["siliconflow_api_key"] == "(not set)"

    def test_summary_returns_all_keys(self):
        EnvConfig._reset()
        cfg = EnvConfig.get()
        s = cfg.summary()
        expected_keys = {
            "siliconflow_api_key",
            "siliconflow_base_url",
            "deepseek_model",
            "default_llm_provider",
            "google_api_key",
            "google_credentials_path",
            "google_project",
            "google_location",
            "gemini_model",
            "rag_knowledge_dir",
            "embedding_provider",
            "embedding_model",
        }
        assert set(s.keys()) == expected_keys

    def test_repr(self):
        EnvConfig._reset()
        cfg = EnvConfig.get()
        r = repr(cfg)
        assert "EnvConfig" in r
        assert "provider=" in r
        assert "embedding=" in r
