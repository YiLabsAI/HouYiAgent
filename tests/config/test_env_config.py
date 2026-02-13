"""Unit tests for houyi.config.env_config — EnvConfig singleton.

Tests cover:
- Singleton behavior (get returns same instance)
- Default values when no env vars set
- Env var overrides for all properties
- reload() re-reads env vars
- summary() masks API keys
- RAG knowledge dir reads from RAG_KNOWLEDGE_DIR
- Google/Vertex AI env var fallback chain
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
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VERTEX_LOCATION", "GOOGLE_LOCATION", "GOOGLE_CLOUD_LOCATION")
        }
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

    def test_rag_embedding_defaults(self):
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("RAG_EMBEDDING_PROVIDER", "RAG_EMBEDDING_MODEL")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.rag_embedding_provider == "openai"
            assert cfg.rag_embedding_model == "text-embedding-3-small"


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


class TestEnvConfigVertexFallbackChain:
    """Google/Vertex AI env vars have a fallback chain."""

    def setup_method(self):
        EnvConfig._reset()

    def teardown_method(self):
        EnvConfig._reset()

    def test_vertex_project_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "VERTEX_PROJECT": "proj-a",
                "GOOGLE_PROJECT_ID": "proj-b",
                "GOOGLE_CLOUD_PROJECT": "proj-c",
            },
        ):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_project_id == "proj-a"

    def test_google_project_id_fallback(self):
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VERTEX_PROJECT", "GOOGLE_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
        }
        env_clean["GOOGLE_PROJECT_ID"] = "proj-b"
        with patch.dict(os.environ, env_clean, clear=True):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_project_id == "proj-b"

    def test_vertex_location_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "VERTEX_LOCATION": "europe-west1",
                "GOOGLE_LOCATION": "asia-east1",
            },
        ):
            EnvConfig._reset()
            cfg = EnvConfig.get()
            assert cfg.google_location == "europe-west1"

    def test_vertex_gemini_model_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "VERTEX_GEMINI_MODEL": "gemini-2.0-flash",
                "GEMINI_MODEL": "gemini-1.5-pro",
            },
        ):
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
        original = cfg.rag_knowledge_dir

        with patch.dict(os.environ, {"RAG_KNOWLEDGE_DIR": "/new/path/"}):
            cfg.reload()
            assert cfg.rag_knowledge_dir == "/new/path/"

        # Reload again without the env var to restore
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
            "gemini_model",
            "google_credentials_path",
            "google_project_id",
            "google_location",
            "rag_knowledge_dir",
            "rag_embedding_provider",
            "rag_embedding_model",
        }
        assert set(s.keys()) == expected_keys

    def test_repr(self):
        EnvConfig._reset()
        cfg = EnvConfig.get()
        r = repr(cfg)
        assert "EnvConfig" in r
        assert "provider=" in r
        assert "rag_dir=" in r
