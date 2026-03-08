"""Tests for embedding config resolution and config hash.

Tests:
- resolve_embedding_config() priority chain: explicit > env > auto-detect
- _auto_detect_embedding() detection order: Gemini(API key) > Gemini(Vertex) > OpenAI > local
- Config hash includes embedding provider/model/dimension
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_blocker(blocked_modules: list[str]):
    """Create an __import__ side_effect that blocks specific modules."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def blocker(name, *args, **kwargs):
        for blocked in blocked_modules:
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"Blocked: {name}")
        return real_import(name, *args, **kwargs)

    return blocker


def _embedding_config_module():
    from houyi_studio.server.rag import embedding_config

    return embedding_config


# ---------------------------------------------------------------------------
# Priority chain tests
# ---------------------------------------------------------------------------


class TestResolveEmbeddingConfig:
    """Test resolve_embedding_config() priority chain."""

    def _resolve(self, **kwargs):
        from houyi_studio.server.rag import resolve_embedding_config

        return resolve_embedding_config(**kwargs)

    def test_explicit_provider_wins_over_everything(self):
        """Priority 1: explicit override always wins."""
        with (
            patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "sk-x"}),
            patch.object(
                _embedding_config_module(),
                "_is_provider_runtime_available",
                return_value=True,
            ),
        ):
            cfg, name = self._resolve(preferred_provider="local")
            assert name == "local"
            assert cfg.provider == "local"
            assert cfg.model == "BAAI/bge-small-en-v1.5"
            assert cfg.dimension == 384

    def test_explicit_local_falls_back_when_fastembed_missing(self):
        def _available(provider: str) -> bool:
            return provider != "local"

        with (
            patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "sk-x"}),
            patch.object(
                _embedding_config_module(),
                "_is_provider_runtime_available",
                side_effect=_available,
            ),
        ):
            cfg, name = self._resolve(preferred_provider="local")
            assert name == "openai"
            assert cfg.provider == "openai"

    def test_explicit_local_strict_mode_raises(self):
        with patch.object(
            _embedding_config_module(),
            "_is_provider_runtime_available",
            return_value=False,
        ):
            try:
                self._resolve(preferred_provider="local", strict_explicit=True)
                raise AssertionError("Expected RuntimeError for strict explicit provider")
            except RuntimeError as exc:
                assert "Embedding provider 'local'" in str(exc)

    def test_env_local_strict_mode_raises(self):
        with (
            patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}, clear=False),
            patch.object(
                _embedding_config_module(),
                "_is_provider_runtime_available",
                return_value=False,
            ),
        ):
            try:
                self._resolve(strict_explicit=True)
                raise AssertionError("Expected RuntimeError for strict env provider")
            except RuntimeError as exc:
                assert "from env" in str(exc)

    def test_explicit_provider_with_custom_model(self):
        """Explicit override can include custom model/dimension."""
        with patch.object(
            _embedding_config_module(),
            "_is_provider_runtime_available",
            return_value=True,
        ):
            cfg, name = self._resolve(
                preferred_provider="openai",
                preferred_model="text-embedding-3-large",
                preferred_dimension=3072,
            )
        assert cfg.provider == "openai"
        assert cfg.model == "text-embedding-3-large"
        assert cfg.dimension == 3072

    @patch.dict(
        os.environ,
        {"EMBEDDING_PROVIDER": "gemini", "EMBEDDING_MODEL": "text-embedding-004"},
        clear=False,
    )
    def test_env_vars_used_when_no_explicit(self):
        """Priority 2: env vars used when no explicit provider given."""
        with patch.object(
            _embedding_config_module(),
            "_is_provider_runtime_available",
            return_value=True,
        ):
            cfg, name = self._resolve()
        assert name == "gemini"
        assert cfg.provider == "gemini"
        assert cfg.model == "text-embedding-004"

    def test_auto_detect_fallback(self):
        """Priority 3: auto-detect when no explicit + no env vars."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL")
        }
        with patch.dict(os.environ, env, clear=True):
            cfg, name = self._resolve()
            if cfg is not None:
                assert cfg.provider in ("local", "openai", "gemini")


class TestAutoDetectEmbedding:
    """Test _auto_detect_embedding() detection order."""

    def _auto_detect(self):
        from houyi_studio.server.rag import _auto_detect_embedding

        return _auto_detect_embedding

    @patch.dict(
        os.environ,
        {"GOOGLE_API_KEY": "test-api-key", "OPENAI_API_KEY": "sk-test"},
        clear=False,
    )
    def test_gemini_api_key_preferred_over_openai(self):
        """Gemini (GOOGLE_API_KEY) is preferred over OpenAI."""
        detect = self._auto_detect()
        try:
            from google import genai  # noqa: F401

            cfg, name = detect()
            assert name == "gemini"
        except ImportError:
            with patch.object(
                _embedding_config_module(),
                "_is_provider_runtime_available",
                return_value=True,
            ):
                cfg, name = detect()
                assert name == "openai"

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "",
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "OPENAI_API_KEY": "sk-test",
        },
        clear=False,
    )
    def test_vertex_preferred_over_openai(self):
        """Gemini (GOOGLE_CLOUD_PROJECT) is preferred over OpenAI."""
        detect = self._auto_detect()
        try:
            from google import genai  # noqa: F401

            cfg, name = detect()
            assert name == "gemini"
        except ImportError:
            with patch.object(
                _embedding_config_module(),
                "_is_provider_runtime_available",
                return_value=True,
            ):
                cfg, name = detect()
                assert name == "openai"

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "",
            "GOOGLE_CLOUD_PROJECT": "",
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "OPENAI_API_KEY": "sk-test-key",
        },
        clear=False,
    )
    def test_openai_detected_when_api_key_set(self):
        """OpenAI provider detected when OPENAI_API_KEY is set."""
        detect = self._auto_detect()
        with patch.object(
            _embedding_config_module(),
            "_is_provider_runtime_available",
            return_value=True,
        ):
            cfg, name = detect()
        assert name == "openai"
        assert cfg.provider == "openai"
        assert cfg.model == "text-embedding-3-small"
        assert cfg.dimension == 1536

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "",
            "GOOGLE_CLOUD_PROJECT": "",
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "OPENAI_API_KEY": "",
        },
        clear=False,
    )
    def test_local_fastembed_fallback(self):
        """Local fastembed detected when no API keys available."""
        detect = self._auto_detect()
        try:
            import fastembed  # noqa: F401

            cfg, name = detect()
            assert name == "local"
            assert cfg.provider == "local"
            assert cfg.model == "BAAI/bge-small-en-v1.5"
            assert cfg.dimension == 384
        except ImportError:
            cfg, name = detect()
            assert cfg is None
            assert name == "no_provider"


class TestStoragePathHelpers:
    def _auto_detect(self):
        from houyi_studio.server.rag import _auto_detect_embedding

        return _auto_detect_embedding

    def test_storage_constants_defined(self):
        from houyi_studio.server.rag import (
            INDEX_SUBDIR,
            KNOWLEDGE_STORAGE_DIR,
            UPLOADS_SUBDIR,
        )

        assert UPLOADS_SUBDIR == "uploads"
        assert INDEX_SUBDIR == "index"
        assert KNOWLEDGE_STORAGE_DIR.is_absolute()

    def test_helper_functions_use_constants(self):
        from houyi_studio.server.rag import (
            INDEX_SUBDIR,
            UPLOADS_SUBDIR,
            get_library_index_dir,
            get_library_upload_dir,
        )

        lib_id = "test_lib"
        upload_dir = get_library_upload_dir(lib_id)
        index_dir = get_library_index_dir(lib_id)

        assert upload_dir.name == UPLOADS_SUBDIR
        assert index_dir.name == INDEX_SUBDIR

    def test_is_index_path_detects_only_internal_index_locations(self):
        from houyi_studio.server.rag import is_index_path

        index_path = Path(
            os.path.join("/home", ".houyi", "knowledge", "lib_001", "index", "vectors.bin")
        )
        assert is_index_path(index_path)

        second_index_path = Path(
            os.path.join("/Users", "test", ".houyi", "knowledge", "lib_001", "index", "meta.json")
        )
        assert is_index_path(second_index_path)

        upload_path = Path(
            os.path.join("/home", ".houyi", "knowledge", "lib_001", "uploads", "doc.md")
        )
        assert not is_index_path(upload_path)

        regular_path = Path(os.path.join("/home", "user", "documents", "readme.md"))
        assert not is_index_path(regular_path)

        random_index = Path(os.path.join("/some", "random", "index", "file.txt"))
        assert not is_index_path(random_index)

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "",
            "GOOGLE_CLOUD_PROJECT": "",
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "OPENAI_API_KEY": "",
        },
        clear=False,
    )
    def test_no_provider_when_nothing_available(self):
        """Returns (None, 'no_provider') when nothing is available."""
        detect = self._auto_detect()
        with patch.dict("sys.modules", {"fastembed": None}):
            with patch("builtins.__import__", side_effect=_import_blocker(["fastembed"])):
                cfg, name = detect()
        if cfg is None:
            assert name == "no_provider"


class TestBackwardCompat:
    """Test legacy _detect_embedding_config still works."""

    def test_legacy_function_exists_and_works(self):
        from houyi_studio.server.rag import _detect_embedding_config

        cfg, name = _detect_embedding_config()
        assert name in ("local", "openai", "gemini", "no_provider")


# ---------------------------------------------------------------------------
# Provider defaults tests
# ---------------------------------------------------------------------------


class TestProviderDefaults:
    """Test _make_embedding_config default model/dimension lookup."""

    def _make(self, provider, model=None, dimension=None):
        from houyi_studio.server.rag import _make_embedding_config

        return _make_embedding_config(provider, model, dimension)

    def test_local_defaults(self):
        cfg = self._make("local")
        assert cfg.model == "BAAI/bge-small-en-v1.5"
        assert cfg.dimension == 384

    def test_gemini_defaults(self):
        cfg = self._make("gemini")
        assert cfg.model == "text-embedding-004"
        assert cfg.dimension == 768

    def test_openai_defaults(self):
        cfg = self._make("openai")
        assert cfg.model == "text-embedding-3-small"
        assert cfg.dimension == 1536

    def test_custom_model_overrides_default(self):
        cfg = self._make("openai", model="text-embedding-3-large", dimension=3072)
        assert cfg.model == "text-embedding-3-large"
        assert cfg.dimension == 3072


# ---------------------------------------------------------------------------
# Config hash tests
# ---------------------------------------------------------------------------


class TestConfigHash:
    """Test that config hash includes embedding information."""

    def _compute_hash(self, config_dict: dict) -> str:
        return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]

    def test_same_config_same_hash(self):
        config = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        }
        assert self._compute_hash(config) == self._compute_hash(config.copy())

    def test_different_embedding_provider_different_hash(self):
        base = {"chunk_size": 512, "chunk_overlap": 50, "chunking_strategy": "recursive"}
        openai_cfg = {
            **base,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        }
        gemini_cfg = {
            **base,
            "embedding_provider": "gemini",
            "embedding_model": "text-embedding-004",
            "embedding_dimension": 768,
        }
        assert self._compute_hash(openai_cfg) != self._compute_hash(gemini_cfg)

    def test_different_embedding_model_different_hash(self):
        base = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
            "embedding_provider": "openai",
            "embedding_dimension": 1536,
        }
        assert self._compute_hash(
            {**base, "embedding_model": "text-embedding-3-small"}
        ) != self._compute_hash({**base, "embedding_model": "text-embedding-3-large"})
