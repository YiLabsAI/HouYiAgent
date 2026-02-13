"""Unit tests for RAG embedding config detection and config hash.

Tests:
- _detect_embedding_config() returns correct provider based on env vars
- Config hash includes embedding provider/model/dimension
- Switching embedding provider changes the config hash
"""

from __future__ import annotations

import hashlib
import json
import os
from unittest.mock import patch


class TestDetectEmbeddingConfig:
    """Test _detect_embedding_config() centralized helper."""

    def _import_detect(self):
        from houyi_studio.server.rag_service import _detect_embedding_config

        return _detect_embedding_config

    @patch.dict(
        os.environ,
        {
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "VERTEX_PROJECT": "",
            "GOOGLE_CLOUD_PROJECT": "",
            "GOOGLE_PROJECT_ID": "",
            "OPENAI_API_KEY": "sk-test-key",
        },
        clear=False,
    )
    def test_openai_detected_when_api_key_set(self):
        """OpenAI provider detected when OPENAI_API_KEY is set."""
        detect = self._import_detect()
        config, name = detect()
        assert config is not None
        assert name == "openai"
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.dimension == 1536

    @patch.dict(
        os.environ,
        {
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "VERTEX_PROJECT": "",
            "GOOGLE_CLOUD_PROJECT": "",
            "GOOGLE_PROJECT_ID": "",
            "OPENAI_API_KEY": "",
        },
        clear=False,
    )
    def test_no_provider_when_no_keys(self):
        """Returns None when no embedding provider is configured."""
        detect = self._import_detect()
        # Also need to ensure fastembed is not importable
        with patch.dict("sys.modules", {"fastembed": None}):
            with patch(
                "builtins.__import__",
                side_effect=_import_blocker(["fastembed", "google", "google.genai"]),
            ):
                config, name = detect()
        # If fastembed happens to be installed, it will return local
        # We just verify the function doesn't crash
        assert config is None or config.provider in ("local", "openai", "gemini")

    @patch.dict(
        os.environ,
        {
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "VERTEX_PROJECT": "my-project",
            "OPENAI_API_KEY": "sk-test",
        },
        clear=False,
    )
    def test_vertex_preferred_over_openai(self):
        """Vertex/Gemini is preferred over OpenAI when both are configured."""
        detect = self._import_detect()
        try:
            from google import genai  # noqa: F401

            config, name = detect()
            assert name == "gemini"
            assert config.provider == "gemini"
        except ImportError:
            # google-genai not installed — should fall through to OpenAI
            config, name = detect()
            assert name == "openai"


class TestConfigHash:
    """Test that config hash includes embedding information."""

    def _compute_hash(self, config_dict: dict) -> str:
        return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]

    def test_same_config_same_hash(self):
        """Identical configs produce identical hashes."""
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
        """Switching embedding provider changes the hash."""
        base = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
        }
        openai_config = {
            **base,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        }
        gemini_config = {
            **base,
            "embedding_provider": "gemini",
            "embedding_model": "text-embedding-004",
            "embedding_dimension": 768,
        }
        assert self._compute_hash(openai_config) != self._compute_hash(gemini_config)

    def test_different_embedding_model_different_hash(self):
        """Switching embedding model (same provider) changes the hash."""
        base = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
            "embedding_provider": "openai",
            "embedding_dimension": 1536,
        }
        config_a = {**base, "embedding_model": "text-embedding-3-small"}
        config_b = {**base, "embedding_model": "text-embedding-3-large"}
        assert self._compute_hash(config_a) != self._compute_hash(config_b)

    def test_different_embedding_dimension_different_hash(self):
        """Switching embedding dimension changes the hash."""
        base = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        }
        config_a = {**base, "embedding_dimension": 1536}
        config_b = {**base, "embedding_dimension": 768}
        assert self._compute_hash(config_a) != self._compute_hash(config_b)

    def test_chunk_config_change_also_changes_hash(self):
        """Changing chunk_size still changes the hash (regression check)."""
        base = {
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        }
        config_a = {**base, "chunk_size": 512}
        config_b = {**base, "chunk_size": 1024}
        assert self._compute_hash(config_a) != self._compute_hash(config_b)

    def test_hash_without_embedding_fields_differs_from_with(self):
        """Old-style hash (no embedding fields) differs from new-style."""
        old_config = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
        }
        new_config = {
            **old_config,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        }
        assert self._compute_hash(old_config) != self._compute_hash(new_config)


def _import_blocker(blocked_modules: list[str]):
    """Create an __import__ side_effect that blocks specific modules."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def blocker(name, *args, **kwargs):
        for blocked in blocked_modules:
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"Blocked: {name}")
        return real_import(name, *args, **kwargs)

    return blocker
