"""Tests for default RAG test storage isolation paths."""

from __future__ import annotations

import os
from pathlib import Path

from houyi.infrastructure.config.env_config import ENV_KNOWLEDGE_STORAGE


def test_default_knowledge_storage_env_uses_tmp_houyi_layout(tmp_path: Path) -> None:
    expected = tmp_path / ".houyi" / "knowledge"
    assert os.environ[ENV_KNOWLEDGE_STORAGE] == str(expected)
    assert os.environ[ENV_KNOWLEDGE_STORAGE] != str(tmp_path / "knowledge")
