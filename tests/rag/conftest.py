"""Pytest fixtures for RAG tests."""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def cleanup_gc():
    """Clean up garbage after each test to close SQLite connections.

    bm25s uses Snowball Stemmer which has internal SQLite connections.
    Force GC after each test to ensure connections are closed.
    """
    yield
    gc.collect()


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create an isolated knowledge directory for a test."""
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    return kb_dir


@pytest.fixture
def write_knowledge_files(knowledge_dir: Path) -> Callable[[dict[str, str]], Path]:
    """Populate the test knowledge directory with one or more files."""

    def _write(files: dict[str, str]) -> Path:
        for relative_path, content in files.items():
            target = knowledge_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        return knowledge_dir

    return _write


@pytest.fixture
def patch_rag_constructor() -> Callable[[Any, Any], Any]:
    """Patch a module-local RAG constructor and yield the constructor mock."""

    @contextmanager
    def _patch(rag_module: Any, rag_instance: Any):
        with patch.object(rag_module, "RAG", return_value=rag_instance) as mock_ctor:
            yield mock_ctor

    return _patch


@pytest.fixture
def patch_skill_rag_builder() -> Callable[[Any, Any], Any]:
    """Patch a module-local build_skill_rag factory and yield the factory mock."""

    @contextmanager
    def _patch(skill_module: Any, rag_instance: Any):
        with patch.object(
            skill_module, "build_skill_rag", return_value=rag_instance
        ) as mock_builder:
            yield mock_builder

    return _patch
