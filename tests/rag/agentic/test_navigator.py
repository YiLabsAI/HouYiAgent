"""Tests for agentic directory navigation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.rag.agentic.navigator import DirectoryNavigator


class TestDirectoryNavigator:
    @pytest.fixture
    def temp_knowledge_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data_structure.md").write_text(
                """
# Knowledge Base Structure

## Directories

- `docs/` - Documentation files
- `reports/` - Report files
                """
            )

            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "data_structure.md").write_text(
                """
# Documentation

- `guide.md` - User guide
- `api.md` - API reference
                """
            )
            (docs_dir / "guide.md").write_text("# User Guide\nThis is a guide.")
            (docs_dir / "api.md").write_text("# API Reference\nAPI documentation.")
            yield tmpdir

    @pytest.mark.asyncio
    async def test_find_candidates_empty_dir(self) -> None:
        navigator = DirectoryNavigator(knowledge_dir="/nonexistent")
        candidates = await navigator.find_candidates("test query")
        assert candidates == []

    @pytest.mark.asyncio
    async def test_find_candidates(self, temp_knowledge_dir: str) -> None:
        navigator = DirectoryNavigator(knowledge_dir=temp_knowledge_dir)
        candidates = await navigator.find_candidates("guide")
        assert len(candidates) > 0
        assert any("guide.md" in c for c in candidates)

    def test_is_searchable_file(self) -> None:
        navigator = DirectoryNavigator(knowledge_dir="/tmp")
        assert navigator._is_searchable_file(Path("/tmp/test.md"))
        assert navigator._is_searchable_file(Path("/tmp/test.txt"))
        assert navigator._is_searchable_file(Path("/tmp/test.py"))
        assert not navigator._is_searchable_file(Path("/tmp/test.jpg"))
        assert not navigator._is_searchable_file(Path("/tmp/test.exe"))
