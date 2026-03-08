"""Tests for agentic file searching."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.rag.agentic.searcher import AgenticSearcher


class TestAgenticSearcher:
    @pytest.fixture
    def temp_search_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test1.txt").write_text(
                """
Line 1: This is a test file.
Line 2: It contains some keywords.
Line 3: RAG is retrieval augmented generation.
Line 4: More content here.
Line 5: End of file.
                """
            )
            (root / "test2.txt").write_text(
                """
Line 1: Another file.
Line 2: This one is about AI.
Line 3: Machine learning is great.
Line 4: End.
                """
            )
            yield tmpdir

    @pytest.mark.asyncio
    async def test_search_files_empty(self) -> None:
        searcher = AgenticSearcher(knowledge_dir="/tmp")
        results = await searcher.search_files([], ["keyword"])
        assert results == []

    @pytest.mark.asyncio
    async def test_search_files_no_keywords(self) -> None:
        searcher = AgenticSearcher(knowledge_dir="/tmp")
        results = await searcher.search_files(["/tmp/test.txt"], [])
        assert results == []

    @pytest.mark.asyncio
    async def test_search_files_basic(self, temp_search_dir: str) -> None:
        searcher = AgenticSearcher(knowledge_dir=temp_search_dir)
        paths = [
            str(Path(temp_search_dir) / "test1.txt"),
            str(Path(temp_search_dir) / "test2.txt"),
        ]
        results = await searcher.search_files(paths, ["RAG"])

        assert len(results) > 0
        assert any("RAG" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_python_grep(self, temp_search_dir: str) -> None:
        searcher = AgenticSearcher(knowledge_dir=temp_search_dir)
        path = str(Path(temp_search_dir) / "test1.txt")
        matches = await searcher._python_grep(path, ["RAG", "test"])

        assert len(matches) > 0
        assert any(m["line_num"] == 4 for m in matches)

    @pytest.mark.asyncio
    async def test_read_context(self, temp_search_dir: str) -> None:
        searcher = AgenticSearcher(knowledge_dir=temp_search_dir)
        path = str(Path(temp_search_dir) / "test1.txt")
        content = await searcher._read_context(path, 3, context_lines=2)

        assert "RAG" in content
        assert len(content) > 0
