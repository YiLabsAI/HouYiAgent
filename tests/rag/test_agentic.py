"""Tests for Agentic mode components."""

import tempfile
from pathlib import Path

import pytest

from houyi.rag.agentic.navigator import DirectoryNavigator
from houyi.rag.agentic.searcher import AgenticSearcher
from houyi.rag.config import AgenticConfig


class TestDirectoryNavigator:
    """Tests for DirectoryNavigator."""

    @pytest.fixture
    def temp_knowledge_dir(self) -> str:
        """Create a temporary knowledge directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            root = Path(tmpdir)

            # Create root index
            (root / "data_structure.md").write_text("""
# Knowledge Base Structure

## Directories

- `docs/` - Documentation files
- `reports/` - Report files
            """)

            # Create subdirectory with index
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "data_structure.md").write_text("""
# Documentation

- `guide.md` - User guide
- `api.md` - API reference
            """)
            (docs_dir / "guide.md").write_text("# User Guide\nThis is a guide.")
            (docs_dir / "api.md").write_text("# API Reference\nAPI documentation.")

            yield tmpdir

    @pytest.mark.asyncio
    async def test_find_candidates_empty_dir(self) -> None:
        """Test finding candidates in non-existent directory."""
        navigator = DirectoryNavigator(knowledge_dir="/nonexistent")
        candidates = await navigator.find_candidates("test query")
        assert candidates == []

    @pytest.mark.asyncio
    async def test_find_candidates(self, temp_knowledge_dir: str) -> None:
        """Test finding candidate files."""
        navigator = DirectoryNavigator(knowledge_dir=temp_knowledge_dir)
        candidates = await navigator.find_candidates("guide")

        # Should find files in the directory
        assert len(candidates) > 0
        assert any("guide.md" in c for c in candidates)

    def test_is_searchable_file(self) -> None:
        """Test file type detection."""
        navigator = DirectoryNavigator(knowledge_dir="/tmp")

        assert navigator._is_searchable_file(Path("/tmp/test.md"))
        assert navigator._is_searchable_file(Path("/tmp/test.txt"))
        assert navigator._is_searchable_file(Path("/tmp/test.py"))
        assert not navigator._is_searchable_file(Path("/tmp/test.jpg"))
        assert not navigator._is_searchable_file(Path("/tmp/test.exe"))


class TestAgenticSearcher:
    """Tests for AgenticSearcher."""

    @pytest.fixture
    def temp_search_dir(self) -> str:
        """Create a temporary directory with searchable files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create test files
            (root / "test1.txt").write_text("""
Line 1: This is a test file.
Line 2: It contains some keywords.
Line 3: RAG is retrieval augmented generation.
Line 4: More content here.
Line 5: End of file.
            """)

            (root / "test2.txt").write_text("""
Line 1: Another file.
Line 2: This one is about AI.
Line 3: Machine learning is great.
Line 4: End.
            """)

            yield tmpdir

    @pytest.mark.asyncio
    async def test_search_files_empty(self) -> None:
        """Test searching with no paths."""
        searcher = AgenticSearcher(knowledge_dir="/tmp")
        results = await searcher.search_files([], ["keyword"])
        assert results == []

    @pytest.mark.asyncio
    async def test_search_files_no_keywords(self) -> None:
        """Test searching with no keywords."""
        searcher = AgenticSearcher(knowledge_dir="/tmp")
        results = await searcher.search_files(["/tmp/test.txt"], [])
        assert results == []

    @pytest.mark.asyncio
    async def test_search_files_basic(self, temp_search_dir: str) -> None:
        """Test basic file search."""
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
        """Test Python-based grep fallback."""
        searcher = AgenticSearcher(knowledge_dir=temp_search_dir)

        path = str(Path(temp_search_dir) / "test1.txt")
        matches = await searcher._python_grep(path, ["RAG", "test"])

        assert len(matches) > 0
        assert any(m["line_num"] == 4 for m in matches)  # RAG is on line 4 (due to leading newline)

    @pytest.mark.asyncio
    async def test_read_context(self, temp_search_dir: str) -> None:
        """Test reading context around a line."""
        searcher = AgenticSearcher(knowledge_dir=temp_search_dir)

        path = str(Path(temp_search_dir) / "test1.txt")
        content = await searcher._read_context(path, 3, context_lines=2)

        assert "RAG" in content
        assert len(content) > 0


class TestAgenticConfig:
    """Tests for AgenticConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = AgenticConfig()
        assert config.max_rounds == 5
        assert config.index_file == "data_structure.md"
        assert config.chunk_limit == 500

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = AgenticConfig(
            max_rounds=3,
            index_file="index.md",
            chunk_limit=200,
        )
        assert config.max_rounds == 3
        assert config.index_file == "index.md"
        assert config.chunk_limit == 200
