"""Tests for Agentic mode components."""

import tempfile
from pathlib import Path

import pytest

from houyi.rag.agentic.mode import (
    AgenticMode,
    RoundResult,
    SearchRoundType,
)
from houyi.rag.agentic.navigator import DirectoryNavigator
from houyi.rag.agentic.searcher import AgenticSearcher
from houyi.rag.config import AgenticConfig
from houyi.rag.types import SearchResult, Source


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


class TestSearchRoundType:
    """Tests for SearchRoundType enum."""

    def test_round_type_values(self) -> None:
        """Test enum values."""
        assert SearchRoundType.BROAD.value == "broad"
        assert SearchRoundType.FOCUSED.value == "focused"
        assert SearchRoundType.SEMANTIC.value == "semantic"
        assert SearchRoundType.CROSS_REF.value == "cross_ref"
        assert SearchRoundType.VERIFY.value == "verify"

    def test_round_type_count(self) -> None:
        """Test that we have exactly 5 round types."""
        assert len(SearchRoundType) == 5


class TestRoundResult:
    """Tests for RoundResult dataclass."""

    def test_round_result_creation(self) -> None:
        """Test creating a RoundResult."""
        result = RoundResult(
            round_type=SearchRoundType.BROAD,
            round_num=0,
            keywords_used=["test", "query"],
            results=[],
            files_searched=5,
        )

        assert result.round_type == SearchRoundType.BROAD
        assert result.round_num == 0
        assert result.keywords_used == ["test", "query"]
        assert result.results == []
        assert result.files_searched == 5
        assert result.metadata == {}

    def test_round_result_with_metadata(self) -> None:
        """Test RoundResult with metadata."""
        result = RoundResult(
            round_type=SearchRoundType.SEMANTIC,
            round_num=2,
            keywords_used=["expanded"],
            results=[],
            files_searched=3,
            metadata={"expanded_from": ["original"]},
        )

        assert result.metadata["expanded_from"] == ["original"]


class TestAgenticMode:
    """Tests for AgenticMode 5-round strategy."""

    @pytest.fixture
    def temp_knowledge_dir(self) -> str:
        """Create a temporary knowledge directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create index file
            (root / "data_structure.md").write_text(
                "# Knowledge Base\n- `doc1.md` - Document 1\n- `doc2.md` - Document 2"
            )

            # Create test documents
            (root / "doc1.md").write_text(
                "# Document 1\nThis document is about Python programming.\n"
                "Python is a great language for machine learning.\n"
                "TensorFlow and PyTorch are popular frameworks."
            )
            (root / "doc2.md").write_text(
                "# Document 2\nThis document covers JavaScript.\n"
                "React and Vue are frontend frameworks.\n"
                "Node.js runs JavaScript on the server."
            )

            yield tmpdir

    def test_extract_keywords_simple(self) -> None:
        """Test simple keyword extraction."""
        config = AgenticConfig()
        mode = AgenticMode(
            config=config,
            knowledge_dir="/tmp",
        )

        keywords = mode._extract_keywords_simple("What is machine learning?")

        assert "machine" in keywords
        assert "learning" in keywords
        assert "what" not in keywords  # stop word
        assert "is" not in keywords  # stop word

    def test_extract_keywords_simple_filters_short(self) -> None:
        """Test that short words are filtered."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        keywords = mode._extract_keywords_simple("A B C test")

        assert "test" in keywords
        assert "A" not in keywords
        assert "B" not in keywords

    def test_should_terminate_early(self) -> None:
        """Test early termination logic."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        # Should not terminate on first round
        results = [SearchResult(chunk_id=f"c{i}", content="test", score=0.8) for i in range(5)]
        assert not mode._should_terminate(results, SearchRoundType.BROAD)

        # Should terminate with many high-score results
        assert mode._should_terminate(results, SearchRoundType.FOCUSED)

    def test_should_terminate_not_enough_results(self) -> None:
        """Test no termination with few results."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        results = [SearchResult(chunk_id="c1", content="test", score=0.8)]
        assert not mode._should_terminate(results, SearchRoundType.FOCUSED)

    def test_get_top_files(self) -> None:
        """Test getting top-scoring files."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        results = [
            SearchResult(chunk_id="c1", content="a", score=0.9, source=Source(file_path="/a.md")),
            SearchResult(chunk_id="c2", content="b", score=0.7, source=Source(file_path="/b.md")),
            SearchResult(chunk_id="c3", content="a2", score=0.8, source=Source(file_path="/a.md")),
            SearchResult(chunk_id="c4", content="c", score=0.6, source=Source(file_path="/c.md")),
        ]

        top_files = mode._get_top_files(results, limit=2)

        assert len(top_files) == 2
        assert top_files[0] == "/a.md"  # highest score
        assert top_files[1] == "/b.md"  # second highest

    def test_extract_entities(self) -> None:
        """Test entity extraction from results."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        results = [
            SearchResult(chunk_id="c1", content='Python and TensorFlow are tools. "machine learning" is important.', score=0.9),
            SearchResult(chunk_id="c2", content="JavaScript runs in browsers. React is popular.", score=0.8),
        ]

        entities = mode._extract_entities(results)

        assert "Python" in entities
        assert "TensorFlow" in entities
        assert "JavaScript" in entities
        assert "React" in entities
        assert "machine learning" in entities

    def test_refine_keywords(self) -> None:
        """Test keyword refinement."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        results = [
            SearchResult(chunk_id="c1", content="Python is great for machine learning tasks", score=0.9),
        ]

        refined = mode._refine_keywords("What is Python machine learning?", results)

        # Should find words common to query and results
        assert "python" in refined or "machine" in refined or "learning" in refined

    def test_deduplicate_results(self) -> None:
        """Test result deduplication."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        results = [
            SearchResult(chunk_id="c1", content="Same content here", score=0.9),
            SearchResult(chunk_id="c2", content="Same content here", score=0.8),  # duplicate
            SearchResult(chunk_id="c3", content="Different content", score=0.7),
        ]

        deduped = mode._deduplicate_results(results)

        assert len(deduped) == 2
        assert deduped[0].score == 0.9  # highest score kept
        assert deduped[1].content == "Different content"

    def test_deduplicate_preserves_order(self) -> None:
        """Test that deduplication preserves score order."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        results = [
            SearchResult(chunk_id="c1", content="Low score", score=0.5),
            SearchResult(chunk_id="c2", content="High score", score=0.9),
            SearchResult(chunk_id="c3", content="Medium score", score=0.7),
        ]

        deduped = mode._deduplicate_results(results)

        assert deduped[0].score == 0.9
        assert deduped[1].score == 0.7
        assert deduped[2].score == 0.5

    @pytest.mark.asyncio
    async def test_search_returns_metadata(self, temp_knowledge_dir: str) -> None:
        """Test that search returns round metadata."""
        config = AgenticConfig(max_rounds=2)
        mode = AgenticMode(config=config, knowledge_dir=temp_knowledge_dir)

        result = await mode.search("Python programming")

        assert "rounds_executed" in result.metadata
        assert "round_history" in result.metadata
        assert "total_files_searched" in result.metadata
        assert result.metadata["rounds_executed"] <= 2

    @pytest.mark.asyncio
    async def test_search_round_history_structure(self, temp_knowledge_dir: str) -> None:
        """Test round history structure in metadata."""
        config = AgenticConfig(max_rounds=3)
        mode = AgenticMode(config=config, knowledge_dir=temp_knowledge_dir)

        result = await mode.search("Python")

        for round_info in result.metadata.get("round_history", []):
            assert "round" in round_info
            assert "type" in round_info
            assert "keywords" in round_info
            assert "results_found" in round_info
            assert "files_searched" in round_info

    @pytest.mark.asyncio
    async def test_search_empty_knowledge_base(self) -> None:
        """Test search with empty knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgenticConfig()
            mode = AgenticMode(config=config, knowledge_dir=tmpdir)

            result = await mode.search("test query")

            assert result.confidence == 0.0
            assert "No relevant" in result.answer
            assert result.metadata.get("rounds_executed") == 0

    @pytest.mark.asyncio
    async def test_search_caps_rounds_at_5(self, temp_knowledge_dir: str) -> None:
        """Test that search caps rounds at 5 even if config says more."""
        config = AgenticConfig(max_rounds=10)
        mode = AgenticMode(config=config, knowledge_dir=temp_knowledge_dir)

        result = await mode.search("Python")

        # Should execute at most 5 rounds
        assert result.metadata["rounds_executed"] <= 5

    @pytest.mark.asyncio
    async def test_build_answer_simple(self) -> None:
        """Test simple answer building."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        results = [
            SearchResult(chunk_id="c1", content="First result content", score=0.9),
            SearchResult(chunk_id="c2", content="Second result content", score=0.8),
        ]

        answer = mode._build_answer_simple(results)

        assert "First result content" in answer
        assert "Second result content" in answer
        assert "---" in answer  # separator

    @pytest.mark.asyncio
    async def test_build_answer_simple_empty(self) -> None:
        """Test simple answer building with no results."""
        config = AgenticConfig()
        mode = AgenticMode(config=config, knowledge_dir="/tmp")

        answer = mode._build_answer_simple([])

        assert "No relevant" in answer
