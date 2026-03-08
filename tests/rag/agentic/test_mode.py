"""Tests for agentic mode orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.rag.agentic.mode import AgenticMode, RoundResult, SearchRoundType
from houyi.rag.config import AgenticConfig
from houyi.rag.types import SearchResult, Source


class TestRoundResult:
    def test_round_result_creation(self) -> None:
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
    @pytest.fixture
    def temp_knowledge_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data_structure.md").write_text(
                "# Knowledge Base\n- `doc1.md` - Document 1\n- `doc2.md` - Document 2"
            )
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

    @pytest.mark.asyncio
    async def test_search_returns_metadata(self, temp_knowledge_dir: str) -> None:
        config = AgenticConfig(max_rounds=2)
        mode = AgenticMode(config=config, knowledge_dir=temp_knowledge_dir)

        result = await mode.search("Python programming")

        assert "rounds_executed" in result.metadata
        assert "round_history" in result.metadata
        assert "total_files_searched" in result.metadata
        assert result.metadata["rounds_executed"] <= 2

    @pytest.mark.asyncio
    async def test_search_round_history_structure(self, temp_knowledge_dir: str) -> None:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgenticConfig()
            mode = AgenticMode(config=config, knowledge_dir=tmpdir)

            result = await mode.search("test query")

            assert result.confidence == 0.0
            assert "No relevant" in result.answer
            assert result.metadata.get("rounds_executed") == 0

    @pytest.mark.asyncio
    async def test_search_caps_rounds_at_5(self, temp_knowledge_dir: str) -> None:
        config = AgenticConfig(max_rounds=10)
        mode = AgenticMode(config=config, knowledge_dir=temp_knowledge_dir)

        result = await mode.search("Python")

        assert result.metadata["rounds_executed"] <= 5

    @pytest.mark.asyncio
    async def test_execute_round_verify_files_searched(
        self,
        temp_knowledge_dir: str,
    ) -> None:
        mode = AgenticMode(config=AgenticConfig(), knowledge_dir=temp_knowledge_dir)
        candidate_paths = [
            str(Path(temp_knowledge_dir) / "doc1.md"),
            str(Path(temp_knowledge_dir) / "doc2.md"),
            str(Path(temp_knowledge_dir) / "doc3.md"),
        ]
        all_results = [
            SearchResult(
                chunk_id="c1",
                content="Python programming and machine learning",
                score=0.9,
                source=Source(file_path=candidate_paths[0]),
            ),
            SearchResult(
                chunk_id="c2",
                content="Python programming with TensorFlow",
                score=0.8,
                source=Source(file_path=candidate_paths[1]),
            ),
        ]

        round_result = await mode._execute_round(
            round_type=SearchRoundType.VERIFY,
            round_num=4,
            query="Python programming",
            candidate_paths=candidate_paths,
            initial_keywords=["python", "programming"],
            all_results=all_results,
            searched_files=set(candidate_paths),
        )

        assert round_result.files_searched == 2

    @pytest.mark.asyncio
    async def test_execute_round_cross_ref_no_entities(
        self,
        temp_knowledge_dir: str,
    ) -> None:
        mode = AgenticMode(config=AgenticConfig(), knowledge_dir=temp_knowledge_dir)
        candidate_paths = [
            str(Path(temp_knowledge_dir) / "doc1.md"),
            str(Path(temp_knowledge_dir) / "doc2.md"),
        ]
        all_results = [
            SearchResult(
                chunk_id="c1",
                content="python machine learning pipelines",
                score=0.9,
            )
        ]

        round_result = await mode._execute_round(
            round_type=SearchRoundType.CROSS_REF,
            round_num=3,
            query="python pipelines",
            candidate_paths=candidate_paths,
            initial_keywords=["python", "pipelines"],
            all_results=all_results,
            searched_files=set(),
        )

        assert round_result.keywords_used == []
        assert round_result.results == []
        assert round_result.files_searched == 0
