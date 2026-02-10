"""Smoke tests for RAG system.

These tests verify basic functionality of the RAG system
without requiring external services (LLM, embedding APIs).
Run these first to ensure the system is working correctly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.rag import RAG, RAGConfig, search
from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
from houyi.rag.indexed.mode import IndexedMode
from houyi.rag.skills.kb_search import KBSearchInput, execute_kb_search
from houyi.rag.types import RAGMode, RetrievalStrategy, SearchResult


class TestSmokeRAGCore:
    """Smoke tests for RAG core functionality."""

    def test_rag_import(self) -> None:
        """Test RAG can be imported."""
        from houyi.rag import RAG

        assert RAG is not None

    def test_rag_instantiation_agentic(self) -> None:
        """Test RAG can be instantiated in agentic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAG(mode="agentic", knowledge_dir=tmpdir)
            assert rag is not None
            assert rag.config.mode == RAGMode.AGENTIC

    def test_rag_instantiation_indexed(self) -> None:
        """Test RAG can be instantiated in indexed mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAG(mode="indexed", knowledge_dir=tmpdir)
            assert rag is not None
            assert rag.config.mode == RAGMode.INDEXED

    def test_rag_instantiation_auto(self) -> None:
        """Test RAG can be instantiated in auto mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAG(mode="auto", knowledge_dir=tmpdir)
            assert rag is not None
            assert rag.config.mode == RAGMode.AUTO


class TestSmokeAgenticMode:
    """Smoke tests for Agentic mode."""

    @pytest.mark.asyncio
    async def test_agentic_query_simple(self) -> None:
        """Test simple agentic query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            (kb_dir / "test.md").write_text("Python is a programming language.")

            rag = RAG(mode="agentic", knowledge_dir=str(kb_dir))
            result = await rag.query("What is Python?")

            assert result is not None
            assert result.answer is not None
            assert result.mode_used == RAGMode.AGENTIC

    @pytest.mark.asyncio
    async def test_agentic_query_empty_kb(self) -> None:
        """Test agentic query on empty knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            rag = RAG(mode="agentic", knowledge_dir=str(kb_dir))
            result = await rag.query("What is Python?")

            assert result is not None
            # Should return no relevant info
            assert result.confidence == 0.0


class TestSmokeIndexedMode:
    """Smoke tests for Indexed mode."""

    def test_indexed_mode_instantiation(self) -> None:
        """Test IndexedMode can be instantiated."""
        config = IndexedConfig(strategies=[RetrievalStrategy.BM25])
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=config,
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(),
                graph_config=GraphConfig(),
            )
            assert mode is not None

    def test_rrf_fusion(self) -> None:
        """Test RRF fusion logic."""
        config = IndexedConfig(strategies=[RetrievalStrategy.BM25])
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=config,
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(),
                graph_config=GraphConfig(),
            )

            results1 = [
                SearchResult(chunk_id="a", content="A", score=0.9),
                SearchResult(chunk_id="b", content="B", score=0.8),
            ]
            results2 = [
                SearchResult(chunk_id="b", content="B", score=0.95),
                SearchResult(chunk_id="c", content="C", score=0.7),
            ]

            fused = mode._rrf_fusion(
                [("vector", results1), ("bm25", results2)],
                k=3,
            )

            assert len(fused) == 3
            # b should rank highest (appears in both)
            assert fused[0].chunk_id == "b"


class TestSmokeSearch:
    """Smoke tests for search function."""

    @pytest.mark.asyncio
    async def test_search_function(self) -> None:
        """Test one-liner search function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            (kb_dir / "test.md").write_text("The answer is 42.")

            result = await search("What is the answer?", knowledge_dir=str(kb_dir))

            assert result is not None
            assert result.answer is not None


class TestSmokeSkills:
    """Smoke tests for RAG skills."""

    def test_skill_imports(self) -> None:
        """Test all RAG skills can be imported."""
        from houyi.rag.skills import (
            kb_analyze_skill,
            kb_graph_skill,
            kb_ingest_skill,
            kb_search_skill,
        )

        assert kb_search_skill is not None
        assert kb_ingest_skill is not None
        assert kb_graph_skill is not None
        assert kb_analyze_skill is not None

    def test_skill_definitions(self) -> None:
        """Test skill definitions are valid."""
        from houyi.rag.skills import (
            kb_analyze_skill,
            kb_graph_skill,
            kb_ingest_skill,
            kb_search_skill,
        )

        assert kb_search_skill.name == "kb-search"
        assert kb_ingest_skill.name == "kb-ingest"
        assert kb_graph_skill.name == "kb-graph"
        assert kb_analyze_skill.name == "kb-analyze"

    @pytest.mark.asyncio
    async def test_kb_search_skill_execution(self) -> None:
        """Test kb_search skill execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            (kb_dir / "test.md").write_text("Python is great.")

            input_data = KBSearchInput(
                query="What is Python?",
                knowledge_dir=str(kb_dir),
                mode="agentic",
            )

            output = await execute_kb_search(input_data)

            assert output is not None
            assert output.answer is not None


class TestSmokeTypes:
    """Smoke tests for RAG types."""

    def test_search_result(self) -> None:
        """Test SearchResult type."""
        result = SearchResult(
            chunk_id="test",
            content="Test content",
            score=0.9,
        )
        assert result.chunk_id == "test"
        assert result.score == 0.9

    def test_rag_config(self) -> None:
        """Test RAGConfig type."""
        config = RAGConfig(mode=RAGMode.AGENTIC)
        assert config.mode == RAGMode.AGENTIC


class TestSmokeConfig:
    """Smoke tests for RAG configuration."""

    def test_embedding_config(self) -> None:
        """Test EmbeddingConfig."""
        config = EmbeddingConfig(
            provider="local",
            dimension=384,
        )
        assert config.provider == "local"
        assert config.dimension == 384

    def test_indexed_config(self) -> None:
        """Test IndexedConfig."""
        config = IndexedConfig(
            strategies=[RetrievalStrategy.BM25, RetrievalStrategy.VECTOR],
            use_rerank=False,
        )
        assert RetrievalStrategy.BM25 in config.strategies
        assert config.use_rerank is False

    def test_graph_config(self) -> None:
        """Test GraphConfig."""
        config = GraphConfig(
            enabled=True,
            ppr_alpha=0.85,
            ppr_top_k=50,
        )
        assert config.enabled is True
        assert config.ppr_alpha == 0.85


class TestSmokeModeSelection:
    """Smoke tests for automatic mode selection."""

    def test_small_kb_selects_agentic(self) -> None:
        """Test auto mode selects agentic for small KB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            for i in range(10):
                (kb_dir / f"doc{i}.md").write_text(f"Content {i}")

            rag = RAG(mode="auto", knowledge_dir=str(kb_dir))
            selected = rag._select_mode("test")

            assert selected == RAGMode.AGENTIC

    def test_large_kb_selects_indexed(self) -> None:
        """Test auto mode selects indexed for large KB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            for i in range(150):
                (kb_dir / f"doc{i}.md").write_text(f"Content {i}")

            rag = RAG(mode="auto", knowledge_dir=str(kb_dir))
            selected = rag._select_mode("test")

            assert selected == RAGMode.INDEXED
