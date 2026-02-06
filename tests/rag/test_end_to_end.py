"""End-to-end tests for RAG system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from houyi.rag import RAGService, search
from houyi.rag.skills.kb_search import KBSearchInput, execute_kb_search


class FakeAdapter:
    """Fake LLM adapter for testing."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize with a list of responses to return."""
        self._responses = responses
        self._index = 0

    async def chat(
        self,
        messages: list[Any],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Any:
        """Mock chat method."""
        response_content = self._responses[self._index % len(self._responses)]
        self._index += 1

        class MockResponse:
            content = response_content

        return MockResponse()


class TestEndToEndAgentic:
    """End-to-end tests for Agentic mode."""

    @pytest.mark.asyncio
    async def test_agentic_search_basic(self) -> None:
        """Test basic agentic search with knowledge directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create test files
            (kb_dir / "rag_overview.md").write_text(
                "# RAG Overview\n\nRAG stands for Retrieval-Augmented Generation.\n"
                "It combines retrieval with generation to provide accurate answers."
            )
            (kb_dir / "implementation.md").write_text(
                "# Implementation\n\nTo implement RAG, you need:\n"
                "1. A vector database\n2. An embedding model\n3. An LLM"
            )

            service = RAGService(mode="agentic", knowledge_dir=str(kb_dir))
            result = await service.query("What is RAG?")

            assert result.answer
            assert result.mode_used.value == "agentic"

    @pytest.mark.asyncio
    async def test_agentic_search_empty_kb(self) -> None:
        """Test agentic search with empty knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            service = RAGService(mode="agentic", knowledge_dir=str(kb_dir))
            result = await service.query("What is RAG?")

            assert "未" in result.answer  # Should indicate no results
            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_agentic_search_with_llm(self) -> None:
        """Test agentic search with LLM adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            (kb_dir / "doc.md").write_text("RAG is a powerful technique for AI systems.")

            fake_adapter = FakeAdapter(
                [
                    '{"keywords": ["RAG", "technique"], "synonyms": {}}',
                    "RAG is a technique that combines retrieval with generation. [1]",
                ]
            )

            service = RAGService(
                mode="agentic",
                knowledge_dir=str(kb_dir),
                llm_adapter=fake_adapter,
            )
            result = await service.query("What is RAG?")

            assert result.answer


class TestEndToEndIndexed:
    """End-to-end tests for Indexed mode."""

    @pytest.mark.asyncio
    async def test_indexed_mode_rrf_fusion(self) -> None:
        """Test indexed mode with RRF fusion logic."""
        from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
        from houyi.rag.indexed.mode import IndexedMode
        from houyi.rag.types import RetrievalStrategy

        config = IndexedConfig(
            strategies=[RetrievalStrategy.BM25],
            use_rerank=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            mode = IndexedMode(
                config=config,
                knowledge_dir=str(kb_dir),
                embedding_config=EmbeddingConfig(),
                graph_config=GraphConfig(),
            )

            # Test RRF fusion logic
            from houyi.rag.types import SearchResult

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

            # "b" should be ranked highest (appears in both lists)
            assert len(fused) == 3
            assert fused[0].chunk_id == "b"


class TestEndToEndSearch:
    """End-to-end tests for the search() function."""

    @pytest.mark.asyncio
    async def test_search_function(self) -> None:
        """Test the one-liner search function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            (kb_dir / "doc.md").write_text("The answer is 42.")

            result = await search("What is the answer?", knowledge_dir=str(kb_dir))

            assert result.answer


class TestEndToEndSkill:
    """End-to-end tests for kb_search skill."""

    @pytest.mark.asyncio
    async def test_skill_execution(self) -> None:
        """Test executing kb_search skill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            (kb_dir / "doc.md").write_text("Python is a programming language.")

            input_data = KBSearchInput(
                query="What is Python?",
                knowledge_dir=str(kb_dir),
                mode="agentic",
            )

            output = await execute_kb_search(input_data)

            assert output.answer
            assert isinstance(output.confidence, float)


class TestModeSelection:
    """Tests for automatic mode selection."""

    @pytest.mark.asyncio
    async def test_auto_selects_agentic_for_small_kb(self) -> None:
        """Test that auto mode selects agentic for small knowledge bases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create a few files (< 100)
            for i in range(10):
                (kb_dir / f"doc{i}.md").write_text(f"Document {i} content")

            service = RAGService(mode="auto", knowledge_dir=str(kb_dir))

            # Mode selection happens during query
            selected = service._select_mode("test query")

            assert selected.value == "agentic"

    @pytest.mark.asyncio
    async def test_auto_selects_indexed_for_large_kb(self) -> None:
        """Test that auto mode selects indexed for large knowledge bases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create many files (> 100)
            for i in range(120):
                (kb_dir / f"doc{i}.md").write_text(f"Document {i} content")

            service = RAGService(mode="auto", knowledge_dir=str(kb_dir))

            selected = service._select_mode("test query")

            assert selected.value == "indexed"


class TestLLMProviderIntegration:
    """Tests for LLM provider integration."""

    @pytest.mark.asyncio
    async def test_service_with_llm_adapter(self) -> None:
        """Test service with injected LLM adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            (kb_dir / "doc.md").write_text("Test content about AI")

            fake_adapter = FakeAdapter(
                [
                    '{"keywords": ["AI"], "synonyms": {}}',
                    "AI stands for Artificial Intelligence.",
                ]
            )

            service = RAGService(
                mode="agentic",
                knowledge_dir=str(kb_dir),
                llm_adapter=fake_adapter,
            )

            result = await service.query("What is AI?")

            assert result.answer
            # Verify adapter was called
            assert fake_adapter._index > 0

    @pytest.mark.asyncio
    async def test_service_with_llm_model_config(self) -> None:
        """Test service stores llm_model in config."""
        service = RAGService(
            mode="indexed",
            llm_model="gpt-4",
        )

        assert service.config.llm_model == "gpt-4"
