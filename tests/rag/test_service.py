"""Tests for RAG class."""

import tempfile
from pathlib import Path

import pytest

from houyi.rag import RAG
from houyi.rag.config import RAGConfig
from houyi.rag.types import RAGMode


class TestRAG:
    """Tests for RAG class."""

    def test_creation_default(self) -> None:
        """Test creating RAG with defaults."""
        rag = RAG()
        assert rag.config.mode == RAGMode.AUTO

    def test_creation_agentic(self) -> None:
        """Test creating RAG in agentic mode."""
        rag = RAG(mode="agentic")
        assert rag.config.mode == RAGMode.AGENTIC

    def test_creation_indexed(self) -> None:
        """Test creating RAG in indexed mode."""
        rag = RAG(mode="indexed")
        assert rag.config.mode == RAGMode.INDEXED

    def test_with_config(self) -> None:
        """Test creating RAG with config object."""
        config = RAGConfig(
            mode=RAGMode.AGENTIC,
            knowledge_dir="/custom/path",
        )
        rag = RAG(config=config)
        assert rag.config.mode == RAGMode.AGENTIC
        assert rag.config.knowledge_dir == "/custom/path"

    def test_with_strategies(self) -> None:
        """Test creating RAG with custom strategies."""
        rag = RAG(
            mode="indexed",
            strategies=["bm25", "vector"],
        )
        assert rag.config.mode == RAGMode.INDEXED

    def test_mode_selection_empty_dir(self) -> None:
        """Test mode selection for non-existent directory."""
        rag = RAG(knowledge_dir="/nonexistent/path")
        mode = rag._select_mode("test query")
        assert mode == RAGMode.AGENTIC

    def test_mode_selection_small_dir(self) -> None:
        """Test mode selection for small directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a few files
            for i in range(10):
                Path(tmpdir, f"file{i}.txt").write_text(f"Content {i}")

            rag = RAG(knowledge_dir=tmpdir)
            mode = rag._select_mode("test query")
            assert mode == RAGMode.AGENTIC

    def test_knowledge_dir_property(self) -> None:
        """Test knowledge_dir property."""
        rag = RAG(knowledge_dir="/test/path")
        assert rag.knowledge_dir == "/test/path"

    def test_llm_string_parsing(self) -> None:
        """Test LLM string parsing (provider:model format)."""
        # Just provider
        rag = RAG(llm="openai")
        assert rag._llm_adapter is None  # Would need API key

        # Provider:model format - doesn't create adapter without credentials
        rag2 = RAG(llm="anthropic:claude-3-opus")
        assert rag2._llm_adapter is None


class TestRAGConfig:
    """Tests for RAG configuration handling."""

    def test_config_property(self) -> None:
        """Test config property."""
        rag = RAG(mode="agentic", knowledge_dir="/test")
        assert rag.config.mode == RAGMode.AGENTIC
        assert rag.config.knowledge_dir == "/test"

    def test_extra_kwargs(self) -> None:
        """Test passing extra kwargs."""
        rag = RAG(
            mode="indexed",
            llm_model="gpt-4",
        )
        assert rag.config.llm_model == "gpt-4"


class TestRAGIndex:
    """Tests for RAG.index() method."""

    @pytest.mark.asyncio
    async def test_index_agentic_mode_noop(self) -> None:
        """Test that index() is a no-op for Agentic mode."""
        rag = RAG(mode="agentic")
        stats = await rag.index()
        assert stats["mode"] == "agentic"
        assert stats["documents"] == 0

    @pytest.mark.asyncio
    async def test_index_with_empty_dir(self) -> None:
        """Test indexing empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAG(knowledge_dir=tmpdir, mode="indexed")
            stats = await rag.index()
            assert stats["documents"] == 0
            assert stats["chunks"] == 0


class TestRAGLLMIntegration:
    """Tests for LLM integration."""

    def test_llm_adapter_param(self) -> None:
        """Test llm_adapter parameter."""
        from unittest.mock import MagicMock

        mock_adapter = MagicMock()
        rag = RAG(llm_adapter=mock_adapter)
        assert rag._llm_adapter is mock_adapter

    def test_llm_provider_param(self) -> None:
        """Test llm_provider parameter."""
        # Won't actually create adapter without credentials, but should not error
        rag = RAG(llm_provider="openai", llm_model="gpt-4")
        # The adapter creation will fail silently without API key
        assert rag.config is not None
