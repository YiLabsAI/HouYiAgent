"""Tests for RAGService."""

import tempfile
from pathlib import Path

from houyi.rag.config import RAGConfig
from houyi.rag.service import RAGService
from houyi.rag.types import RAGMode


class TestRAGService:
    """Tests for RAGService."""

    def test_service_creation_default(self) -> None:
        """Test creating RAGService with defaults."""
        service = RAGService()
        assert service.config.mode == RAGMode.AUTO

    def test_service_creation_agentic(self) -> None:
        """Test creating RAGService in agentic mode."""
        service = RAGService(mode="agentic")
        assert service.config.mode == RAGMode.AGENTIC

    def test_service_creation_indexed(self) -> None:
        """Test creating RAGService in indexed mode."""
        service = RAGService(mode="indexed")
        assert service.config.mode == RAGMode.INDEXED

    def test_service_with_config(self) -> None:
        """Test creating RAGService with config object."""
        config = RAGConfig(
            mode=RAGMode.AGENTIC,
            knowledge_dir="/custom/path",
        )
        service = RAGService(config=config)
        assert service.config.mode == RAGMode.AGENTIC
        assert service.config.knowledge_dir == "/custom/path"

    def test_service_with_retrieval_strategies(self) -> None:
        """Test creating RAGService with custom strategies."""
        service = RAGService(
            mode="indexed",
            retrieval=["bm25", "vector"],
        )
        assert service.config.mode == RAGMode.INDEXED

    def test_mode_selection_empty_dir(self) -> None:
        """Test mode selection for non-existent directory."""
        service = RAGService(knowledge_dir="/nonexistent/path")
        mode = service._select_mode("test query")
        assert mode == RAGMode.AGENTIC

    def test_mode_selection_small_dir(self) -> None:
        """Test mode selection for small directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a few files
            for i in range(10):
                Path(tmpdir, f"file{i}.txt").write_text(f"Content {i}")

            service = RAGService(knowledge_dir=tmpdir)
            mode = service._select_mode("test query")
            assert mode == RAGMode.AGENTIC


class TestRAGServiceConfig:
    """Tests for RAGService configuration handling."""

    def test_config_property(self) -> None:
        """Test config property."""
        service = RAGService(mode="agentic", knowledge_dir="/test")
        assert service.config.mode == RAGMode.AGENTIC
        assert service.config.knowledge_dir == "/test"

    def test_extra_kwargs(self) -> None:
        """Test passing extra kwargs."""
        service = RAGService(
            mode="indexed",
            llm_model="gpt-4",
        )
        assert service.config.llm_model == "gpt-4"
