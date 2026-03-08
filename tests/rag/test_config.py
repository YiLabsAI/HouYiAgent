"""Tests for RAG configuration models."""

from __future__ import annotations

from pathlib import Path

from houyi.rag.config import AgenticConfig, EmbeddingConfig, GraphConfig, RAGConfig


class TestEmbeddingConfig:
    def test_creation(self) -> None:
        config = EmbeddingConfig(
            provider="local",
            dimension=384,
        )
        assert config.provider == "local"
        assert config.dimension == 384


class TestAgenticConfig:
    def test_default_values(self) -> None:
        config = AgenticConfig()
        assert config.max_rounds == 5
        assert config.index_file == "data_structure.md"
        assert config.chunk_limit == 500

    def test_custom_values(self) -> None:
        config = AgenticConfig(
            max_rounds=3,
            index_file="index.md",
            chunk_limit=200,
        )
        assert config.max_rounds == 3
        assert config.index_file == "index.md"
        assert config.chunk_limit == 200


class TestGraphConfig:
    def test_creation(self) -> None:
        config = GraphConfig(
            enabled=True,
            ppr_alpha=0.85,
            ppr_top_k=50,
        )
        assert config.enabled is True
        assert config.ppr_alpha == 0.85
        assert config.ppr_top_k == 50


class TestRAGConfig:
    def test_index_dir_custom_value(self) -> None:
        config = RAGConfig(
            knowledge_dir="/path/to/knowledge",
            index_dir="/path/to/index",
        )

        assert config.index_dir == "/path/to/index"
        assert config.get_index_dir() == "/path/to/index"

    def test_index_dir_default(self) -> None:
        knowledge_dir = str(Path("/path/to/knowledge"))
        config = RAGConfig(knowledge_dir=knowledge_dir)

        assert config.index_dir is None
        expected = str(Path(knowledge_dir) / ".houyi")
        assert config.get_index_dir() == expected
