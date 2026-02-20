"""Tests for RAG types and configuration."""

import pytest

from houyi.rag.config import AgenticConfig, IndexedConfig, RAGConfig
from houyi.rag.types import (
    Chunk,
    Document,
    Entity,
    QualitySummary,
    RAGMode,
    Relation,
    RetrievalResult,
    RetrievalStrategy,
    SearchResult,
)


class TestRAGMode:
    """Tests for RAGMode enum."""

    def test_mode_values(self) -> None:
        """Test RAGMode enum values."""
        assert RAGMode.AGENTIC == "agentic"
        assert RAGMode.INDEXED == "indexed"
        assert RAGMode.AUTO == "auto"

    def test_mode_from_string(self) -> None:
        """Test creating RAGMode from string."""
        assert RAGMode("agentic") == RAGMode.AGENTIC
        assert RAGMode("indexed") == RAGMode.INDEXED
        assert RAGMode("auto") == RAGMode.AUTO


class TestRetrievalStrategy:
    """Tests for RetrievalStrategy enum."""

    def test_strategy_values(self) -> None:
        """Test RetrievalStrategy enum values."""
        assert RetrievalStrategy.BM25 == "bm25"
        assert RetrievalStrategy.VECTOR == "vector"
        assert RetrievalStrategy.GRAPH == "graph"


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self) -> None:
        """Test creating a Document."""
        doc = Document(
            doc_id="doc1",
            content="Test content",
            source="/path/to/file.txt",
        )
        assert doc.doc_id == "doc1"
        assert doc.content == "Test content"
        assert doc.source == "/path/to/file.txt"
        assert doc.doc_type == "text"
        assert doc.metadata == {}

    def test_document_with_metadata(self) -> None:
        """Test Document with metadata."""
        doc = Document(
            doc_id="doc2",
            content="Content",
            metadata={"author": "Test Author"},
        )
        assert doc.metadata["author"] == "Test Author"


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation(self) -> None:
        """Test creating a Chunk."""
        chunk = Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            content="Chunk content",
        )
        assert chunk.chunk_id == "chunk1"
        assert chunk.doc_id == "doc1"
        assert chunk.content == "Chunk content"
        assert chunk.embedding is None

    def test_chunk_with_embedding(self) -> None:
        """Test Chunk with embedding."""
        embedding = [0.1, 0.2, 0.3]
        chunk = Chunk(
            chunk_id="chunk2",
            doc_id="doc1",
            content="Content",
            embedding=embedding,
        )
        assert chunk.embedding == embedding


class TestEntity:
    """Tests for Entity model."""

    def test_entity_creation(self) -> None:
        """Test creating an Entity."""
        entity = Entity(
            entity_id="e1",
            name="Test Entity",
            entity_type="concept",
        )
        assert entity.entity_id == "e1"
        assert entity.name == "Test Entity"
        assert entity.entity_type == "concept"


class TestRelation:
    """Tests for Relation model."""

    def test_relation_creation(self) -> None:
        """Test creating a Relation."""
        rel = Relation(
            rel_id="r1",
            source_id="e1",
            target_id="e2",
            rel_type="causes",
        )
        assert rel.rel_id == "r1"
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.rel_type == "causes"
        assert rel.weight == 1.0


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result_creation(self) -> None:
        """Test creating a SearchResult."""
        result = SearchResult(
            content="Result content",
            score=0.85,
        )
        assert result.content == "Result content"
        assert result.score == 0.85
        assert result.source is None


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_retrieval_result_creation(self) -> None:
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            answer="Test answer",
            confidence=0.9,
        )
        assert result.answer == "Test answer"
        assert result.confidence == 0.9
        assert result.sources == []
        assert result.mode_used == RAGMode.AUTO


class TestRAGConfig:
    """Tests for RAGConfig."""

    def test_default_config(self) -> None:
        """Test default RAGConfig."""
        config = RAGConfig()
        assert config.mode == RAGMode.AUTO
        assert config.knowledge_dir == "knowledge/"
        assert isinstance(config.agentic, AgenticConfig)
        assert isinstance(config.indexed, IndexedConfig)

    def test_for_agentic(self) -> None:
        """Test creating agentic config."""
        config = RAGConfig.for_agentic(knowledge_dir="/custom/path")
        assert config.mode == RAGMode.AGENTIC
        assert config.knowledge_dir == "/custom/path"

    def test_for_indexed(self) -> None:
        """Test creating indexed config."""
        config = RAGConfig.for_indexed(
            knowledge_dir="/data",
            strategies=[RetrievalStrategy.VECTOR],
        )
        assert config.mode == RAGMode.INDEXED
        assert config.knowledge_dir == "/data"
        assert RetrievalStrategy.VECTOR in config.indexed.strategies


class TestAgenticConfig:
    """Tests for AgenticConfig."""

    def test_default_values(self) -> None:
        """Test default AgenticConfig values."""
        config = AgenticConfig()
        assert config.max_rounds == 5
        assert config.index_file == "data_structure.md"
        assert config.chunk_limit == 500


class TestIndexedConfig:
    """Tests for IndexedConfig."""

    def test_default_values(self) -> None:
        """Test default IndexedConfig values."""
        config = IndexedConfig()
        assert config.top_k == 10
        assert config.use_rerank is True
        assert config.use_crag is True
        assert RetrievalStrategy.BM25 in config.strategies
        assert RetrievalStrategy.VECTOR in config.strategies


class TestQualitySummary:
    """Tests for QualitySummary model (v1.1)."""

    def test_quality_summary_defaults(self) -> None:
        """Test QualitySummary default values."""
        qs = QualitySummary()
        assert qs.min_score == 0.0
        assert qs.max_score == 0.0
        assert qs.avg_score == 0.0
        assert qs.relevance == "unknown"
        assert qs.coverage == "unknown"
        assert qs.confidence_level == "unknown"
        assert qs.suggestion is None

    def test_quality_summary_from_scores_high(self) -> None:
        """Test QualitySummary.from_scores with high scores."""
        scores = [0.9, 0.85, 0.8, 0.75, 0.7]
        qs = QualitySummary.from_scores(scores)

        assert qs.min_score == 0.7
        assert qs.max_score == 0.9
        assert qs.avg_score == 0.8
        assert qs.relevance == "high"
        assert qs.coverage == "high"
        assert qs.above_threshold_count == 5
        assert qs.total_count == 5

    def test_quality_summary_from_scores_medium(self) -> None:
        """Test QualitySummary.from_scores with medium scores."""
        scores = [0.7, 0.6, 0.5, 0.4, 0.3]
        qs = QualitySummary.from_scores(scores)

        assert qs.min_score == 0.3
        assert qs.max_score == 0.7
        assert qs.avg_score == pytest.approx(0.5)
        assert qs.relevance == "medium"
        assert qs.above_threshold_count == 2

    def test_quality_summary_from_scores_low(self) -> None:
        """Test QualitySummary.from_scores with low scores."""
        scores = [0.3, 0.2, 0.1, 0.15, 0.25]
        qs = QualitySummary.from_scores(scores)

        assert qs.avg_score == pytest.approx(0.2)
        assert qs.relevance == "low"
        assert qs.above_threshold_count == 0
        assert qs.suggestion is not None  # Should have improvement suggestion

    def test_quality_summary_from_empty_scores(self) -> None:
        """Test QualitySummary.from_scores with empty list."""
        qs = QualitySummary.from_scores([])
        assert qs.total_count == 0
        assert qs.relevance == "unknown"

    def test_quality_summary_score_distribution(self) -> None:
        """Test QualitySummary score distribution buckets."""
        scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
        qs = QualitySummary.from_scores(scores)

        assert qs.score_distribution["80-100"] == 2
        assert qs.score_distribution["60-80"] == 2
        assert qs.score_distribution["40-60"] == 2
        assert qs.score_distribution["20-40"] == 2
        assert qs.score_distribution["0-20"] == 2


class TestRetrievalResultWithQuality:
    """Tests for RetrievalResult with quality field (v1.1)."""

    def test_retrieval_result_with_quality(self) -> None:
        """Test RetrievalResult includes quality field."""
        import pytest

        quality = QualitySummary.from_scores([0.8, 0.7, 0.6])
        result = RetrievalResult(
            answer="Test answer",
            confidence=0.75,
            quality=quality,
        )
        assert result.quality is not None
        assert result.quality.avg_score == pytest.approx(0.7)
        assert result.quality.relevance == "high"

    def test_retrieval_result_quality_none_by_default(self) -> None:
        """Test RetrievalResult quality is None by default."""
        result = RetrievalResult(answer="Test")
        assert result.quality is None
