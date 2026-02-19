"""Tests for Query Analyzer."""

import pytest

from houyi.rag.processors.query_analyzer import (
    QueryAnalysis,
    QueryAnalyzer,
    QueryType,
    analyze_query,
)
from houyi.rag.types import RetrievalStrategy


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer."""

    def test_classify_factual_query(self) -> None:
        """Test factual query classification."""
        analyzer = QueryAnalyzer()

        result = analyzer._classify_query_type("what is the capital of france")
        assert result == QueryType.FACTUAL

        result = analyzer._classify_query_type("who invented the telephone")
        assert result == QueryType.FACTUAL

        result = analyzer._classify_query_type("when did world war 2 end")
        assert result == QueryType.FACTUAL

        result = analyzer._classify_query_type("where is the eiffel tower")
        assert result == QueryType.FACTUAL

    def test_classify_procedural_query(self) -> None:
        """Test procedural query classification."""
        analyzer = QueryAnalyzer()

        result = analyzer._classify_query_type("how to install python")
        assert result == QueryType.PROCEDURAL

        result = analyzer._classify_query_type("how do i create a function")
        assert result == QueryType.PROCEDURAL

        result = analyzer._classify_query_type("how can i debug this code")
        assert result == QueryType.PROCEDURAL

    def test_classify_analytical_query(self) -> None:
        """Test analytical query classification."""
        analyzer = QueryAnalyzer()

        result = analyzer._classify_query_type("why is python popular")
        assert result == QueryType.ANALYTICAL

        result = analyzer._classify_query_type("compare react and vue")
        assert result == QueryType.ANALYTICAL

        result = analyzer._classify_query_type("what is the difference between tcp and udp")
        assert result == QueryType.ANALYTICAL

    def test_classify_keyword_query(self) -> None:
        """Test keyword query classification."""
        analyzer = QueryAnalyzer()

        # Short queries without question words
        result = analyzer._classify_query_type("python")
        assert result == QueryType.KEYWORD

        result = analyzer._classify_query_type("docker install")
        assert result == QueryType.KEYWORD

    def test_classify_exploratory_query(self) -> None:
        """Test exploratory query classification."""
        analyzer = QueryAnalyzer()

        # Longer, unclassified queries default to exploratory
        result = analyzer._classify_query_type(
            "tell me everything about machine learning and deep learning"
        )
        assert result == QueryType.EXPLORATORY

    def test_extract_keywords(self) -> None:
        """Test keyword extraction."""
        analyzer = QueryAnalyzer()

        keywords = analyzer._extract_keywords("what is machine learning in python")
        assert "machine" in keywords
        assert "learning" in keywords
        assert "python" in keywords
        # Stop words should be filtered
        assert "what" not in keywords
        assert "is" not in keywords
        assert "in" not in keywords

    def test_extract_entities(self) -> None:
        """Test entity extraction."""
        analyzer = QueryAnalyzer()

        # Quoted phrases
        entities = analyzer._extract_entities('search for "machine learning"')
        assert "machine learning" in entities

        # CamelCase
        entities = analyzer._extract_entities("what is TensorFlow")
        assert "TensorFlow" in entities

    def test_select_strategies_factual(self) -> None:
        """Test strategy selection for factual queries."""
        analyzer = QueryAnalyzer()

        strategies = analyzer._select_strategies(QueryType.FACTUAL, ["capital"], [])
        assert RetrievalStrategy.BM25 in strategies
        assert RetrievalStrategy.VECTOR in strategies

    def test_select_strategies_with_entities(self) -> None:
        """Test strategy selection adds graph when entities present."""
        analyzer = QueryAnalyzer()

        strategies = analyzer._select_strategies(QueryType.KEYWORD, ["python"], ["TensorFlow"])
        # Should add graph because entities were detected
        assert RetrievalStrategy.GRAPH in strategies

    @pytest.mark.asyncio
    async def test_analyze_heuristic(self) -> None:
        """Test full heuristic analysis."""
        analyzer = QueryAnalyzer()

        result = await analyzer.analyze("how to install python on windows")

        assert isinstance(result, QueryAnalysis)
        assert result.query_type == QueryType.PROCEDURAL
        assert "install" in result.keywords or "python" in result.keywords
        assert result.intent == "learn_process"
        assert len(result.recommended_strategies) > 0
        assert result.confidence > 0
        assert result.metadata.get("analysis_method") == "heuristic"

    @pytest.mark.asyncio
    async def test_analyze_query_convenience(self) -> None:
        """Test analyze_query convenience function."""
        result = await analyze_query("what is RAG")

        assert isinstance(result, QueryAnalysis)
        assert result.query == "what is RAG"

    def test_query_analysis_to_dict(self) -> None:
        """Test QueryAnalysis.to_dict() for UI serialization."""
        analysis = QueryAnalysis(
            query="test query",
            query_type=QueryType.FACTUAL,
            keywords=["test"],
            entities=["Entity"],
            intent="find_specific_fact",
            recommended_strategies=[RetrievalStrategy.BM25, RetrievalStrategy.VECTOR],
            confidence=0.8,
            metadata={"test": "value"},
        )

        result = analysis.to_dict()

        assert result["query"] == "test query"
        assert result["query_type"] == "factual"
        assert result["keywords"] == ["test"]
        assert result["entities"] == ["Entity"]
        assert result["intent"] == "find_specific_fact"
        assert result["recommended_strategies"] == ["bm25", "vector"]
        assert result["confidence"] == 0.8
        assert result["metadata"]["test"] == "value"

    def test_query_type_enum(self) -> None:
        """Test QueryType enum values."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.CONCEPTUAL.value == "conceptual"
        assert QueryType.PROCEDURAL.value == "procedural"
        assert QueryType.ANALYTICAL.value == "analytical"
        assert QueryType.KEYWORD.value == "keyword"
        assert QueryType.EXPLORATORY.value == "exploratory"


class TestQueryAnalyzerWithLLM:
    """Tests for QueryAnalyzer with LLM."""

    @pytest.mark.asyncio
    async def test_analyze_with_llm_fallback(self) -> None:
        """Test that LLM failure falls back to heuristic."""
        from typing import Any

        class FailingAdapter:
            async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
                raise RuntimeError("LLM unavailable")

        analyzer = QueryAnalyzer(adapter=FailingAdapter())  # type: ignore[arg-type]
        result = await analyzer.analyze("how to test code")

        # Should fall back to heuristic
        assert result.metadata.get("analysis_method") == "heuristic"
        assert result.query_type == QueryType.PROCEDURAL

    @pytest.mark.asyncio
    async def test_analyze_with_mock_llm(self) -> None:
        """Test analysis with mock LLM."""
        from typing import Any

        class MockAdapter:
            async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
                class MockResponse:
                    content = """{
                        "query_type": "procedural",
                        "keywords": ["install", "python"],
                        "entities": ["Python"],
                        "intent": "learn installation process",
                        "strategies": ["vector", "bm25"],
                        "confidence": 0.9
                    }"""

                return MockResponse()

        analyzer = QueryAnalyzer(adapter=MockAdapter())  # type: ignore[arg-type]
        result = await analyzer.analyze("how to install python")

        assert result.metadata.get("analysis_method") == "llm"
        assert result.query_type == QueryType.PROCEDURAL
        assert result.confidence == 0.9
        assert RetrievalStrategy.VECTOR in result.recommended_strategies
        assert RetrievalStrategy.BM25 in result.recommended_strategies
