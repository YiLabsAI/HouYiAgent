"""Tests for RAG benchmark metrics."""

from __future__ import annotations

import pytest

from tests.rag.benchmark.metrics import (
    BenchmarkMetrics,
    aggregate_metrics,
    calculate_metrics,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


class TestPrecisionAtK:
    """Tests for Precision@K calculation."""

    def test_perfect_precision(self) -> None:
        """Test perfect precision when all retrieved are relevant."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c", "d", "e"}
        assert calculate_precision_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_precision(self) -> None:
        """Test zero precision when none retrieved are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y", "z"}
        assert calculate_precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self) -> None:
        """Test partial precision."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        assert calculate_precision_at_k(retrieved, relevant, 5) == 0.6

    def test_precision_at_1(self) -> None:
        """Test P@1."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert calculate_precision_at_k(retrieved, relevant, 1) == 1.0

        retrieved = ["b", "a", "c"]
        assert calculate_precision_at_k(retrieved, relevant, 1) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """Test when k > len(retrieved)."""
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # Only 2 documents, but asking for P@5
        assert calculate_precision_at_k(retrieved, relevant, 5) == 0.4

    def test_invalid_k(self) -> None:
        """Test with invalid k values."""
        retrieved = ["a", "b"]
        relevant = {"a"}
        assert calculate_precision_at_k(retrieved, relevant, 0) == 0.0
        assert calculate_precision_at_k(retrieved, relevant, -1) == 0.0


class TestRecallAtK:
    """Tests for Recall@K calculation."""

    def test_perfect_recall(self) -> None:
        """Test perfect recall when all relevant are retrieved."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        assert calculate_recall_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_recall(self) -> None:
        """Test zero recall when none relevant are retrieved."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert calculate_recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self) -> None:
        """Test partial recall."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c", "d", "e"}
        assert calculate_recall_at_k(retrieved, relevant, 3) == 0.5

    def test_empty_relevant(self) -> None:
        """Test with no relevant documents."""
        retrieved = ["a", "b", "c"]
        relevant: set[str] = set()
        assert calculate_recall_at_k(retrieved, relevant, 3) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank calculation."""

    def test_first_position(self) -> None:
        """Test MRR when relevant doc is first."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == 1.0

    def test_second_position(self) -> None:
        """Test MRR when relevant doc is second."""
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == 0.5

    def test_third_position(self) -> None:
        """Test MRR when relevant doc is third."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_found(self) -> None:
        """Test MRR when no relevant doc is found."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == 0.0

    def test_multiple_relevant(self) -> None:
        """Test MRR with multiple relevant (uses first found)."""
        retrieved = ["x", "a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert calculate_mrr(retrieved, relevant) == 0.5


class TestNDCG:
    """Tests for NDCG calculation."""

    def test_perfect_ranking(self) -> None:
        """Test NDCG with perfect ranking."""
        retrieved = ["a", "b", "c"]
        scores = {"a": 1.0, "b": 0.8, "c": 0.6}
        assert calculate_ndcg(retrieved, scores, 3) == 1.0

    def test_reverse_ranking(self) -> None:
        """Test NDCG with reverse ranking."""
        retrieved = ["c", "b", "a"]
        scores = {"a": 1.0, "b": 0.5, "c": 0.0}
        # Not perfect, should be < 1.0
        ndcg = calculate_ndcg(retrieved, scores, 3)
        assert ndcg < 1.0
        assert ndcg > 0.0

    def test_empty_relevance(self) -> None:
        """Test NDCG with empty relevance scores."""
        retrieved = ["a", "b", "c"]
        scores: dict[str, float] = {}
        assert calculate_ndcg(retrieved, scores, 3) == 0.0

    def test_invalid_k(self) -> None:
        """Test NDCG with invalid k."""
        retrieved = ["a", "b"]
        scores = {"a": 1.0}
        assert calculate_ndcg(retrieved, scores, 0) == 0.0


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_all_metrics(self) -> None:
        """Test calculating all metrics at once."""
        retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        relevant = {"a", "c", "e", "g", "i"}

        metrics = calculate_metrics(retrieved, relevant, latency_ms=50.0)

        assert metrics.precision_at_1 == 1.0
        assert metrics.precision_at_5 == 0.6
        assert metrics.recall_at_5 == 0.6
        assert metrics.mrr == 1.0
        assert metrics.latency_ms == 50.0
        assert metrics.query_count == 1
        assert metrics.successful_queries == 1

    def test_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = BenchmarkMetrics(
            precision_at_1=0.8,
            mrr=0.75,
            query_count=10,
            successful_queries=9,
        )
        d = metrics.to_dict()
        assert d["precision@1"] == 0.8
        assert d["mrr"] == 0.75
        assert d["success_rate"] == 0.9


class TestAggregateMetrics:
    """Tests for metric aggregation."""

    def test_aggregate_two_queries(self) -> None:
        """Test aggregating metrics from two queries."""
        m1 = BenchmarkMetrics(
            precision_at_1=1.0,
            mrr=1.0,
            latency_ms=100.0,
            query_count=1,
            successful_queries=1,
        )
        m2 = BenchmarkMetrics(
            precision_at_1=0.0,
            mrr=0.5,
            latency_ms=200.0,
            query_count=1,
            successful_queries=1,
        )

        agg = aggregate_metrics([m1, m2])

        assert agg.precision_at_1 == 0.5
        assert agg.mrr == 0.75
        assert agg.latency_ms == 150.0
        assert agg.query_count == 2
        assert agg.successful_queries == 2

    def test_aggregate_empty(self) -> None:
        """Test aggregating empty list."""
        agg = aggregate_metrics([])
        assert agg.query_count == 0
        assert agg.precision_at_1 == 0.0
