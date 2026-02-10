"""Tests for RAG benchmark runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tests.rag.benchmark.runner import (
    BenchmarkDataset,
    BenchmarkQuery,
    BenchmarkResult,
    BenchmarkRunner,
    create_simple_dataset,
)


class TestBenchmarkQuery:
    """Tests for BenchmarkQuery."""

    def test_create_query(self) -> None:
        """Test creating a benchmark query."""
        query = BenchmarkQuery(
            query="What is RAG?",
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert query.query == "What is RAG?"
        assert "doc1" in query.relevant_doc_ids

    def test_query_with_scores(self) -> None:
        """Test query with relevance scores."""
        query = BenchmarkQuery(
            query="What is RAG?",
            relevant_doc_ids={"doc1", "doc2"},
            relevance_scores={"doc1": 1.0, "doc2": 0.5},
        )
        assert query.relevance_scores["doc1"] == 1.0


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset."""

    def test_create_dataset(self) -> None:
        """Test creating a benchmark dataset."""
        queries = [
            BenchmarkQuery(query="q1", relevant_doc_ids={"d1"}),
            BenchmarkQuery(query="q2", relevant_doc_ids={"d2"}),
        ]
        dataset = BenchmarkDataset(
            name="test",
            queries=queries,
            knowledge_dir="/tmp/kb",
        )
        assert dataset.name == "test"
        assert len(dataset.queries) == 2

    def test_from_dict(self) -> None:
        """Test creating dataset from dictionary."""
        data = {
            "name": "test_dataset",
            "knowledge_dir": "/tmp/kb",
            "queries": [
                {"query": "q1", "relevant_doc_ids": ["d1", "d2"]},
                {"query": "q2", "relevant_doc_ids": ["d3"]},
            ],
        }
        dataset = BenchmarkDataset.from_dict(data)

        assert dataset.name == "test_dataset"
        assert len(dataset.queries) == 2
        assert "d1" in dataset.queries[0].relevant_doc_ids

    def test_create_simple_dataset(self) -> None:
        """Test create_simple_dataset helper."""
        dataset = create_simple_dataset(
            name="simple",
            knowledge_dir="/tmp/kb",
            queries_with_relevance=[
                ("What is RAG?", ["doc1", "doc2"]),
                ("How does it work?", ["doc2", "doc3"]),
            ],
        )
        assert dataset.name == "simple"
        assert len(dataset.queries) == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_summary(self) -> None:
        """Test result summary generation."""
        from tests.rag.benchmark.metrics import BenchmarkMetrics

        result = BenchmarkResult(
            dataset_name="test",
            mode="agentic",
            metrics=BenchmarkMetrics(
                precision_at_1=0.8,
                precision_at_5=0.6,
                recall_at_5=0.7,
                mrr=0.85,
                ndcg_at_5=0.75,
                latency_ms=100.0,
                query_count=10,
                successful_queries=9,
            ),
            query_results=[],
        )

        summary = result.summary()

        assert "test" in summary
        assert "agentic" in summary
        assert "0.800" in summary  # P@1
        assert "100.0ms" in summary


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.mark.asyncio
    async def test_run_benchmark(self) -> None:
        """Test running a benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create test documents
            (kb_dir / "doc1.md").write_text("RAG is Retrieval-Augmented Generation.")
            (kb_dir / "doc2.md").write_text("It combines search with LLMs.")
            (kb_dir / "doc3.md").write_text("Python is a programming language.")

            dataset = create_simple_dataset(
                name="test",
                knowledge_dir=str(kb_dir),
                queries_with_relevance=[
                    ("What is RAG?", ["doc1.md"]),
                ],
            )

            runner = BenchmarkRunner(mode="agentic")
            result = await runner.run(dataset)

            assert result.dataset_name == "test"
            assert result.mode == "agentic"
            assert result.metrics.query_count == 1

    @pytest.mark.asyncio
    async def test_run_benchmark_multiple_queries(self) -> None:
        """Test running benchmark with multiple queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            (kb_dir / "doc1.md").write_text("Topic A content")
            (kb_dir / "doc2.md").write_text("Topic B content")

            dataset = BenchmarkDataset(
                name="multi",
                knowledge_dir=str(kb_dir),
                queries=[
                    BenchmarkQuery(query="Topic A", relevant_doc_ids={"doc1.md"}),
                    BenchmarkQuery(query="Topic B", relevant_doc_ids={"doc2.md"}),
                ],
            )

            runner = BenchmarkRunner(mode="agentic")
            result = await runner.run(dataset)

            assert result.metrics.query_count == 2
            assert len(result.query_results) == 2
