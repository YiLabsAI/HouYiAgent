"""Benchmark runner for RAG system evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from houyi.rag import RAG

from .metrics import BenchmarkMetrics, aggregate_metrics, calculate_metrics


@dataclass
class BenchmarkQuery:
    """A single benchmark query with expected results."""

    query: str
    relevant_doc_ids: set[str]
    relevance_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """A benchmark dataset with queries and ground truth."""

    name: str
    queries: list[BenchmarkQuery]
    knowledge_dir: str
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkDataset:
        """Create dataset from dictionary."""
        queries = [
            BenchmarkQuery(
                query=q["query"],
                relevant_doc_ids=set(q.get("relevant_doc_ids", [])),
                relevance_scores=q.get("relevance_scores", {}),
                metadata=q.get("metadata", {}),
            )
            for q in data.get("queries", [])
        ]
        return cls(
            name=data.get("name", "unnamed"),
            queries=queries,
            knowledge_dir=data.get("knowledge_dir", ""),
            description=data.get("description", ""),
        )


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    dataset_name: str
    mode: str
    metrics: BenchmarkMetrics
    query_results: list[dict[str, Any]]
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        m = self.metrics
        return f"""
Benchmark Results: {self.dataset_name}
Mode: {self.mode}
Queries: {m.query_count} (Success: {m.successful_queries})

Precision:
  P@1:  {m.precision_at_1:.3f}
  P@5:  {m.precision_at_5:.3f}
  P@10: {m.precision_at_10:.3f}

Recall:
  R@1:  {m.recall_at_1:.3f}
  R@5:  {m.recall_at_5:.3f}
  R@10: {m.recall_at_10:.3f}

Ranking:
  MRR:     {m.mrr:.3f}
  NDCG@5:  {m.ndcg_at_5:.3f}
  NDCG@10: {m.ndcg_at_10:.3f}

Latency:
  Avg: {m.latency_ms:.1f}ms
"""


class BenchmarkRunner:
    """Runner for RAG benchmark evaluation."""

    def __init__(
        self,
        rag: RAG | None = None,
        mode: str = "agentic",
        knowledge_dir: str = "",
    ) -> None:
        """Initialize benchmark runner.

        Args:
            rag: Optional pre-configured RAG instance
            mode: RAG mode to use ("agentic" or "indexed")
            knowledge_dir: Knowledge directory path
        """
        self._rag = rag
        self._mode = mode
        self._knowledge_dir = knowledge_dir

    async def run(self, dataset: BenchmarkDataset) -> BenchmarkResult:
        """Run benchmark on a dataset.

        Args:
            dataset: Benchmark dataset with queries and ground truth

        Returns:
            BenchmarkResult with aggregated metrics
        """
        # Create RAG instance if not provided
        rag = self._rag
        if rag is None:
            kb_dir = dataset.knowledge_dir or self._knowledge_dir
            rag = RAG(mode=self._mode, knowledge_dir=kb_dir)

        metrics_list: list[BenchmarkMetrics] = []
        query_results: list[dict[str, Any]] = []

        for bq in dataset.queries:
            # Execute query and measure latency
            start_time = time.perf_counter()
            try:
                result = await rag.query(bq.query)
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Extract retrieved document IDs from sources
                retrieved_ids = [
                    s.doc_id for s in result.sources if s.doc_id
                ]

                # Calculate metrics
                metrics = calculate_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=bq.relevant_doc_ids,
                    relevance_scores=bq.relevance_scores or None,
                    latency_ms=latency_ms,
                )

                query_results.append({
                    "query": bq.query,
                    "retrieved": retrieved_ids,
                    "expected": list(bq.relevant_doc_ids),
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "latency_ms": latency_ms,
                    "metrics": metrics.to_dict(),
                })

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                metrics = BenchmarkMetrics(
                    latency_ms=latency_ms,
                    query_count=1,
                    successful_queries=0,
                )
                query_results.append({
                    "query": bq.query,
                    "error": str(e),
                    "latency_ms": latency_ms,
                })

            metrics_list.append(metrics)

        # Aggregate metrics
        aggregated = aggregate_metrics(metrics_list)

        return BenchmarkResult(
            dataset_name=dataset.name,
            mode=self._mode,
            metrics=aggregated,
            query_results=query_results,
            config={
                "mode": self._mode,
                "knowledge_dir": dataset.knowledge_dir or self._knowledge_dir,
            },
        )


def create_simple_dataset(
    name: str,
    knowledge_dir: str,
    queries_with_relevance: list[tuple[str, list[str]]],
) -> BenchmarkDataset:
    """Create a simple benchmark dataset.

    Args:
        name: Dataset name
        knowledge_dir: Path to knowledge directory
        queries_with_relevance: List of (query, relevant_doc_ids) tuples

    Returns:
        BenchmarkDataset instance
    """
    queries = [
        BenchmarkQuery(query=q, relevant_doc_ids=set(r))
        for q, r in queries_with_relevance
    ]
    return BenchmarkDataset(
        name=name,
        queries=queries,
        knowledge_dir=knowledge_dir,
    )
