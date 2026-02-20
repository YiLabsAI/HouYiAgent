"""RAG Benchmark module for retrieval effectiveness evaluation."""

from tests.rag.benchmark.metrics import (
    BenchmarkMetrics,
    aggregate_metrics,
    calculate_metrics,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from tests.rag.benchmark.runner import (
    BenchmarkDataset,
    BenchmarkQuery,
    BenchmarkResult,
    BenchmarkRunner,
    create_simple_dataset,
)

__all__ = [
    # Runner
    "BenchmarkDataset",
    # Metrics
    "BenchmarkMetrics",
    "BenchmarkQuery",
    "BenchmarkResult",
    "BenchmarkRunner",
    "aggregate_metrics",
    "calculate_metrics",
    "calculate_mrr",
    "calculate_ndcg",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "create_simple_dataset",
]
