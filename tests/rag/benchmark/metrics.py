"""Retrieval metrics for RAG benchmark evaluation.

Implements standard IR evaluation metrics:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BenchmarkMetrics:
    """Container for benchmark evaluation metrics."""

    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    latency_ms: float = 0.0
    query_count: int = 0
    successful_queries: int = 0

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "precision@1": self.precision_at_1,
            "precision@3": self.precision_at_3,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "latency_ms": self.latency_ms,
            "query_count": self.query_count,
            "success_rate": (
                self.successful_queries / self.query_count
                if self.query_count > 0
                else 0.0
            ),
        }


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K.

    Precision@K = |Retrieved@K ∩ Relevant| / K

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K.

    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / rank_of_first_relevant_document

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: Set of relevant document IDs

    Returns:
        MRR score (0.0 to 1.0)
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    DCG@K = Σ (rel_i / log2(i + 1)) for i = 1 to K
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevance_scores: Dict mapping doc_id to relevance score (0-1)
        k: Number of top results to consider

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    def dcg(scores: list[float], n: int) -> float:
        """Calculate DCG."""
        return sum(
            score / math.log2(i + 2)  # i + 2 because log2(1) = 0
            for i, score in enumerate(scores[:n])
        )

    # Get relevance scores for retrieved documents
    retrieved_scores = [
        relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]
    ]

    # Calculate DCG
    dcg_score = dcg(retrieved_scores, k)

    # Calculate ideal DCG (perfect ranking)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg_score = dcg(ideal_scores, k)

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score


def calculate_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    relevance_scores: dict[str, float] | None = None,
    latency_ms: float = 0.0,
) -> BenchmarkMetrics:
    """Calculate all benchmark metrics for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: Set of relevant document IDs
        relevance_scores: Optional dict mapping doc_id to relevance score
        latency_ms: Query latency in milliseconds

    Returns:
        BenchmarkMetrics with all metrics calculated
    """
    # Use binary relevance if no scores provided
    if relevance_scores is None:
        relevance_scores = dict.fromkeys(relevant_ids, 1.0)

    return BenchmarkMetrics(
        precision_at_1=calculate_precision_at_k(retrieved_ids, relevant_ids, 1),
        precision_at_3=calculate_precision_at_k(retrieved_ids, relevant_ids, 3),
        precision_at_5=calculate_precision_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_10=calculate_precision_at_k(retrieved_ids, relevant_ids, 10),
        recall_at_1=calculate_recall_at_k(retrieved_ids, relevant_ids, 1),
        recall_at_3=calculate_recall_at_k(retrieved_ids, relevant_ids, 3),
        recall_at_5=calculate_recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=calculate_recall_at_k(retrieved_ids, relevant_ids, 10),
        mrr=calculate_mrr(retrieved_ids, relevant_ids),
        ndcg_at_5=calculate_ndcg(retrieved_ids, relevance_scores, 5),
        ndcg_at_10=calculate_ndcg(retrieved_ids, relevance_scores, 10),
        latency_ms=latency_ms,
        query_count=1,
        successful_queries=1 if retrieved_ids else 0,
    )


def aggregate_metrics(metrics_list: list[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Aggregate metrics from multiple queries.

    Args:
        metrics_list: List of BenchmarkMetrics from individual queries

    Returns:
        Aggregated BenchmarkMetrics with averaged values
    """
    if not metrics_list:
        return BenchmarkMetrics()

    n = len(metrics_list)
    return BenchmarkMetrics(
        precision_at_1=sum(m.precision_at_1 for m in metrics_list) / n,
        precision_at_3=sum(m.precision_at_3 for m in metrics_list) / n,
        precision_at_5=sum(m.precision_at_5 for m in metrics_list) / n,
        precision_at_10=sum(m.precision_at_10 for m in metrics_list) / n,
        recall_at_1=sum(m.recall_at_1 for m in metrics_list) / n,
        recall_at_3=sum(m.recall_at_3 for m in metrics_list) / n,
        recall_at_5=sum(m.recall_at_5 for m in metrics_list) / n,
        recall_at_10=sum(m.recall_at_10 for m in metrics_list) / n,
        mrr=sum(m.mrr for m in metrics_list) / n,
        ndcg_at_5=sum(m.ndcg_at_5 for m in metrics_list) / n,
        ndcg_at_10=sum(m.ndcg_at_10 for m in metrics_list) / n,
        latency_ms=sum(m.latency_ms for m in metrics_list) / n,
        query_count=sum(m.query_count for m in metrics_list),
        successful_queries=sum(m.successful_queries for m in metrics_list),
    )
