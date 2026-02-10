"""Personalized PageRank (PPR) algorithm.

Standalone implementation for graph-based retrieval ranking.

Reference:
- HippoRAG (NeurIPS 2024): https://github.com/OSU-NLP-Group/HippoRAG
- Original PageRank: Brin & Page, 1998
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PPRResult:
    """Result of PPR computation."""

    scores: dict[str, float]  # entity_id -> score
    iterations: int
    converged: bool


class PPRCalculator:
    """Personalized PageRank calculator.

    Computes relevance scores for graph nodes based on seed nodes.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize PPR calculator.

        Args:
            alpha: Teleport probability (damping factor), default 0.85
            max_iter: Maximum iterations for power iteration
            epsilon: Convergence threshold
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon

    def compute(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        seeds: list[str],
        entity_ids: list[str] | None = None,
    ) -> PPRResult:
        """Compute PPR scores.

        Args:
            adjacency: Adjacency list {src: [(dst, weight), ...]}
            seeds: Seed entity IDs for personalization
            entity_ids: Optional ordered list of all entity IDs

        Returns:
            PPRResult with scores for all nodes
        """
        if not seeds:
            return PPRResult(scores={}, iterations=0, converged=True)

        # Build entity index
        if entity_ids is None:
            entity_ids = list(
                set(adjacency.keys())
                | {dst for neighbors in adjacency.values() for dst, _ in neighbors}
            )

        if not entity_ids:
            return PPRResult(scores={}, iterations=0, converged=True)

        n = len(entity_ids)
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        # Personalization vector (uniform over seeds)
        p = np.zeros(n)
        valid_seeds = [s for s in seeds if s in id_to_idx]
        if not valid_seeds:
            return PPRResult(scores={}, iterations=0, converged=True)

        for seed in valid_seeds:
            p[id_to_idx[seed]] = 1.0 / len(valid_seeds)

        # Build transition matrix (column-stochastic)
        # For efficiency, we use sparse representation
        out_degree = {eid: sum(w for _, w in neighbors) for eid, neighbors in adjacency.items()}

        # Initialize scores
        scores = p.copy()

        # Power iteration
        converged = False
        iterations = 0

        for iteration in range(self.max_iter):
            new_scores = np.zeros(n)

            # Random walk contribution
            for src_id, neighbors in adjacency.items():
                if src_id not in id_to_idx:
                    continue
                src_idx = id_to_idx[src_id]
                src_score = scores[src_idx]
                total_weight = out_degree.get(src_id, 0)

                if total_weight > 0:
                    for dst_id, weight in neighbors:
                        if dst_id in id_to_idx:
                            dst_idx = id_to_idx[dst_id]
                            new_scores[dst_idx] += (
                                (1 - self.alpha) * src_score * weight / total_weight
                            )

            # Teleport contribution
            new_scores += self.alpha * p

            # Handle dangling nodes (no outgoing edges)
            dangling_sum = 0.0
            for i, eid in enumerate(entity_ids):
                if eid not in adjacency or not adjacency[eid]:
                    dangling_sum += scores[i]
            new_scores += (1 - self.alpha) * dangling_sum * p

            # Check convergence
            diff = np.sum(np.abs(new_scores - scores))
            scores = new_scores
            iterations = iteration + 1

            if diff < self.epsilon:
                converged = True
                break

        # Convert to dict, filtering zero scores
        result_scores = {entity_ids[i]: float(scores[i]) for i in range(n) if scores[i] > 1e-10}

        return PPRResult(
            scores=result_scores,
            iterations=iterations,
            converged=converged,
        )

    def compute_batch(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        seed_batches: list[list[str]],
        entity_ids: list[str] | None = None,
    ) -> list[PPRResult]:
        """Compute PPR for multiple seed sets.

        Args:
            adjacency: Adjacency list
            seed_batches: List of seed sets
            entity_ids: Optional ordered entity IDs

        Returns:
            List of PPRResults
        """
        return [self.compute(adjacency, seeds, entity_ids) for seeds in seed_batches]

    def top_k(
        self,
        result: PPRResult,
        k: int = 10,
        exclude_seeds: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Get top-k entities by PPR score.

        Args:
            result: PPR computation result
            k: Number of top entities to return
            exclude_seeds: Optional seeds to exclude from results

        Returns:
            List of (entity_id, score) tuples
        """
        scores = result.scores
        if exclude_seeds:
            scores = {eid: s for eid, s in scores.items() if eid not in exclude_seeds}

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]


def compute_ppr(
    adjacency: dict[str, list[tuple[str, float]]],
    seeds: list[str],
    alpha: float = 0.85,
    max_iter: int = 100,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Convenience function to compute PPR scores.

    Args:
        adjacency: Adjacency list {src: [(dst, weight), ...]}
        seeds: Seed entity IDs
        alpha: Teleport probability
        max_iter: Maximum iterations
        epsilon: Convergence threshold

    Returns:
        Dict of entity_id -> PPR score
    """
    calculator = PPRCalculator(alpha=alpha, max_iter=max_iter, epsilon=epsilon)
    result = calculator.compute(adjacency, seeds)
    return result.scores
