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
        empty = PPRResult(scores={}, iterations=0, converged=True)
        if not seeds:
            return empty

        if entity_ids is None:
            entity_ids = list(
                set(adjacency.keys())
                | {dst for neighbors in adjacency.values() for dst, _ in neighbors}
            )
        if not entity_ids:
            return empty

        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        p = self._personalization_vector(seeds, id_to_idx, len(entity_ids))
        if p is None:
            return empty

        out_degree = {eid: sum(w for _, w in neighbors) for eid, neighbors in adjacency.items()}

        scores, iterations, converged = self._power_iterate(
            adjacency,
            entity_ids,
            id_to_idx,
            p,
            out_degree,
        )

        result_scores = {
            entity_ids[i]: float(scores[i]) for i in range(len(entity_ids)) if scores[i] > 1e-10
        }
        return PPRResult(scores=result_scores, iterations=iterations, converged=converged)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _personalization_vector(
        seeds: list[str],
        id_to_idx: dict[str, int],
        n: int,
    ) -> np.ndarray | None:
        """Build uniform personalization vector over valid seeds."""
        valid = [s for s in seeds if s in id_to_idx]
        if not valid:
            return None
        p = np.zeros(n)
        for seed in valid:
            p[id_to_idx[seed]] = 1.0 / len(valid)
        return p

    def _power_iterate(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        entity_ids: list[str],
        id_to_idx: dict[str, int],
        p: np.ndarray,
        out_degree: dict[str, float],
    ) -> tuple[np.ndarray, int, bool]:
        """Run power iteration until convergence or max_iter."""
        n = len(entity_ids)
        scores = p.copy()
        converged = False
        iterations = 0

        for iteration in range(self.max_iter):
            new_scores = self._walk_step(adjacency, id_to_idx, scores, out_degree, n)
            new_scores += self.alpha * p
            dangling_sum = sum(
                scores[i]
                for i, eid in enumerate(entity_ids)
                if eid not in adjacency or not adjacency[eid]
            )
            new_scores += (1 - self.alpha) * dangling_sum * p

            diff = float(np.sum(np.abs(new_scores - scores)))
            scores = new_scores
            iterations = iteration + 1
            if diff < self.epsilon:
                converged = True
                break

        return scores, iterations, converged

    def _walk_step(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        id_to_idx: dict[str, int],
        scores: np.ndarray,
        out_degree: dict[str, float],
        n: int,
    ) -> np.ndarray:
        """Single random-walk contribution step."""
        new_scores = np.zeros(n)
        for src_id, neighbors in adjacency.items():
            if src_id not in id_to_idx:
                continue
            src_idx = id_to_idx[src_id]
            total_weight = out_degree.get(src_id, 0)
            if total_weight <= 0:
                continue
            src_score = scores[src_idx]
            for dst_id, weight in neighbors:
                if dst_id in id_to_idx:
                    new_scores[id_to_idx[dst_id]] += (
                        (1 - self.alpha) * src_score * weight / total_weight
                    )
        return new_scores

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
