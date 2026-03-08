"""Tests for PPR calculator."""

from __future__ import annotations

from houyi.rag.indexed.graph.ppr import PPRCalculator, PPRResult, compute_ppr


class TestPPRCalculator:
    def test_compute_empty_seeds(self) -> None:
        calc = PPRCalculator()
        result = calc.compute({}, [])
        assert result.scores == {}
        assert result.converged is True

    def test_compute_simple_graph(self) -> None:
        calc = PPRCalculator(alpha=0.85, max_iter=50)
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [],
        }

        result = calc.compute(adjacency, seeds=["A"])

        assert "A" in result.scores
        assert result.scores["A"] > result.scores.get("C", 0)
        assert result.iterations > 0

    def test_compute_cyclic_graph(self) -> None:
        calc = PPRCalculator(alpha=0.85, max_iter=100)
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [("A", 1.0)],
        }

        result = calc.compute(adjacency, seeds=["A"])

        assert len(result.scores) == 3
        assert result.converged is True

    def test_compute_weighted_edges(self) -> None:
        calc = PPRCalculator(alpha=0.85)
        adjacency = {
            "A": [("B", 0.9), ("C", 0.1)],
        }

        result = calc.compute(adjacency, seeds=["A"])

        assert result.scores.get("B", 0) > result.scores.get("C", 0)

    def test_top_k(self) -> None:
        calc = PPRCalculator()
        result = PPRResult(
            scores={"A": 0.5, "B": 0.3, "C": 0.2, "D": 0.1},
            iterations=10,
            converged=True,
        )

        top_2 = calc.top_k(result, k=2)
        assert len(top_2) == 2
        assert top_2[0][0] == "A"
        assert top_2[1][0] == "B"

    def test_top_k_exclude_seeds(self) -> None:
        calc = PPRCalculator()
        result = PPRResult(
            scores={"A": 0.5, "B": 0.3, "C": 0.2},
            iterations=10,
            converged=True,
        )

        top = calc.top_k(result, k=2, exclude_seeds=["A"])
        assert len(top) == 2
        assert top[0][0] == "B"

    def test_compute_ppr_convenience(self) -> None:
        adjacency = {"A": [("B", 1.0)]}
        scores = compute_ppr(adjacency, seeds=["A"])
        assert isinstance(scores, dict)
        assert "A" in scores
