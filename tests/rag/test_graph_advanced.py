"""Tests for PPR and community detection."""


from houyi.rag.indexed.graph.community import (
    Community,
    CommunityDetectionResult,
    CommunityDetector,
    detect_communities,
)
from houyi.rag.indexed.graph.ppr import PPRCalculator, PPRResult, compute_ppr


class TestPPRCalculator:
    """Tests for PPRCalculator."""

    def test_compute_empty_seeds(self) -> None:
        """Test with empty seeds."""
        calc = PPRCalculator()
        result = calc.compute({}, [])
        assert result.scores == {}
        assert result.converged is True

    def test_compute_simple_graph(self) -> None:
        """Test with simple graph."""
        calc = PPRCalculator(alpha=0.85, max_iter=50)

        # Simple chain: A -> B -> C
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [],
        }

        result = calc.compute(adjacency, seeds=["A"])

        assert "A" in result.scores
        assert result.scores["A"] > result.scores.get("C", 0)  # A should have higher score
        assert result.iterations > 0

    def test_compute_cyclic_graph(self) -> None:
        """Test with cyclic graph."""
        calc = PPRCalculator(alpha=0.85, max_iter=100)

        # Cycle: A -> B -> C -> A
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [("A", 1.0)],
        }

        result = calc.compute(adjacency, seeds=["A"])

        # All nodes should have scores due to cycle
        assert len(result.scores) == 3
        assert result.converged is True

    def test_compute_weighted_edges(self) -> None:
        """Test with weighted edges."""
        calc = PPRCalculator(alpha=0.85)

        adjacency = {
            "A": [("B", 0.9), ("C", 0.1)],
        }

        result = calc.compute(adjacency, seeds=["A"])

        # B should have higher score due to higher weight
        assert result.scores.get("B", 0) > result.scores.get("C", 0)

    def test_top_k(self) -> None:
        """Test top_k extraction."""
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
        """Test top_k with seed exclusion."""
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
        """Test convenience function."""
        adjacency = {"A": [("B", 1.0)]}
        scores = compute_ppr(adjacency, seeds=["A"])
        assert isinstance(scores, dict)
        assert "A" in scores


class TestCommunityDetector:
    """Tests for CommunityDetector."""

    def test_detect_empty_graph(self) -> None:
        """Test with empty graph."""
        detector = CommunityDetector()
        result = detector.detect({})
        assert result.communities == []
        assert result.modularity == 0.0

    def test_detect_connected_components_fallback(self) -> None:
        """Test connected components fallback."""
        detector = CommunityDetector()

        # Two disconnected components
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0)],
            "C": [("D", 1.0)],
            "D": [("C", 1.0)],
        }

        result = detector._detect_connected_components(adjacency)

        assert len(result.communities) == 2
        assert result.num_communities == 2

        # Check partition
        assert result.partition["A"] == result.partition["B"]
        assert result.partition["C"] == result.partition["D"]
        assert result.partition["A"] != result.partition["C"]

    def test_detect_single_component(self) -> None:
        """Test single connected component."""
        detector = CommunityDetector()

        adjacency = {
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [("A", 1.0)],
        }

        result = detector._detect_connected_components(adjacency)

        assert len(result.communities) == 1
        assert len(result.communities[0].members) == 3

    def test_community_dataclass(self) -> None:
        """Test Community dataclass."""
        community = Community(id=0, members=["A", "B", "C"])
        assert community.size == 3
        assert community.id == 0

    def test_get_community_subgraph(self) -> None:
        """Test extracting community subgraph."""
        detector = CommunityDetector()

        community = Community(id=0, members=["A", "B"])
        adjacency = {
            "A": [("B", 1.0), ("C", 1.0)],
            "B": [("A", 1.0)],
            "C": [("A", 1.0)],
        }

        subgraph = detector.get_community_subgraph(community, adjacency)

        # Should only include edges within community
        assert "A" in subgraph
        assert ("B", 1.0) in subgraph["A"]
        assert ("C", 1.0) not in subgraph["A"]

    def test_detect_communities_convenience(self) -> None:
        """Test convenience function."""
        adjacency = {"A": [("B", 1.0)], "B": [("A", 1.0)]}
        result = detect_communities(adjacency)
        assert isinstance(result, CommunityDetectionResult)
