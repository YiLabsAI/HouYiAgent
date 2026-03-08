"""Tests for community detection."""

from __future__ import annotations

from houyi.rag.indexed.graph.community import (
    Community,
    CommunityDetectionResult,
    CommunityDetector,
    detect_communities,
)


class TestCommunityDetector:
    def test_detect_empty_graph(self) -> None:
        detector = CommunityDetector()
        result = detector.detect({})
        assert result.communities == []
        assert result.modularity == 0.0

    def test_detect_connected_components_fallback(self) -> None:
        detector = CommunityDetector()
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0)],
            "C": [("D", 1.0)],
            "D": [("C", 1.0)],
        }

        result = detector._detect_connected_components(adjacency)

        assert len(result.communities) == 2
        assert result.num_communities == 2
        assert result.partition["A"] == result.partition["B"]
        assert result.partition["C"] == result.partition["D"]
        assert result.partition["A"] != result.partition["C"]

    def test_detect_single_component(self) -> None:
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
        community = Community(id=0, members=["A", "B", "C"])
        assert community.size == 3
        assert community.id == 0

    def test_get_community_subgraph(self) -> None:
        detector = CommunityDetector()
        community = Community(id=0, members=["A", "B"])
        adjacency = {
            "A": [("B", 1.0), ("C", 1.0)],
            "B": [("A", 1.0)],
            "C": [("A", 1.0)],
        }

        subgraph = detector.get_community_subgraph(community, adjacency)

        assert "A" in subgraph
        assert ("B", 1.0) in subgraph["A"]
        assert ("C", 1.0) not in subgraph["A"]

    def test_detect_communities_convenience(self) -> None:
        adjacency = {"A": [("B", 1.0)], "B": [("A", 1.0)]}
        result = detect_communities(adjacency)
        assert isinstance(result, CommunityDetectionResult)
