"""Community detection using Louvain algorithm.

Provides community detection for knowledge graphs to enable
hierarchical summarization and topic clustering.

Reference:
- Louvain method: Blondel et al., 2008
- python-louvain: https://github.com/taynaud/python-louvain
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A detected community."""

    id: int
    members: list[str]  # Entity IDs
    size: int = 0
    modularity_contribution: float = 0.0
    label: str = ""  # Optional descriptive label

    def __post_init__(self) -> None:
        self.size = len(self.members)


@dataclass
class CommunityDetectionResult:
    """Result of community detection."""

    communities: list[Community]
    partition: dict[str, int]  # entity_id -> community_id
    modularity: float
    num_communities: int = 0
    hierarchy: list[dict[str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.num_communities = len(self.communities)


class CommunityDetector:
    """Community detection using Louvain algorithm.

    Falls back to simple connected components if python-louvain is not installed.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        """Initialize community detector.

        Args:
            resolution: Resolution parameter for Louvain (higher = more communities)
            random_state: Random seed for reproducibility
        """
        self.resolution = resolution
        self.random_state = random_state
        self._has_louvain = self._check_louvain()

    def _check_louvain(self) -> bool:
        """Check if python-louvain is available."""
        available = importlib.util.find_spec("community") is not None
        if not available:
            logger.debug("python-louvain not installed, using fallback")
        return available

    def detect(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
    ) -> CommunityDetectionResult:
        """Detect communities in the graph.

        Args:
            adjacency: Adjacency list {src: [(dst, weight), ...]}

        Returns:
            CommunityDetectionResult with detected communities
        """
        if not adjacency:
            return CommunityDetectionResult(
                communities=[],
                partition={},
                modularity=0.0,
            )

        if self._has_louvain:
            return self._detect_louvain(adjacency)
        return self._detect_connected_components(adjacency)

    def _detect_louvain(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
    ) -> CommunityDetectionResult:
        """Detect communities using Louvain algorithm.

        Args:
            adjacency: Adjacency list

        Returns:
            CommunityDetectionResult
        """
        try:
            import community as community_louvain
            import networkx as nx
        except ImportError:
            return self._detect_connected_components(adjacency)

        # Build NetworkX graph
        G = nx.Graph()
        for src, neighbors in adjacency.items():
            G.add_node(src)
            for dst, weight in neighbors:
                G.add_edge(src, dst, weight=weight)

        if G.number_of_nodes() == 0:
            return CommunityDetectionResult(
                communities=[],
                partition={},
                modularity=0.0,
            )

        # Run Louvain
        partition = community_louvain.best_partition(
            G,
            resolution=self.resolution,
            random_state=self.random_state,
        )

        # Calculate modularity
        modularity = community_louvain.modularity(partition, G)

        # Build communities
        community_members: dict[int, list[str]] = {}
        for entity_id, comm_id in partition.items():
            if comm_id not in community_members:
                community_members[comm_id] = []
            community_members[comm_id].append(entity_id)

        communities = [
            Community(id=comm_id, members=members)
            for comm_id, members in sorted(community_members.items())
        ]

        return CommunityDetectionResult(
            communities=communities,
            partition=partition,
            modularity=modularity,
        )

    def _detect_connected_components(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
    ) -> CommunityDetectionResult:
        """Fallback: detect connected components as communities.

        Args:
            adjacency: Adjacency list

        Returns:
            CommunityDetectionResult
        """
        # Build undirected adjacency
        undirected: dict[str, set[str]] = {}
        all_nodes: set[str] = set()

        for src, neighbors in adjacency.items():
            all_nodes.add(src)
            if src not in undirected:
                undirected[src] = set()
            for dst, _ in neighbors:
                all_nodes.add(dst)
                undirected[src].add(dst)
                if dst not in undirected:
                    undirected[dst] = set()
                undirected[dst].add(src)

        # Find connected components using BFS
        visited: set[str] = set()
        communities: list[Community] = []
        partition: dict[str, int] = {}

        for node in all_nodes:
            if node in visited:
                continue

            # BFS to find component
            component: list[str] = []
            queue = [node]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)

                for neighbor in undirected.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Create community
            comm_id = len(communities)
            communities.append(Community(id=comm_id, members=component))
            for member in component:
                partition[member] = comm_id

        return CommunityDetectionResult(
            communities=communities,
            partition=partition,
            modularity=0.0,  # Not computed for fallback
        )

    def get_community_subgraph(
        self,
        community: Community,
        adjacency: dict[str, list[tuple[str, float]]],
    ) -> dict[str, list[tuple[str, float]]]:
        """Extract subgraph for a community.

        Args:
            community: Target community
            adjacency: Full adjacency list

        Returns:
            Adjacency list for community subgraph
        """
        members = set(community.members)
        subgraph: dict[str, list[tuple[str, float]]] = {}

        for src in members:
            if src in adjacency:
                filtered_neighbors = [
                    (dst, weight) for dst, weight in adjacency[src] if dst in members
                ]
                if filtered_neighbors:
                    subgraph[src] = filtered_neighbors

        return subgraph


def detect_communities(
    adjacency: dict[str, list[tuple[str, float]]],
    resolution: float = 1.0,
) -> CommunityDetectionResult:
    """Convenience function to detect communities.

    Args:
        adjacency: Adjacency list {src: [(dst, weight), ...]}
        resolution: Resolution parameter for Louvain

    Returns:
        CommunityDetectionResult
    """
    detector = CommunityDetector(resolution=resolution)
    return detector.detect(adjacency)
