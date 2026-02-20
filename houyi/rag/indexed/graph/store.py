"""Graph store using adjacency list with SQLite persistence.

SQLite does not natively support graphs, but can simulate graph operations
using adjacency tables and recursive CTEs for traversal.

Reference:
- SQLite Recursive CTE: https://sqlite.org/lang_with.html
- LightRAG storage design: https://github.com/HKUDS/LightRAG
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from houyi.rag.config import GraphConfig
from houyi.rag.types import Entity, Relation, SearchResult, Source


class GraphStore:
    """Lightweight graph store using adjacency list + SQLite.

    Supports:
    - Entity and relation storage
    - BFS/DFS traversal via recursive CTE
    - PPR (Personalized PageRank) for ranking
    - Louvain community detection (optional)
    """

    def __init__(
        self,
        knowledge_dir: str | None = None,
        config: GraphConfig | None = None,
    ) -> None:
        """Initialize graph store.

        Args:
            knowledge_dir: Knowledge base directory
            config: Graph configuration
        """
        if knowledge_dir is None:
            from houyi.config import env

            knowledge_dir = env.rag_knowledge_dir
        self.knowledge_dir = Path(knowledge_dir)
        self.config = config or GraphConfig()

        self._db_path = self.knowledge_dir / ".houyi" / "graph.db"
        self._conn: sqlite3.Connection | None = None

        # In-memory adjacency list for fast access
        self._entities: dict[str, Entity] = {}
        self._adjacency: dict[
            str, list[tuple[str, str, float]]
        ] = {}  # src -> [(dst, rel_type, weight)]

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        self.close()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def __aenter__(self) -> GraphStore:
        """Async context manager entry."""
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.close()

    async def load(self) -> None:
        """Load graph from SQLite database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        # Create tables if not exist
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'unknown',
                embedding TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS relations (
                rel_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                rel_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES entities(entity_id),
                FOREIGN KEY (target_id) REFERENCES entities(entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
        """)

        # Load into memory
        await self._load_to_memory()

    async def _load_to_memory(self) -> None:
        """Load graph data into memory for fast access."""
        if self._conn is None:
            return

        # Load entities
        cursor = self._conn.execute("SELECT * FROM entities")
        for row in cursor:
            embedding = json.loads(row["embedding"]) if row["embedding"] else None
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            self._entities[row["entity_id"]] = Entity(
                entity_id=row["entity_id"],
                name=row["name"],
                entity_type=row["entity_type"],
                embedding=embedding,
                metadata=metadata,
            )

        # Load adjacency list
        cursor = self._conn.execute("SELECT * FROM relations")
        for row in cursor:
            src = row["source_id"]
            if src not in self._adjacency:
                self._adjacency[src] = []
            self._adjacency[src].append(
                (
                    row["target_id"],
                    row["rel_type"],
                    row["weight"],
                )
            )

    async def save(self) -> None:
        """Save graph to SQLite database."""
        if self._conn:
            self._conn.commit()

    async def add_entities(self, entities: list[Entity]) -> None:
        """Add entities to graph."""
        if self._conn is None:
            await self.load()

        for entity in entities:
            embedding_json = json.dumps(entity.embedding) if entity.embedding else None
            metadata_json = json.dumps(entity.metadata) if entity.metadata else None

            self._conn.execute(
                """INSERT OR REPLACE INTO entities
                   (entity_id, name, entity_type, embedding, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (entity.entity_id, entity.name, entity.entity_type, embedding_json, metadata_json),
            )
            self._entities[entity.entity_id] = entity

        self._conn.commit()

    async def add_relations(self, relations: list[Relation]) -> None:
        """Add relations to graph."""
        if self._conn is None:
            await self.load()

        for rel in relations:
            metadata_json = json.dumps(rel.metadata) if rel.metadata else None

            self._conn.execute(
                """INSERT OR REPLACE INTO relations
                   (rel_id, source_id, target_id, rel_type, weight, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (rel.rel_id, rel.source_id, rel.target_id, rel.rel_type, rel.weight, metadata_json),
            )

            # Update adjacency list
            if rel.source_id not in self._adjacency:
                self._adjacency[rel.source_id] = []
            self._adjacency[rel.source_id].append(
                (
                    rel.target_id,
                    rel.rel_type,
                    rel.weight,
                )
            )

        self._conn.commit()

    async def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[SearchResult]:
        """Search graph using PPR from query entities.

        Args:
            query: Query string (used to find seed entities)
            k: Number of results

        Returns:
            List of search results
        """
        # Find seed entities matching query
        seed_entities = self._find_seed_entities(query)
        if not seed_entities:
            return []

        # Run PPR from seeds
        ppr_scores = await self._ppr(
            seeds=seed_entities,
            alpha=self.config.ppr_alpha,
            max_iter=100,
        )

        # Sort by score and return top-k
        sorted_entities = sorted(
            ppr_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = []
        for entity_id, score in sorted_entities:
            entity = self._entities.get(entity_id)
            if entity:
                results.append(
                    SearchResult(
                        chunk_id=entity.entity_id,
                        content=entity.name,
                        score=score,
                        source=Source(
                            file_path="graph",
                            location=entity.entity_type,
                            snippet=entity.name,
                            score=score,
                        ),
                        metadata={"entity_type": entity.entity_type},
                    )
                )

        return results

    def _find_seed_entities(self, query: str) -> list[str]:
        """Find entities matching query text."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        matches = []
        for entity_id, entity in self._entities.items():
            name_lower = entity.name.lower()
            # Simple matching: entity name contains query word
            if any(word in name_lower for word in query_words) or query_lower in name_lower:
                matches.append(entity_id)

        return matches[:10]  # Limit seeds

    async def _ppr(
        self,
        seeds: list[str],
        alpha: float = 0.85,
        max_iter: int = 100,
        epsilon: float = 1e-6,
    ) -> dict[str, float]:
        """Personalized PageRank algorithm.

        Algorithm adapted from HippoRAG (NeurIPS 2024).
        Reference: https://github.com/OSU-NLP-Group/HippoRAG

        Args:
            seeds: Seed entity IDs
            alpha: Teleport probability (damping factor)
            max_iter: Maximum iterations
            epsilon: Convergence threshold

        Returns:
            Dict of entity_id -> PPR score
        """
        if not seeds or not self._entities:
            return {}

        # Initialize scores
        n = len(self._entities)
        entity_ids = list(self._entities.keys())
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        # Personalization vector (uniform over seeds)
        p = [0.0] * n
        for seed in seeds:
            if seed in id_to_idx:
                p[id_to_idx[seed]] = 1.0 / len(seeds)

        # Initialize scores
        scores = list(p)

        # Power iteration
        for _ in range(max_iter):
            new_scores = [0.0] * n

            # Random walk contribution
            for src_id, neighbors in self._adjacency.items():
                if src_id not in id_to_idx:
                    continue
                src_idx = id_to_idx[src_id]
                src_score = scores[src_idx]

                if neighbors:
                    total_weight = sum(w for _, _, w in neighbors)
                    for dst_id, _, weight in neighbors:
                        if dst_id in id_to_idx:
                            dst_idx = id_to_idx[dst_id]
                            new_scores[dst_idx] += (1 - alpha) * src_score * weight / total_weight

            # Teleport contribution
            for i in range(n):
                new_scores[i] += alpha * p[i]

            # Check convergence
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores

            if diff < epsilon:
                break

        # Convert to dict
        return {entity_ids[i]: scores[i] for i in range(n) if scores[i] > 0}

    async def bfs_traverse(
        self,
        start_id: str,
        max_depth: int = 3,
    ) -> list[tuple[str, int]]:
        """BFS traversal from a starting entity.

        Uses SQLite recursive CTE for traversal.

        Args:
            start_id: Starting entity ID
            max_depth: Maximum traversal depth

        Returns:
            List of (entity_id, depth) tuples
        """
        if self._conn is None:
            return []

        # Use recursive CTE for BFS
        query = """
            WITH RECURSIVE traverse(entity_id, depth) AS (
                SELECT ?, 0
                UNION ALL
                SELECT r.target_id, t.depth + 1
                FROM traverse t
                JOIN relations r ON r.source_id = t.entity_id
                WHERE t.depth < ?
            )
            SELECT DISTINCT entity_id, MIN(depth) as depth
            FROM traverse
            GROUP BY entity_id
            ORDER BY depth
        """

        cursor = self._conn.execute(query, (start_id, max_depth))
        return [(row["entity_id"], row["depth"]) for row in cursor]

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def get_neighbors(self, entity_id: str) -> list[tuple[str, str, float]]:
        """Get neighbors of an entity."""
        return self._adjacency.get(entity_id, [])

    def count_entities(self) -> int:
        """Get number of entities."""
        return len(self._entities)

    def count_relations(self) -> int:
        """Get number of relations."""
        return sum(len(neighbors) for neighbors in self._adjacency.values())
