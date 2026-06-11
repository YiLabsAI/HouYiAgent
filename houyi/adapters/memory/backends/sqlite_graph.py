"""SQLite graph store — edge CRUD, community labels, and recursive CTE traversal.

Extracted from SQLiteMemoryBackend to enforce SRP: the memory backend
handles fact/record CRUD; this module handles the graph index (edges,
communities, traversal).  The backend delegates graph calls here,
mirroring the pattern used by SQLiteFTSSearch, SQLiteVectorSearch, etc.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from houyi.adapters.memory.backends.sqlite_connection import SQLiteConnectionManager
from houyi.adapters.memory.types import (
    GraphTraversalResult,
    MemoryEdge,
    MemoryRelation,
)


class SQLiteGraphStore:
    """Handles all graph-index SQLite operations: edges, community labels,
    and recursive BFS traversal."""

    def __init__(self, conn_manager: SQLiteConnectionManager) -> None:
        self._conn_manager = conn_manager

    def _conn(self) -> sqlite3.Connection:
        return self._conn_manager.get_connection()

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def add_edge(self, edge: MemoryEdge, *, conn: sqlite3.Connection | None = None) -> None:
        """Insert or update an edge row."""
        c = conn or self._conn()
        c.execute(
            """
            INSERT INTO memory_edges
            (edge_id, namespace, source_unit_id, target_unit_id,
             source_type, target_type, relation, weight,
             valid_from, valid_to, created_at, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
            weight=excluded.weight,
            valid_from=excluded.valid_from,
            valid_to=excluded.valid_to,
            provenance=excluded.provenance
            """,
            (
                edge.edge_id,
                edge.namespace,
                edge.source_unit_id,
                edge.target_unit_id,
                edge.source_type,
                edge.target_type,
                edge.relation.value,
                edge.weight,
                edge.valid_from,
                edge.valid_to,
                edge.created_at,
                edge.provenance,
            ),
        )

    def delete_edge(self, edge_id: str, *, conn: sqlite3.Connection | None = None) -> bool:
        """Delete an edge by ID."""
        c = conn or self._conn()
        cur = c.execute("DELETE FROM memory_edges WHERE edge_id=?", (edge_id,))
        return cur.rowcount > 0

    def invalidate_edge(
        self, edge_id: str, valid_to: float, *, conn: sqlite3.Connection | None = None
    ) -> bool:
        """Set valid_to on an open edge, closing it."""
        c = conn or self._conn()
        cur = c.execute(
            "UPDATE memory_edges SET valid_to=? WHERE edge_id=? AND valid_to IS NULL",
            (valid_to, edge_id),
        )
        return cur.rowcount > 0

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        """Retrieve an edge by ID."""
        row = (
            self._conn()
            .execute("SELECT * FROM memory_edges WHERE edge_id=?", (edge_id,))
            .fetchone()
        )
        if row is None:
            return None
        d = dict(row)
        return MemoryEdge(
            edge_id=d["edge_id"],
            namespace=d["namespace"],
            source_unit_id=d["source_unit_id"],
            target_unit_id=d["target_unit_id"],
            source_type=d["source_type"],
            target_type=d["target_type"],
            relation=MemoryRelation(d["relation"]),
            weight=d["weight"],
            valid_from=d["valid_from"],
            valid_to=d.get("valid_to"),
            created_at=d["created_at"],
            provenance=d.get("provenance"),
        )

    # ------------------------------------------------------------------
    # Community labels
    # ------------------------------------------------------------------

    def get_community_id(self, namespace: str, node_type: str, node_id: str) -> str | None:
        """Look up community ID for a node."""
        row = (
            self._conn()
            .execute(
                """
                SELECT community_id FROM memory_community_labels
                WHERE namespace=? AND node_type=? AND node_id=?
                """,
                (namespace, node_type, node_id),
            )
            .fetchone()
        )
        return row[0] if row is not None else None

    def put_community_label(
        self,
        namespace: str,
        node_type: str,
        node_id: str,
        community_id: str,
        weight: float = 1.0,
        updated_at: float | None = None,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Insert or update a community label."""
        ts = updated_at if updated_at is not None else time.time()
        c = conn or self._conn()
        c.execute(
            """
            INSERT INTO memory_community_labels
            (namespace, node_type, node_id, community_id, weight, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, node_type, node_id) DO UPDATE SET
            community_id=excluded.community_id,
            weight=excluded.weight,
            updated_at=excluded.updated_at
            """,
            (namespace, node_type, node_id, community_id, weight, ts),
        )

    # ------------------------------------------------------------------
    # Recursive CTE traversal
    # ------------------------------------------------------------------

    def traverse_graph(
        self,
        *,
        namespace: str,
        start_nodes: list[tuple[str, str]],
        max_depth: int = 3,
        direction: str = "bidirectional",
        as_of: float | None = None,
        relation_types: list[str] | None = None,
    ) -> list[GraphTraversalResult]:
        """BFS graph traversal via recursive CTE."""
        if not start_nodes:
            return []

        ts = as_of if as_of is not None else time.time()
        conn = self._conn()

        relation_filter_clause = ""
        if relation_types:
            placeholders = ",".join("?" for _ in relation_types)
            relation_filter_clause = f"AND e.relation IN ({placeholders})"

        anchor_parts = []
        anchor_params = []
        for node_id, node_type in start_nodes:
            anchor_parts.append("SELECT ?, ?, 0, ',' || ? || ',', NULL, 1.0, NULL")
            anchor_params.extend([node_id, node_type, node_id])
        anchor_sql = " UNION ALL ".join(anchor_parts)

        query = _build_bfs_query(direction, anchor_sql, relation_filter_clause)

        params: tuple[Any, ...] = (*anchor_params, max_depth, namespace, ts, ts)
        if relation_types:
            params = (*params, *relation_types)

        rows = conn.execute(query, params).fetchall()

        visited: dict[tuple[str, str], tuple[int, str | None, float | None, str | None]] = {}
        for row in rows:
            key = (row["node_id"], row["node_type"])
            depth = int(row["depth"])
            rel = row["relation"]
            wt = row["weight"]
            parent = row["parent_node_id"]
            if key not in visited or depth < visited[key][0]:
                visited[key] = (depth, rel, wt, parent)

        return [
            GraphTraversalResult(
                node_id=k[0],
                node_type=k[1],
                depth=d,
                last_edge_relation=r,
                last_edge_weight=w,
                parent_node_id=p,
            )
            for k, (d, r, w, p) in visited.items()
        ]


# ------------------------------------------------------------------
# BFS query builder (private)
# ------------------------------------------------------------------


def _build_bfs_query(direction: str, anchor_sql: str, relation_filter: str) -> str:
    """Build the recursive CTE query for the given traversal direction."""
    if direction == "forward":
        return f"""
            WITH RECURSIVE bfs(node_id, node_type, depth, path, relation, weight, parent_node_id) AS (
                {anchor_sql}
                UNION ALL
                SELECT
                    e.target_unit_id,
                    e.target_type,
                    b.depth + 1,
                    b.path || e.target_unit_id || ',',
                    e.relation,
                    e.weight,
                    b.node_id
                FROM memory_edges e
                JOIN bfs b ON (e.source_unit_id = b.node_id AND e.source_type = b.node_type)
                WHERE b.depth < ?
                  AND e.namespace = ?
                  AND e.valid_from <= ?
                  AND (e.valid_to IS NULL OR e.valid_to > ?)
                  AND instr(b.path, ',' || e.target_unit_id || ',') = 0
                  {relation_filter}
            )
            SELECT node_id, node_type, depth, relation, weight, parent_node_id FROM bfs WHERE depth > 0;
        """
    elif direction == "backward":
        return f"""
            WITH RECURSIVE bfs(node_id, node_type, depth, path, relation, weight, parent_node_id) AS (
                {anchor_sql}
                UNION ALL
                SELECT
                    e.source_unit_id,
                    e.source_type,
                    b.depth + 1,
                    b.path || e.source_unit_id || ',',
                    e.relation,
                    e.weight,
                    b.node_id
                FROM memory_edges e
                JOIN bfs b ON (e.target_unit_id = b.node_id AND e.target_type = b.node_type)
                WHERE b.depth < ?
                  AND e.namespace = ?
                  AND e.valid_from <= ?
                  AND (e.valid_to IS NULL OR e.valid_to > ?)
                  AND instr(b.path, ',' || e.source_unit_id || ',') = 0
                  {relation_filter}
            )
            SELECT node_id, node_type, depth, relation, weight, parent_node_id FROM bfs WHERE depth > 0;
        """
    else:  # bidirectional
        return f"""
            WITH RECURSIVE bfs(node_id, node_type, depth, path, relation, weight, parent_node_id) AS (
                {anchor_sql}
                UNION ALL
                SELECT
                    CASE WHEN e.source_unit_id = b.node_id THEN e.target_unit_id ELSE e.source_unit_id END,
                    CASE WHEN e.source_unit_id = b.node_id THEN e.target_type ELSE e.source_type END,
                    b.depth + 1,
                    b.path || (CASE WHEN e.source_unit_id = b.node_id THEN e.target_unit_id ELSE e.source_unit_id END) || ',',
                    e.relation,
                    e.weight,
                    b.node_id
                FROM memory_edges e
                JOIN bfs b ON (
                    (e.source_unit_id = b.node_id AND e.source_type = b.node_type)
                    OR (e.target_unit_id = b.node_id AND e.target_type = b.node_type)
                )
                WHERE b.depth < ?
                  AND e.namespace = ?
                  AND e.valid_from <= ?
                  AND (e.valid_to IS NULL OR e.valid_to > ?)
                  AND instr(b.path, ',' || (CASE WHEN e.source_unit_id = b.node_id THEN e.target_unit_id ELSE e.source_unit_id END) || ',') = 0
                  {relation_filter}
            )
            SELECT node_id, node_type, depth, relation, weight, parent_node_id FROM bfs WHERE depth > 0;
        """
