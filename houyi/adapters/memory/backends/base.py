"""Pluggable storage protocols for the Memory Engine.

Three abstract surfaces are defined here, one per persistence concern:

- MemoryBackend — coarse-grained record store + FTS5 + embedding cache; the original v1 surface and still the central store.
- EntityStateView — fast-path materialized view over (namespace, entity, attribute) triples with bi-temporal validity; the recall hot path for single-hop factual lookup and as-of queries.
- CandidateInbox — parking lot for facts that must not enter the main store (vague certainty, missing source anchor); analogous to the Memory Inbox UI surface but consumed by the writer pipeline.

Concrete drivers (e.g. SQLite) implement all three by sharing a single connection so cross-surface transactions stay atomic; alternative drivers (Postgres, in-memory) can supply their own implementations.
drivers (Postgres, in-memory) can supply their own implementations.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Any

from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    EntityStateRecord,
    GraphTraversalResult,
    MemoryEdge,
    MemoryEvent,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)


class MemoryBackend(ABC):
    """Pluggable storage backend for the Memory Engine.

    Default implementation: SQLiteMemoryBackend (stdlib sqlite3, zero deps).
    Extension points: QMDBackend, RedisBackend, etc.

    All methods are synchronous — SQLite and most local backends are inherently
    synchronous. Async callers (MemoryEngine, MemoryRetriever) invoke these
    from their own event-loop–safe wrappers.
    """

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def put(self, record: MemoryRecord) -> None:
        """Insert or update a memory record."""

    @abstractmethod
    def get(self, key: str, scope: MemoryScope) -> MemoryRecord | None:
        """Retrieve a single record by (key, scope). Returns None if missing or expired."""

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        """Retrieve a single record by record_id. Returns None if missing."""
        raise NotImplementedError

    @abstractmethod
    def list_by_scope(self, scope: MemoryScope) -> list[MemoryRecord]:
        """List non-expired records in *scope*, ordered by updated_at DESC."""

    @abstractmethod
    def list_by_type(
        self,
        memory_type: MemoryType,
        scope: MemoryScope | None = None,
    ) -> list[MemoryRecord]:
        """List non-expired records by memory type, optionally filtered by scope."""

    @abstractmethod
    def all_records(self, *, include_expired: bool = False) -> list[MemoryRecord]:
        """Return all records across all scopes."""

    @abstractmethod
    def delete(self, key: str, scope: MemoryScope) -> bool:
        """Delete a record. Returns True if found and deleted."""

    @abstractmethod
    def clear(self, scope: MemoryScope | None = None) -> int:
        """Clear records. If scope is None, clear all. Returns count deleted."""

        # ------------------------------------------------------------------
        # Full-text search (FTS5)
        # ------------------------------------------------------------------

    @abstractmethod
    def search_fts(
        self,
        query: str,
        scope: MemoryScope | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """BM25 full-text search. Returns (record, bm25_score) pairs ranked by relevance."""

        # ------------------------------------------------------------------
        # Embedding cache
        # ------------------------------------------------------------------

    @abstractmethod
    def get_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
    ) -> list[float] | None:
        """Retrieve a cached embedding vector."""

    @abstractmethod
    def put_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
        embedding: list[float],
    ) -> None:
        """Cache an embedding vector for a record."""

    @abstractmethod
    def search_embedding(
        self,
        query_embedding: list[float],
        scope: MemoryScope | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """Cosine similarity search over cached embeddings.

        Returns (record, similarity) pairs ranked by descending similarity.
        """

    @abstractmethod
    def search_vector(
        self,
        query_embedding: list[float],
        *,
        scope: MemoryScope | None = None,
        rowid_filter: list[int] | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """Vector similarity search.

        Returns (record, similarity) pairs ranked by descending similarity.
        """

    # ------------------------------------------------------------------
    # GraphIndex (HouYi-Mesh)
    # ------------------------------------------------------------------

    def add_edge(self, edge: MemoryEdge) -> None:
        """Insert or update a graph edge in memory_edges."""
        raise NotImplementedError

    def delete_edge(self, edge_id: str) -> bool:
        """Delete a graph edge. Returns True if deleted."""
        raise NotImplementedError

    def invalidate_edge(self, edge_id: str, valid_to: float) -> bool:
        """Close/invalidate an active edge."""
        raise NotImplementedError

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        """Retrieve a graph edge."""
        raise NotImplementedError

    def get_community_id(
        self,
        namespace: str,
        node_type: str,
        node_id: str,
    ) -> str | None:
        """Retrieve the community_id for a given node."""
        raise NotImplementedError

    def put_community_label(
        self,
        namespace: str,
        node_type: str,
        node_id: str,
        community_id: str,
        weight: float = 1.0,
        updated_at: float | None = None,
    ) -> None:
        """Insert or update a community label."""
        raise NotImplementedError

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
        """Perform a recursive BFS traversal from start_nodes using SQLite CTE."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Any:
        """Return a transaction context manager.

        Default implementation is a no-op; subclasses override as needed.
        """
        yield

    def close(self) -> None:  # noqa: B027
        """Release resources. Default is a no-op; subclasses override as needed."""


class EntityStateView(ABC):
    """Materialized current/historical view over entity attribute values.

    Each row encodes the value of a (namespace, entity, attribute)
    triple over a closed-open [valid_from, valid_to) interval. The
    view is the recall fast path for factual lookup ("what is X's
    current Y?"), single-hop relational queries, and bi-temporal
    as-of queries. It is kept in sync by the writer pipeline:
    every accepted AtomicFact is projected into one row, and any
    superseded active row is closed by setting its valid_to.
    """

    @abstractmethod
    def upsert(
        self,
        namespace: str,
        entity: str,
        attribute: str,
        value: str,
        *,
        certainty: Certainty = Certainty.CERTAIN,
        valid_from: float | None = None,
        source_unit_id: str | None = None,
        qualifiers: dict[str, str] | None = None,
        accumulate: bool = False,
    ) -> EntityStateRecord:
        """Insert a new active state, closing any prior active row first.

        - If a row with valid_to IS NULL already exists for the
        triple, its valid_to is set to the new valid_from.
        - The new row is inserted with valid_to = NULL.
        - valid_from must be greater than or equal to any existing
        active row's valid_from; otherwise raise ValueError.
        """

    @abstractmethod
    def invalidate(
        self,
        namespace: str,
        entity: str,
        attribute: str,
        *,
        valid_to: float | None = None,
    ) -> bool:
        """Close the currently active row without inserting a successor.

        Returns True if an active row was closed.
        """

    @abstractmethod
    def get_active(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        """Return all currently active rows for an entity."""

    def get_by_id(self, state_id: str) -> EntityStateRecord | None:
        """Retrieve a single EntityStateRecord by state_id. Returns None if missing."""
        raise NotImplementedError

    @abstractmethod
    def get_as_of(
        self,
        namespace: str,
        entity: str,
        ts: float,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        """Return rows that were active at instant ts.

        A row is considered active when valid_from <= ts and
        (valid_to IS NULL OR valid_to > ts).
        """

    @abstractmethod
    def get_history(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        """Return every recorded version of an attribute, newest first."""

    @abstractmethod
    def list_entities(self, namespace: str) -> list[str]:
        """Return all distinct entity names that have active rows.

        Used by recall-time full enumeration when the query scope
        is not restricted to a specific entity.
        """


class CandidateInbox(ABC):
    """Parking lot for facts that must not enter the main entity-state view.

    Two routing reasons share this surface:

    - vague : the fact has certainty=VAGUE and would
    pollute the recall fast path with guesses.
    - sourceless : extraction returned content but no source anchor
    could be attached, so the AtomicFact provenance contract cannot
    be honoured. The raw extraction payload is parked verbatim so
    the agent can replay it once a source becomes available.

    Both reasons keep enough structured columns to support targeted
    lookups, plus the full serialized payload for later replay.
    """

    @abstractmethod
    def add(self, namespace: str, fact: AtomicFact) -> str:
        """Persist a vague-certainty fact and return its candidate id."""

    @abstractmethod
    def add_sourceless(
        self,
        namespace: str,
        raw_payload: dict[str, Any],
    ) -> str:
        """Persist an extraction payload that lacks a usable source anchor."""

    @abstractmethod
    def list_for(
        self,
        namespace: str,
        entity: str | None = None,
        attribute: str | None = None,
        reason: str | None = None,
    ) -> list[AtomicFact]:
        """Return parked candidates, optionally filtered.

        reason=None means vague (the historical default and the
        only reason whose payload round-trips into AtomicFact).
        """

    @abstractmethod
    def list_sourceless(self, namespace: str) -> list[dict[str, Any]]:
        """Return raw payloads parked under reason='sourceless'."""


class EventView(ABC):
    """Read/write surface for first-class MemoryEvent records.

    Events are immutable temporal occurrences. valid_to on a MemoryEvent
    means the system no longer remembers it (logical deletion), NOT that
    the event ended.
    """

    @abstractmethod
    def add_event(self, event: MemoryEvent) -> MemoryEvent:
        """INSERT OR REPLACE an event. Returns the stored event."""

    @abstractmethod
    def get_event(self, event_id: str) -> MemoryEvent | None:
        """Retrieve a single event by event_id."""

    @abstractmethod
    def get_events_by_subject(self, namespace: str, subject: str) -> list[MemoryEvent]:
        """Return all active events for a given entity."""

    @abstractmethod
    def get_events_by_subject_and_action(
        self,
        namespace: str,
        subject: str,
        action: str,
    ) -> list[MemoryEvent]:
        """Return active events matching both subject and action."""
