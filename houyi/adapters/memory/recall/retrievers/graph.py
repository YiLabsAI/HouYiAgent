from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from houyi.adapters.memory.backends.base import EntityStateView, EventView, MemoryBackend
from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.retrievers.entity_state import _infer_entity_attribute
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    EntityStateRecord,
    GraphTraversalResult,
    MemoryEvent,
    MemoryRecord,
)

logger = logging.getLogger(__name__)

_INTERNAL_METADATA_KEYS: frozenset[str] = frozenset({"session_id", "turn_id"})


def _is_identity_anchor(row: EntityStateRecord) -> bool:
    """Return True for self-loop identity anchor rows (e.g. 'X | identity | X').

    The extractor's auto-derive mechanism upserts these rows purely as graph
    edge endpoints; they carry no answer value and otherwise flood recall ranks
    with high coverage bonuses, so the graph retriever must not surface them.
    """
    return (
        row.attribute == "identity"
        and row.entity.strip().casefold() == row.value.strip().casefold()
    )


class GraphRetriever(Retriever):
    """Retrieve connected candidates from the HouYi-Mesh GraphIndex using SQLite CTE BFS.

    By using Entity-First seed discovery, we locate initial anchor points in both
    EntityStateView and Memories FTS. From there, we perform a bi-temporal-aware
    BFS traversal (up to depth=3) over memory_edges, and return traversed records.
    """

    def __init__(
        self, backend: MemoryBackend, view: EntityStateView, event_view: EventView | None = None
    ) -> None:
        if backend is None:
            raise ValueError("backend is required")
        if view is None:
            raise ValueError("view is required")
        self._backend = backend
        self._view = view
        self._event_view = event_view

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        # 1. Seed Discovery: Find initial start nodes
        seed_nodes = await self._discover_seeds(query)
        if not seed_nodes:
            return []

        from houyi.adapters.memory.recall.types import QueryType

        # Per-query-type traversal profile: (relation_types, max_depth).
        #
        # factual_lookup is intentionally shallow and relation-restricted.
        # A single-entity attribute answer lives on the seed's own row or a
        # direct equivalence/support edge; the seed set already contains every
        # active state row for the entity (see _discover_seeds), so a deep
        # bidirectional related_to fan-out only drags in the entity's unrelated
        # attributes (e.g. "apple pie", "boot camp") that crowd the true
        # evidence out of the fused top-k. Keeping same_as/supports preserves
        # graph's unique value here — coreference/alias resolution — without
        # the flooding. Multi-hop and temporal query types keep the full
        # depth-3 traversal where deep chains are the whole point.
        relation_types: list[str] | None = None
        max_depth = 3
        if ctx.query_type == QueryType.TEMPORAL_QUERY:
            relation_types = [
                "precedes",
                "causes",
                "same_as",
                "related_to",
                "supports",
                "participates_in",
                "involves",
                "narrative_next",
            ]
        elif ctx.query_type == QueryType.RELATIONAL_CHAIN:
            relation_types = [
                "causes",
                "related_to",
                "same_as",
                "supports",
                "participates_in",
                "involves",
            ]
        elif ctx.query_type == QueryType.NEGATION_CHECK:
            relation_types = ["related_to", "same_as", "supports"]
        elif ctx.query_type == QueryType.FACTUAL_LOOKUP:
            # If the query asks for lists, quantities, or plural attributes (e.g. books, places, tournaments),
            # allow deep RELATED_TO traversal to retrieve all connected items, otherwise restrict to depth-1 same_as.
            query_lower = (query.text or "").lower()
            is_aggregation = any(
                kw in query_lower
                for kw in [
                    "books",
                    "places",
                    "tournaments",
                    "hobbies",
                    "interests",
                    "movies",
                    "how many",
                    "how often",
                    "list of",
                    "kind of",
                    "types of",
                ]
            )
            if is_aggregation:
                relation_types = ["causes", "related_to", "same_as", "supports"]
                max_depth = 3
            else:
                relation_types = ["same_as", "supports"]
                max_depth = 1

        # 2. SQLite CTE Graph Traversal (Offloaded to a background thread)
        traversed_nodes = await asyncio.to_thread(
            self._backend.traverse_graph,
            namespace=query.namespace,
            start_nodes=seed_nodes[:20],
            max_depth=max_depth,
            direction="bidirectional",
            as_of=query.as_of,
            relation_types=relation_types,
        )

        if not traversed_nodes:
            return []

        # 3. Resolve traversed node IDs into RecallCandidates (Offloaded to a thread)
        candidates = await asyncio.to_thread(self._resolve_nodes, traversed_nodes)
        return candidates

    async def _discover_seeds(self, query: RecallQuery) -> list[tuple[str, str]]:
        """Identify seed entities and facts from the Query."""
        seeds: dict[tuple[str, str], None] = {}

        # Check caller hint
        if query.entity_hint:
            ent = query.entity_hint.strip()
            if ent:
                seeds[(ent, "state")] = None
                # Also gather active state_ids for this entity
                active_rows = await asyncio.to_thread(self._view.get_active, query.namespace, ent)
                for r in active_rows:
                    seeds[(r.state_id, "state")] = None

        # Try incorporating entity_state's robust entity inference for highly accurate seeds
        inferred = _infer_entity_attribute(query)
        if inferred and inferred.entity:
            ent = inferred.entity.strip()
            seeds[(ent, "state")] = None
            try:
                active_rows = await asyncio.to_thread(self._view.get_active, query.namespace, ent)
                for r in active_rows:
                    seeds[(r.state_id, "state")] = None
            except Exception as exc:
                logger.debug("Failed to fetch active rows for entity seed '%s': %s", ent, exc)

        # Parse text-based entities
        await self._discover_entities_from_text(query.namespace, query.text, seeds)

        # Also find direct seed facts by performing a lightweight FTS keyword scan
        # over the memories table.
        try:
            fts_hits = await asyncio.to_thread(self._backend.search_fts, query.text, limit=10)
            for rec, _ in fts_hits:
                seeds[(rec.record_id, "fact")] = None
        except Exception as exc:
            logger.debug("Lightweight FTS seed discovery bypassed: %s", exc)

        return list(seeds.keys())

    async def _discover_entities_from_text(
        self, namespace: str, text: str, seeds: dict[tuple[str, str], None]
    ) -> None:
        """Helper to parse CJK or English capital words as entity state seeds."""
        try:
            active_entities = set(await asyncio.to_thread(self._view.list_entities, namespace))
        except Exception as exc:
            logger.debug("Failed to list active entities for substring pre-filtering: %s", exc)
            active_entities = set()

        await self._discover_active_entities(namespace, text, seeds, active_entities)
        await self._discover_unseen_entities(namespace, text, seeds, active_entities)

    async def _discover_active_entities(
        self,
        namespace: str,
        text: str,
        seeds: dict[tuple[str, str], None],
        active_entities: set[str],
    ) -> None:
        """Scan text for known active entities in the namespace."""
        for ent in active_entities:
            # To prevent matching John in Johnson, enforce word boundaries for English-only entities
            if any(ord(ch) > 127 for ch in ent):
                # CJK entity: simple substring check is safe and extremely accurate
                if ent in text:
                    seeds[(ent, "state")] = None
                    try:
                        active_rows = await asyncio.to_thread(self._view.get_active, namespace, ent)
                        for r in active_rows:
                            seeds[(r.state_id, "state")] = None
                    except Exception as exc:
                        logger.debug("Failed to get active rows: %s", exc)
            else:
                # English/Latin entity: use regex word boundary to prevent partial word matches
                pattern = r"\b" + re.escape(ent) + r"\b"
                if re.search(pattern, text, re.IGNORECASE):
                    seeds[(ent, "state")] = None
                    try:
                        active_rows = await asyncio.to_thread(self._view.get_active, namespace, ent)
                        for r in active_rows:
                            seeds[(r.state_id, "state")] = None
                    except Exception as exc:
                        logger.debug("Failed to get active rows: %s", exc)

    async def _discover_unseen_entities(
        self,
        namespace: str,
        text: str,
        seeds: dict[tuple[str, str], None],
        active_entities: set[str],
    ) -> None:
        """Extract unseen proper nouns or CJK n-grams from query text."""
        for word in text.split():
            clean_word = word.strip(".,!?:;'\"()")
            if clean_word.lower().endswith("'s"):
                clean_word = clean_word[:-2].strip(".,!?:;'\"()")
            if not clean_word:
                continue
            is_entity = clean_word[0].isupper() or bool(re.search(r"[\u4e00-\u9fff]", clean_word))
            if not is_entity:
                continue
            w_low = clean_word.lower()
            if w_low in {
                "when",
                "where",
                "what",
                "who",
                "whom",
                "whose",
                "why",
                "how",
                "which",
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            }:
                continue

            has_cjk = bool(re.search(r"[\u4e00-\u9fff]", clean_word))
            if has_cjk and len(clean_word) > 8:
                await self._add_cjk_ngrams(namespace, clean_word, seeds, active_entities)
            else:
                await self._add_unseen_entity_seeds(namespace, clean_word, seeds, active_entities)

    async def _add_cjk_ngrams(
        self,
        namespace: str,
        clean_word: str,
        seeds: dict[tuple[str, str], None],
        active_entities: set[str],
    ) -> None:
        """Generate sliding window n-grams for unseen CJK sentence fragments."""
        for n in (2, 3, 4):
            for i in range(len(clean_word) - n + 1):
                cjk_sub = clean_word[i : i + n]
                # We always add cjk_sub as a potential seed node
                seeds[(cjk_sub, "state")] = None
                # Pre-filter query check: only hit DB if the sub is indeed a known active entity
                if cjk_sub in active_entities:
                    try:
                        active_rows = await asyncio.to_thread(
                            self._view.get_active, namespace, cjk_sub
                        )
                        for r in active_rows:
                            seeds[(r.state_id, "state")] = None
                    except Exception as exc:
                        logger.debug("Failed to get active rows: %s", exc)

    async def _add_unseen_entity_seeds(
        self,
        namespace: str,
        clean_word: str,
        seeds: dict[tuple[str, str], None],
        active_entities: set[str],
    ) -> None:
        """Add seed mappings for a standalone unseen entity."""
        seeds[(clean_word, "state")] = None
        # Pre-filter query check: only hit DB if the word is indeed a known active entity
        if clean_word in active_entities:
            try:
                active_rows = await asyncio.to_thread(self._view.get_active, namespace, clean_word)
                for r in active_rows:
                    seeds[(r.state_id, "state")] = None
            except Exception as exc:
                logger.debug("Failed to get active rows: %s", exc)

    def _resolve_nodes(self, traversed_nodes: list[GraphTraversalResult]) -> list[RecallCandidate]:
        """Load traversed node IDs and construct RecallCandidates."""
        candidates: list[RecallCandidate] = []

        for node in traversed_nodes:
            node_id: str = node.node_id
            node_type: str = node.node_type
            depth: int = node.depth
            relation: str | None = node.last_edge_relation
            weight: float | None = node.last_edge_weight

            # Base score with slower decay (e.g. max(5.0, 10.0 - depth * 1.0))
            base_score = max(5.0, 10.0 - depth * 1.0)

            # Relation boost (e.g. causes -> 1.4x, precedes -> 1.3x)
            rel_boost = 1.0
            if relation:
                rel_boost = {
                    "causes": 1.4,
                    "precedes": 1.3,
                    "narrative_next": 1.2,
                    "participates_in": 1.3,
                    "involves": 1.2,
                    "supports": 1.2,
                    "same_as": 1.1,
                    "related_to": 1.0,
                }.get(relation.lower(), 1.0)

            # Weight factor (bounded between 0.5 and 2.0)
            weight_factor = max(0.5, min(2.0, weight if weight is not None else 1.0))

            final_score = base_score * rel_boost * weight_factor

            if node_type == "state":
                row = self._view.get_by_id(node_id)
                if row is not None and not _is_identity_anchor(row):
                    candidates.append(
                        self._candidate_from_state_row(
                            row, final_score, depth, relation, weight, node.parent_node_id
                        )
                    )
            elif node_type == "event":
                if self._event_view is not None:
                    event = self._event_view.get_event(node_id)
                    if event is not None:
                        candidates.append(
                            self._candidate_from_event(
                                event, final_score, depth, relation, weight, node.parent_node_id
                            )
                        )
            elif node_type == "fact":
                rec = self._backend.get_by_id(node_id)
                if rec is not None:
                    candidates.append(
                        self._candidate_from_record(
                            rec, final_score, depth, relation, weight, node.parent_node_id
                        )
                    )

        return candidates

    def _candidate_from_event(
        self,
        event: MemoryEvent,
        score: float,
        depth: int,
        relation: str | None = None,
        weight: float | None = None,
        parent_node_id: str | None = None,
    ) -> RecallCandidate:
        fact = AtomicFact(
            subject=event.subject,
            predicate=event.action,
            object=f"{event.object} ({event.timestamp})",
            certainty=event.certainty,
            source_anchor=event.source_anchor,
            qualifiers=event.qualifiers,
            event_time=event.timestamp,
        )
        signals: dict[str, Any] = {
            "bfs_depth": depth,
            "node_type": "event",
            "event_id": event.event_id,
        }
        if relation:
            signals["last_edge_relation"] = relation
        if weight is not None:
            signals["last_edge_weight"] = weight
        if parent_node_id:
            signals["parent_node_id"] = parent_node_id

        return RecallCandidate(
            fact=fact,
            score=score,
            matched_by=RetrieverKind.GRAPH,
            retriever_name=self.name,
            signals=signals,
            explanation=f"graph BFS (depth={depth}) event: {event.subject} {event.action} {event.object}",
        )

    def _candidate_from_state_row(
        self,
        row: EntityStateRecord,
        score: float,
        depth: int,
        relation: str | None = None,
        weight: float | None = None,
        parent_node_id: str | None = None,
    ) -> RecallCandidate:
        fact = AtomicFact(
            subject=row.entity,
            predicate=row.attribute,
            object=row.value,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            certainty=row.certainty,
            source_anchor=row.source_unit_id or row.state_id,
            qualifiers=row.qualifiers,
        )
        signals: dict[str, Any] = {
            "entity": row.entity,
            "attribute": row.attribute,
            "bfs_depth": depth,
            "node_type": "state",
        }
        if relation:
            signals["last_edge_relation"] = relation
        if weight is not None:
            signals["last_edge_weight"] = weight
        if parent_node_id:
            signals["parent_node_id"] = parent_node_id

        return RecallCandidate(
            fact=fact,
            score=score,
            matched_by=RetrieverKind.GRAPH,
            retriever_name=self.name,
            signals=signals,
            explanation=f"graph BFS (depth={depth}) for {row.entity}.{row.attribute}",
        )

    def _candidate_from_record(
        self,
        record: MemoryRecord,
        score: float,
        depth: int,
        relation: str | None = None,
        weight: float | None = None,
        parent_node_id: str | None = None,
    ) -> RecallCandidate:
        fact = AtomicFact(
            subject=record.key,
            predicate="content",
            object=record.content,
            certainty=Certainty.CERTAIN,
            source_anchor=record.record_id,
            qualifiers={
                k: str(v)
                for k, v in record.metadata.items()
                if isinstance(k, str)
                and isinstance(v, (str, int, float))
                and k not in _INTERNAL_METADATA_KEYS
            },
        )
        signals: dict[str, Any] = {
            "scope": record.scope.value,
            "memory_type": record.memory_type.value,
            "bfs_depth": depth,
            "node_type": "fact",
        }
        if relation:
            signals["last_edge_relation"] = relation
        if weight is not None:
            signals["last_edge_weight"] = weight
        if parent_node_id:
            signals["parent_node_id"] = parent_node_id

        return RecallCandidate(
            fact=fact,
            score=score,
            matched_by=RetrieverKind.GRAPH,
            retriever_name=self.name,
            signals=signals,
            explanation=f"graph BFS (depth={depth}) for {record.key}",
        )
