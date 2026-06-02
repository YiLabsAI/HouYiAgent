from __future__ import annotations

import asyncio
import logging
import re

from houyi.adapters.memory.backends.base import EntityStateView, MemoryBackend
from houyi.adapters.memory.recall.retrievers.base import Retriever
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
    MemoryRecord,
)

logger = logging.getLogger(__name__)

_INTERNAL_METADATA_KEYS: frozenset[str] = frozenset({"session_id", "turn_id"})


class GraphRetriever(Retriever):
    """Retrieve connected candidates from the HouYi-Mesh GraphIndex using SQLite CTE BFS.

    By using Entity-First seed discovery, we locate initial anchor points in both
    EntityStateView and Memories FTS. From there, we perform a bi-temporal-aware
    BFS traversal (up to depth=3) over memory_edges, and return traversed records.
    """

    def __init__(self, backend: MemoryBackend, view: EntityStateView) -> None:
        if backend is None:
            raise ValueError("backend is required")
        if view is None:
            raise ValueError("view is required")
        self._backend = backend
        self._view = view

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        # 1. Seed Discovery: Find initial start nodes
        seed_nodes = await self._discover_seeds(query)
        if not seed_nodes:
            return []

        # 2. SQLite CTE Graph Traversal (Offloaded to a background thread)
        traversed_nodes = await asyncio.to_thread(
            self._backend.traverse_graph,
            namespace=query.namespace,
            start_nodes=seed_nodes,
            max_depth=3,
            direction="bidirectional",
            as_of=query.as_of,
        )

        if not traversed_nodes:
            return []

        # 3. Resolve traversed node IDs into RecallCandidates (Offloaded to a thread)
        candidates = await asyncio.to_thread(self._resolve_nodes, traversed_nodes)
        return candidates

    async def _discover_seeds(self, query: RecallQuery) -> list[tuple[str, str]]:
        """Identify seed entities and facts from the Query."""
        seeds: set[tuple[str, str]] = set()

        # Check caller hint
        if query.entity_hint:
            ent = query.entity_hint.strip()
            if ent:
                seeds.add((ent, "state"))
                # Also gather active state_ids for this entity
                active_rows = await asyncio.to_thread(self._view.get_active, query.namespace, ent)
                for r in active_rows:
                    seeds.add((r.state_id, "state"))

        # Parse text-based entities
        await self._discover_entities_from_text(query.namespace, query.text, seeds)

        # Also find direct seed facts by performing a lightweight FTS keyword scan
        # over the memories table.
        try:
            fts_hits = await asyncio.to_thread(self._backend.search_fts, query.text, limit=3)
            for rec, _ in fts_hits:
                seeds.add((rec.record_id, "fact"))
        except Exception as exc:
            logger.debug("Lightweight FTS seed discovery bypassed: %s", exc)

        return list(seeds)

    async def _discover_entities_from_text(
        self, namespace: str, text: str, seeds: set[tuple[str, str]]
    ) -> None:
        """Helper to parse CJK or English capital words as entity state seeds."""
        for word in text.split():
            clean_word = word.strip(".,!?:;'\"()")
            if clean_word:
                is_entity = clean_word[0].isupper() or bool(
                    re.search(r"[\u4e00-\u9fff]", clean_word)
                )
                if is_entity:
                    w_low = clean_word.lower()
                    if w_low not in {
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
                        seeds.add((clean_word, "state"))
                        active_rows = await asyncio.to_thread(
                            self._view.get_active, namespace, clean_word
                        )
                        for r in active_rows:
                            seeds.add((r.state_id, "state"))

    def _resolve_nodes(self, traversed_nodes: list[GraphTraversalResult]) -> list[RecallCandidate]:
        """Load traversed node IDs and construct RecallCandidates."""
        candidates: list[RecallCandidate] = []

        for node in traversed_nodes:
            node_id: str = node.node_id
            node_type: str = node.node_type
            depth: int = node.depth

            # Base score decreases slightly with BFS hop distance
            base_score = max(1.0, 10.0 - depth * 1.5)

            if node_type == "state":
                row = self._view.get_by_id(node_id)
                if row is not None:
                    candidates.append(self._candidate_from_state_row(row, base_score, depth))
            elif node_type == "fact":
                rec = self._backend.get_by_id(node_id)
                if rec is not None:
                    candidates.append(self._candidate_from_record(rec, base_score, depth))

        return candidates

    def _candidate_from_state_row(
        self,
        row: EntityStateRecord,
        score: float,
        depth: int,
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
        return RecallCandidate(
            fact=fact,
            score=score,
            matched_by=RetrieverKind.GRAPH,
            retriever_name=self.name,
            signals={
                "entity": row.entity,
                "attribute": row.attribute,
                "bfs_depth": depth,
                "node_type": "state",
            },
            explanation=f"graph BFS (depth={depth}) for {row.entity}.{row.attribute}",
        )

    def _candidate_from_record(
        self,
        record: MemoryRecord,
        score: float,
        depth: int,
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
        return RecallCandidate(
            fact=fact,
            score=score,
            matched_by=RetrieverKind.GRAPH,
            retriever_name=self.name,
            signals={
                "scope": record.scope.value,
                "memory_type": record.memory_type.value,
                "bfs_depth": depth,
                "node_type": "fact",
            },
            explanation=f"graph BFS (depth={depth}) for {record.key}",
        )
