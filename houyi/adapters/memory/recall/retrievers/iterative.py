"""Iterative multi-hop retriever for entity-state chains.

The retriever resolves simple chains by repeatedly querying the
materialized entity-state view. It is intentionally conservative: if a
chain cannot be parsed or a hop has no active row, retrieval returns the
partial evidence gathered so far instead of fabricating the missing
link.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from houyi.adapters.memory.backends.base import EntityStateView
from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.retrievers.entity_state import _extract_event_time
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, EntityStateRecord


@dataclass(frozen=True)
class ChainPlan:
    """Parsed chain head and ordered attributes."""

    head: str
    attributes: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class GapAnalysis:
    """Subqueries needed to fill evidence gaps."""

    subqueries: tuple[RecallQuery, ...] = ()


class GapAnalyzer(Protocol):
    """Analyze first-round evidence and propose bounded subqueries."""

    def analyze(
        self,
        query: RecallQuery,
        candidates: list[RecallCandidate],
        *,
        max_subqueries: int,
    ) -> GapAnalysis:
        """Return follow-up recall queries for missing relation hops."""
        ...


class RuleGapAnalyzer:
    """Build relation-chain subqueries from parsed chain syntax."""

    def analyze(
        self,
        query: RecallQuery,
        candidates: list[RecallCandidate],
        *,
        max_subqueries: int,
    ) -> GapAnalysis:
        plan = _parse_chain(query.text, max_hops=max_subqueries + 1)
        if plan is None or not candidates:
            return GapAnalysis()

        subqueries: list[RecallQuery] = []
        next_attrs = plan.attributes[1:]
        frontier = [candidate.fact.object for candidate in candidates]
        for entity, attribute in zip(frontier, next_attrs, strict=False):
            subqueries.append(
                RecallQuery(
                    text=f"{attribute} of {entity}",
                    namespace=query.namespace,
                    entity_hint=entity,
                    attribute_hint=attribute,
                    top_k=query.top_k,
                )
            )
            if len(subqueries) >= max_subqueries:
                break
        return GapAnalysis(subqueries=tuple(subqueries))


class IterativeMultiHopRetriever(Retriever):
    """Resolve entity-state chains one hop at a time."""

    def __init__(
        self,
        view: EntityStateView,
        *,
        max_hops: int = 4,
        delegate: Retriever | None = None,
        analyzer: GapAnalyzer | None = None,
        max_subqueries: int = 3,
    ) -> None:
        if view is None:
            raise ValueError("view is required")
        if max_hops <= 0:
            raise ValueError("max_hops must be > 0")
        if max_subqueries <= 0:
            raise ValueError("max_subqueries must be > 0")
        self._view = view
        self._max_hops = max_hops
        self._delegate = delegate
        self._analyzer = analyzer or RuleGapAnalyzer()
        self._max_subqueries = max_subqueries

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        if self._delegate is not None:
            return await self._retrieve_with_delegate(query, ctx)
        return await self._retrieve_chain(query)

    async def _retrieve_chain(self, query: RecallQuery) -> list[RecallCandidate]:
        plan = _parse_chain(query.text, max_hops=self._max_hops)
        if plan is None:
            return []

        entity = plan.head
        candidates: list[RecallCandidate] = []
        for hop_index, attribute in enumerate(plan.attributes, start=1):
            rows = await asyncio.to_thread(
                self._view.get_active,
                query.namespace,
                entity,
                attribute,
            )
            if not rows:
                break
            row = rows[0]
            candidates.append(
                _candidate_from_row(
                    row,
                    self.name,
                    hop_index=hop_index,
                    plan=plan,
                )
            )
            entity = row.value
        return candidates

    async def _retrieve_with_delegate(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        delegate = self._delegate
        if delegate is None:
            return []

        plan = _parse_chain(query.text, max_hops=self._max_hops)
        if plan is None:
            return []

        first_query = RecallQuery(
            text=query.text,
            namespace=query.namespace,
            entity_hint=plan.head,
            top_k=query.top_k,
        )
        first_round = await delegate.retrieve(first_query, ctx)
        for candidate in first_round:
            _mark_round(candidate, 1)

        analysis = self._analyzer.analyze(
            query,
            first_round,
            max_subqueries=self._max_subqueries,
        )
        second_round: list[RecallCandidate] = []
        for subquery in analysis.subqueries[: self._max_subqueries]:
            hits = await delegate.retrieve(subquery, ctx)
            for candidate in hits:
                _mark_round(candidate, 2)
            second_round.extend(hits)
        return [*first_round, *second_round]


def _parse_chain(text: str, *, max_hops: int) -> ChainPlan | None:
    stripped = text.strip().strip("?")
    if not stripped:
        return None

    if "\u7684" in stripped:
        parts = [p.strip() for p in stripped.split("\u7684") if p.strip()]
        if len(parts) >= 3:
            return ChainPlan(parts[0], tuple(parts[1 : max_hops + 1]), "zh_chain")

    lowered = stripped.casefold()
    separators = [" of ", " for ", " from "]
    for sep in separators:
        if sep in lowered:
            parts = [p.strip() for p in lowered.split(sep) if p.strip()]
            if len(parts) >= 2:
                head = parts[-1]
                attrs = tuple(reversed(parts[:-1]))[:max_hops]
                return ChainPlan(head, attrs, "en_of_chain")

    return None


def _candidate_from_row(
    row: EntityStateRecord,
    retriever_name: str,
    *,
    hop_index: int,
    plan: ChainPlan,
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
        event_time=_extract_event_time(row.qualifiers),
    )
    return RecallCandidate(
        fact=fact,
        score=max(0.1, 1.0 / hop_index),
        matched_by=RetrieverKind.ITERATIVE,
        retriever_name=retriever_name,
        signals={
            "hop_index": hop_index,
            "chain_head": plan.head,
            "chain_source": plan.source,
            "chain_length": len(plan.attributes),
        },
        explanation=f"chain hop {hop_index}: {row.entity}.{row.attribute}",
    )


def _mark_round(candidate: RecallCandidate, round_index: int) -> None:
    candidate.matched_by = RetrieverKind.ITERATIVE
    candidate.retriever_name = "IterativeMultiHopRetriever"
    candidate.signals = dict(candidate.signals)
    candidate.signals["iteration_round"] = round_index


__all__ = [
    "ChainPlan",
    "GapAnalysis",
    "GapAnalyzer",
    "IterativeMultiHopRetriever",
    "RuleGapAnalyzer",
]
