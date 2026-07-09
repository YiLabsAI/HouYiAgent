"""Recall-layer wrapper around the store-layer VectorRetriever.

The store-layer ~houyi.adapters.memory.vector_retriever.VectorRetriever
returns (MemoryRecord, similarity) tuples and intentionally knows
nothing about AtomicFact or RecallCandidate. This
module adapts it to the recall pipeline so the orchestrator can dispatch
a vector slot just like any other retriever.

Design notes:

- The wrapped retriever still owns the two-stage prefilter → vector
 rerank logic (). We don't reimplement that here.
- Conversion MemoryRecord -> AtomicFact restores the real
 subject/predicate/object/source_anchor/valid_from from the record
 metadata and provenance when available (fact records promoted from
 AtomicFact carry fact_subject/predicate/object + source_ids). This
 lets a vector candidate merge with the same fact surfaced by
 entity_state/event/graph during RRF fusion, which groups by the
 proposition triple. Non-fact records fall back to the synthetic
 subject = record.key / predicate = "content" form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty, MemoryRecord

# Internal metadata keys that should never be exposed as fact qualifiers.
# These are housekeeping fields (source tracking, turn identification) that
# add noise to the LLM prompt without providing semantic value. The
# fact_subject/predicate/object triple is restored onto the AtomicFact fields
# in _candidate_from_record, so it must not also leak into qualifiers.
_INTERNAL_METADATA_KEYS: frozenset[str] = frozenset(
    {"session_id", "turn_id", "fact_subject", "fact_predicate", "fact_object"}
)

if TYPE_CHECKING:
    from houyi.adapters.memory.vector_retriever import VectorRetriever


class VectorRecallRetriever(Retriever):
    """Recall-layer Retriever wrapping a store-layer VectorRetriever.

    Construction takes the already-built store retriever so the wrapper
    has no embedding / backend wiring concerns of its own. This separation
    keeps the store-layer building block usable on its own (e.g. by the
    legacy MemoryRetriever) and lets tests inject a fake
    VectorRetriever without standing up SQLite + an embedding provider.
    """

    def __init__(self, vector_retriever: VectorRetriever) -> None:
        if vector_retriever is None:
            raise ValueError("vector_retriever is required")
        self._vector_retriever = vector_retriever

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        del ctx  # store-layer retriever has its own knobs; deadlines are
        # honored by the underlying SQLite call, not enforced here.
        hits = await self._vector_retriever.retrieve(
            query.text,
            top_k=query.top_k,
        )
        return [_candidate_from_record(record, score) for record, score in hits]


def _candidate_from_record(record: MemoryRecord, score: float) -> RecallCandidate:
    """Wrap a MemoryRecord as a recall candidate.

    Restores the real subject/predicate/object/source_anchor/valid_from
    from the record's metadata and provenance so the candidate carries
    the SAME proposition triple as the entity_state/event/graph
    retrievers. RRF fusion groups candidates by (subject, predicate,
    object, valid_day); a vector candidate with the synthetic triple
    (record.key / 'content' / record.content) and a record_id anchor
    never matched the real triple, so the same gold fact retrieved by
    both vector and entity_state was treated as two unrelated
    candidates -- RRF votes never combined, starving single-source
    golds of cross-source agreement and dropping them below the
    cross-encoder's scored pool. Falls back to the synthetic form when
    metadata lacks the triple (non-fact records).
    """
    md = record.metadata or {}
    subject = md.get("fact_subject") or record.key
    predicate = md.get("fact_predicate") or "content"
    obj = md.get("fact_object") or record.content
    src_ids = record.provenance.source_ids if record.provenance else ()
    anchor = src_ids[0] if src_ids else record.record_id
    fact = AtomicFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        certainty=Certainty.CERTAIN,
        source_anchor=anchor,
        valid_from=record.valid_from,
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
        matched_by=RetrieverKind.VECTOR,
        retriever_name="VectorRecallRetriever",
        signals={
            "scope": record.scope.value,
            "memory_type": record.memory_type.value,
            "similarity": score,
        },
        explanation=f"vector similarity={score:.3f} for {record.key}",
    )


__all__ = ["VectorRecallRetriever"]
