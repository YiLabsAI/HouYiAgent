"""Recall-layer wrapper around the store-layer VectorRetriever.

The store-layer ~houyi.adapters.memory.vector_retriever.VectorRetriever
returns (MemoryRecord, similarity) tuples and intentionally knows
nothing about AtomicFact or RecallCandidate. This
module adapts it to the recall pipeline so the orchestrator can dispatch
a vector slot just like any other retriever.

Design notes:

- The wrapped retriever still owns the two-stage prefilter → vector
 rerank logic (). We don't reimplement that here.
- Conversion MemoryRecord → AtomicFact is intentionally lossy: a
 free-form memory unit has no native subject/predicate/object. We
 synthesize subject = record.key and predicate = "content" so
 downstream consumers can still index and explain the candidate, and
 preserve provenance via source_anchor = record.record_id.
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

    The 6-tuple is synthesized so the candidate plays nicely with the
    rest of the recall pipeline (fusion, IDK guard, source rehydration).
    """
    fact = AtomicFact(
        subject=record.key,
        predicate="content",
        object=record.content,
        certainty=Certainty.CERTAIN,
        source_anchor=record.record_id,
        qualifiers={
            **(
                {str(k): str(v) for k, v in record.metadata.items() if isinstance(k, str)}
                if isinstance(record.metadata, dict)
                else {}
            ),
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
