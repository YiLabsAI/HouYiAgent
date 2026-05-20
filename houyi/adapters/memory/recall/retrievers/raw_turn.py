"""Raw-turn fallback retriever placeholder.

The storage interface for raw conversation turns is intentionally not
assumed here. The retriever exposes the stable recall contract and
returns no candidates until a raw-turn index is injected by callers.
"""

from __future__ import annotations

from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.types import RecallCandidate, RecallQuery, RetrieverContext


class RawTurnLogRetriever(Retriever):
    """Fallback retriever for unstructured turn logs."""

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        return []


__all__ = ["RawTurnLogRetriever"]
