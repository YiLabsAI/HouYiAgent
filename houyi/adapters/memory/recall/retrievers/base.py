"""Retriever ABC — one method, one contract.

Every retriever takes a RecallQuery + RetrieverContext
and returns a list of RecallCandidate. The contract is
deliberately narrow:

1. Stateless: retrievers must hold no per-call state. All
 per-request data flows through query and ctx; instance
 state is for infrastructure handles only (e.g. an
 EntityStateView bound at construction).
2. Best-effort: a retriever that has nothing relevant returns
 [] rather than raising. Genuine errors (storage broken, LLM
 adapter missing for a retriever that needs one) raise
 RetrieverError; the orchestrator catches and converts to
 trace entries so one broken retriever cannot kill the whole call.
3. No mutation: retrievers must not write to memory. The recall
 pipeline is pure read-side; any "promote to hot tier" decision
 belongs to background memory maintenance, not here.
4. No knowledge of routing: retrievers do not inspect
 QueryType; the router decides who runs and the orchestrator
 dispatches. This keeps each retriever testable in isolation.

The ABC lives in its own module (rather than recall/types.py) so
the value types stay free of behavior — a useful split when type
files get re-exported into prompts and other read-only contexts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
)


class RetrieverError(RuntimeError):
    """Raised when a retriever cannot complete due to infrastructure.

    Reserved for genuine failures (DB down, LLM adapter missing for a
    retriever that requires one). Empty-result is not an error —
    return []. The orchestrator catches RetrieverError and
    converts it into a trace entry; the recall call as a whole still
    completes (with whatever the other retrievers produced).
    """


class Retriever(ABC):
    """Abstract base for all memory retrievers.

    Subclasses implement exactly retrieve. The class name
    appears in RecallCandidate.retriever_name and in trace
    logs, so subclasses SHOULD use a descriptive name (e.g.
    EntityStateRetriever not EntityRet).
    """

    @property
    def name(self) -> str:
        """Default name = class name; override only for variants.

        e.g. an EntityStateRetriever configured for
        negation_check mode might override to
        "EntityStateRetriever[negation]" so trace logs disambiguate.
        """
        return type(self).__name__

    @abstractmethod
    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        """Return zero or more candidates for query.

        Implementations MUST:

        - Return [] rather than raising when nothing matches.
        - Set RecallCandidate.matched_by to the matching
        RetrieverKind value.
        - Set RecallCandidate.retriever_name to name.
        - Honor ctx.deadline_ms on a best-effort basis (no hard
        cancellation required, but loops that may run long should
        check elapsed time and break early).

        Implementations MUST NOT:

        - Inspect QueryType (the orchestrator already routed).
        - Mutate any memory state.
        - Block the event loop on synchronous I/O — use
        asyncio.to_thread for SQLite calls, mirroring the
        ingestor convention.
        """
        raise NotImplementedError  # pragma: no cover - abstract


__all__ = ["Retriever", "RetrieverError"]
