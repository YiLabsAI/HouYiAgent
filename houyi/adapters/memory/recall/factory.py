"""Internal recall orchestrator factory.

Module-private helper used by build_memory_engine to assemble the
default recall stack from a backend handle, an entity-state view, and
an optional embedding provider. Not exported from the package
__init__ on purpose: callers should not construct a RecallOrchestrator
directly. They should call build_memory_engine and use MemoryEngine.

The default stack:

  - entity_state: EntityStateRetriever
  - timeline:     TimelineRetriever
  - iterative:    IterativeMultiHopRetriever (delegate=entity_state)
  - raw_turn:     RawTurnLogRetriever
  - vector:       VectorRecallRetriever (only when embedding provider
                  is supplied; backed by VectorRetriever over the same
                  backend)

Router is the standard CascadingRouter with Tier0RuleRouter as the
deterministic first stage; tier1/tier2 LLM stages stay disabled by
default to avoid hidden cost. Fuser, deduplicator, reranker, and IDK
guard fall back to the orchestrator's defaults so behavior matches
RecallOrchestrator(...).
"""

from __future__ import annotations

from typing import Any, Protocol

from houyi.adapters.embedding import EmbeddingProvider
from houyi.adapters.memory.backends.base import EntityStateView, MemoryBackend
from houyi.adapters.memory.recall.orchestrator import (
    RecallOrchestrator,
    RecallPipelineConfig,
)
from houyi.adapters.memory.recall.retrievers.entity_state import EntityStateRetriever
from houyi.adapters.memory.recall.retrievers.graph import GraphRetriever
from houyi.adapters.memory.recall.retrievers.iterative import IterativeMultiHopRetriever
from houyi.adapters.memory.recall.retrievers.raw_turn import RawTurnLogRetriever
from houyi.adapters.memory.recall.retrievers.timeline import TimelineRetriever
from houyi.adapters.memory.recall.retrievers.vector import VectorRecallRetriever
from houyi.adapters.memory.recall.router import CascadingRouter, Tier0RuleRouter
from houyi.adapters.memory.vector_retriever import VectorRetriever


class _RecallBackend(Protocol):
    """Subset of MemoryBackend the recall stack needs.

    Lifted into a Protocol so unit tests can construct the orchestrator
    against a fake without standing up SQLite. Real callers always pass
    a SQLiteMemoryBackend, which already satisfies MemoryBackend.
    """

    def search_vector(self, *args: Any, **kwargs: Any) -> Any: ...


def _build_default_recall_orchestrator(
    *,
    backend: MemoryBackend,
    entity_state: EntityStateView,
    embedding_provider: EmbeddingProvider | None = None,
    config: RecallPipelineConfig | None = None,
) -> RecallOrchestrator:
    """Assemble the default RecallOrchestrator.

    The vector retriever is registered only when an embedding provider
    is supplied. The route table in RecallPipelineConfig already lists
    vector as opt-in, so omitting it is safe: routes that mention
    vector simply skip that slot.

    Returns a fully-wired RecallOrchestrator. The caller is responsible
    for the orchestrator's lifetime; the factory holds no state.
    """
    if backend is None:
        raise ValueError("backend is required")
    if entity_state is None:
        raise ValueError("entity_state is required")

    entity_state_retriever = EntityStateRetriever(entity_state)
    retrievers: dict[str, Any] = {
        "entity_state": entity_state_retriever,
        "graph": GraphRetriever(backend, entity_state),
        "timeline": TimelineRetriever(entity_state),
        "iterative": IterativeMultiHopRetriever(
            entity_state,
            delegate=entity_state_retriever,
        ),
        "raw_turn": RawTurnLogRetriever(),
    }
    if embedding_provider is not None:
        store_vector = VectorRetriever(backend, embedding_provider)
        retrievers["vector"] = VectorRecallRetriever(store_vector)

    router = CascadingRouter(tier0=Tier0RuleRouter())
    return RecallOrchestrator(
        router=router,
        retrievers=retrievers,
        config=config,
    )


__all__ = ["_build_default_recall_orchestrator"]
