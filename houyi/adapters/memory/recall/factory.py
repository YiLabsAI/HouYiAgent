"""Assemble the default RecallOrchestrator.

The factory wires entity_state, graph, timeline, iterative, raw_turn,
and (when available) event and vector retrievers. The backend itself
serves as the EventView provider since SQLiteMemoryBackend implements
all EventView methods (add_event, get_events_by_subject, etc.); no
separate event_view parameter is needed.
"""

from __future__ import annotations

from typing import Any

from houyi.adapters.embedding import EmbeddingProvider
from houyi.adapters.memory.backends.base import EntityStateView, EventView, MemoryBackend
from houyi.adapters.memory.recall.enumeration import EnumerationBooster
from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator, RecallPipelineConfig
from houyi.adapters.memory.recall.rerank_cross_encoder import build_default_reranker
from houyi.adapters.memory.recall.retrievers.entity_state import EntityStateRetriever
from houyi.adapters.memory.recall.retrievers.event import EventRetriever
from houyi.adapters.memory.recall.retrievers.graph import GraphRetriever
from houyi.adapters.memory.recall.retrievers.iterative import IterativeMultiHopRetriever
from houyi.adapters.memory.recall.retrievers.raw_turn import RawTurnLogRetriever
from houyi.adapters.memory.recall.retrievers.timeline import TimelineRetriever
from houyi.adapters.memory.recall.retrievers.vector import VectorRecallRetriever
from houyi.adapters.memory.recall.router import CascadingRouter, Tier0RuleRouter
from houyi.adapters.memory.vector_retriever import VectorRetriever


def _build_default_recall_orchestrator(
    *,
    backend: MemoryBackend,
    entity_state: EntityStateView,
    embedding_provider: EmbeddingProvider | None = None,
    config: RecallPipelineConfig | None = None,
    llm_adapter: Any | None = None,
) -> RecallOrchestrator:
    """Assemble the default RecallOrchestrator.

    The vector retriever is registered only when an embedding provider
    is supplied. The route table in RecallPipelineConfig already lists
    vector as opt-in, so omitting it is safe: routes that mention
    vector simply skip that slot.

    The event retriever is registered when the backend implements the
    EventView interface (duck-typed: has add_event, get_events_by_subject,
    etc.). SQLiteMemoryBackend satisfies this by construction, so the
    event path is always active in the default configuration.

    Returns a fully-wired RecallOrchestrator. The caller is responsible
    for the orchestrator's lifetime; the factory holds no state.
    """
    if backend is None:
        raise ValueError("backend is required")
    if entity_state is None:
        raise ValueError("entity_state is required")

    # The backend itself serves as EventView -- SQLiteMemoryBackend
    # implements add_event, get_events_by_subject, etc. A separate
    # event_view parameter would be redundant since the backend already
    # has all the methods.
    event_view: EventView | None = None
    if isinstance(backend, EventView):
        event_view = backend
    elif hasattr(backend, "add_event") and hasattr(backend, "get_events_by_subject"):
        event_view = backend  # type: ignore[assignment]

    entity_state_retriever = EntityStateRetriever(entity_state)
    retrievers: dict[str, Any] = {
        "entity_state": entity_state_retriever,
        "graph": GraphRetriever(backend, entity_state, event_view=event_view),
        "timeline": TimelineRetriever(entity_state),
        "iterative": IterativeMultiHopRetriever(
            entity_state,
            delegate=entity_state_retriever,
        ),
        "raw_turn": RawTurnLogRetriever(),
    }
    if event_view is not None:
        retrievers["event"] = EventRetriever(event_view)
    if embedding_provider is not None:
        store_vector = VectorRetriever(backend, embedding_provider)
        retrievers["vector"] = VectorRecallRetriever(store_vector)

    router = CascadingRouter(tier0=Tier0RuleRouter())
    return RecallOrchestrator(
        router=router,
        retrievers=retrievers,
        config=config,
        enum_booster=EnumerationBooster(backend),
        reranker=build_default_reranker(llm_adapter=llm_adapter),
    )


__all__ = ["_build_default_recall_orchestrator"]
