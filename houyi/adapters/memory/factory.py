"""Public factory for the memory subsystem.

build_memory_engine is the canonical entry point for assembling a
fully-wired MemoryEngine. Callers (bench, studio, downstream SDK
users) get a single object that hides the concrete backend, the L1
extractor, the entity-state view, the candidate inbox, the turn
writer, the extractor and embedding-backfill workers, and the recall
orchestrator behind the small MemoryEngine surface.

The returned engine is NOT started. Callers must either call
await engine.start() / await engine.stop() explicitly or use the
async-with context manager so worker lifecycles stay deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from houyi.adapters.embedding import (
    EmbeddingProvider,
    make_embedding_provider,
)
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.extractor import AtomicFactExtractor
from houyi.adapters.memory.recall.factory import _build_default_recall_orchestrator
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.workers.embedding_backfill import EmbeddingBackfillWorker
from houyi.adapters.memory.workers.extractor_worker import ExtractorWorker


def build_memory_engine(
    *,
    data_dir: str | Path | None = None,
    llm_adapter: Any | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    namespace: str = "default",
) -> MemoryEngine:
    """Construct a fully wired MemoryEngine.

    Arguments:
        data_dir: Directory where the SQLite database lives. The
            backend creates data_dir/.houyi/memory.db on first use.
            Pass None to use an in-memory database (tests).
        llm_adapter: An object exposing the awaitable chat(messages,
            ...) shape consumed by AtomicFactExtractor and the legacy
            MemoryCandidateExtractor. When None the L1 extractor still
            constructs but extract calls will fail; callers that only
            need the recall path can pass None safely.
        embedding_provider: An EmbeddingProvider instance. When None
            the engine runs in lexical-only mode: the vector retriever
            is omitted from the orchestrator and no embedding backfill
            worker is started. Pass make_embedding_provider() to follow
            the project-wide EMBEDDING_PROVIDER env contract.
        namespace: Default namespace used by the extractor worker. The
            recall path takes namespace from each RecallQuery.

    Returns:
        A MemoryEngine that is NOT started. Use:

            async with build_memory_engine(...) as engine:
                ...

        or explicit await engine.start() / await engine.stop().
    """
    backend = (
        SQLiteMemoryBackend(data_dir=data_dir)
        if data_dir is not None
        else SQLiteMemoryBackend(db_path=":memory:")
    )

    store = MemoryStore(backend=backend)
    entity_state = SQLiteEntityStateView(backend)
    candidate_inbox = SQLiteCandidateInbox(backend)

    turn_writer = TurnWriter(backend)

    # The backend itself acts as EventView -- SQLiteMemoryBackend
    # implements add_event, get_events_by_subject, etc. so no
    # separate event_view instance is needed.
    extractor_worker: ExtractorWorker | None = None
    if llm_adapter is not None:
        atomic_extractor = AtomicFactExtractor(llm_adapter)
        extractor_worker = ExtractorWorker(
            backend=backend,
            extractor=atomic_extractor,
            entity_state=entity_state,
            candidate_inbox=candidate_inbox,
            event_view=backend,
        )

    backfill_worker: EmbeddingBackfillWorker | None = None
    if embedding_provider is not None:
        backfill_worker = EmbeddingBackfillWorker(
            backend=backend,
            provider=embedding_provider,
        )

    recall_orchestrator = _build_default_recall_orchestrator(
        backend=backend,
        entity_state=entity_state,
        embedding_provider=embedding_provider,
    )

    return MemoryEngine(
        store,
        embedding_provider=embedding_provider,
        llm_adapter=llm_adapter,
        recall_orchestrator=recall_orchestrator,
        turn_writer=turn_writer,
        extractor_worker=extractor_worker,
        backfill_worker=backfill_worker,
    )


def build_memory_engine_from_env(
    *,
    data_dir: str | Path | None = None,
    llm_adapter: Any | None = None,
    namespace: str = "default",
) -> MemoryEngine:
    """Convenience wrapper that resolves the embedding provider from env.

    Honors EMBEDDING_PROVIDER and downstream provider keys via
    make_embedding_provider. When the resolved provider raises (for
    example, missing API key) the engine is built without an embedding
    provider so the recall path still works in lexical-only mode.
    """
    try:
        embedding_provider: EmbeddingProvider | None = make_embedding_provider()
    except Exception:
        embedding_provider = None
    return build_memory_engine(
        data_dir=data_dir,
        llm_adapter=llm_adapter,
        embedding_provider=embedding_provider,
        namespace=namespace,
    )


__all__ = ["build_memory_engine", "build_memory_engine_from_env"]
