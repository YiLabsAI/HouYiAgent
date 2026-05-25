"""Tests for the Sprint-8 MemoryEngine facade and related factories.

Covers:
- build_memory_engine / build_memory_engine_from_env
- MemoryEngine.write_turn / flush / start / stop
- MemoryEngine async-context-manager
- _build_default_recall_orchestrator with and without an
  embedding provider
- _candidate_to_memory_recall adapter

These tests stay at the public-facade layer; lower-level components
(TurnWriter, ExtractorWorker, EmbeddingBackfillWorker, recall
retrievers) already have dedicated unit tests elsewhere.
"""

from __future__ import annotations

import asyncio

import pytest

from houyi.adapters.embedding import NoOpEmbeddingProvider
from houyi.adapters.memory import (
    MemoryEngine,
    build_memory_engine,
    build_memory_engine_from_env,
)
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.engine import _candidate_to_memory_recall
from houyi.adapters.memory.recall.factory import _build_default_recall_orchestrator
from houyi.adapters.memory.recall.types import RecallCandidate, RetrieverKind
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    RawTurn,
    RecallMatchMethod,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_turn(turn_id: str = "t1", text: str = "alice likes tea") -> RawTurn:
    return RawTurn(
        turn_id=turn_id,
        namespace="ns",
        session_id="s1",
        role="user",
        content=text,
    )


def _make_fact(
    subject: str = "alice",
    predicate: str = "likes",
    obj: str = "tea",
    anchor: str = "src-1",
) -> AtomicFact:
    return AtomicFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        certainty=Certainty.CERTAIN,
        source_anchor=anchor,
    )


# ---------------------------------------------------------------------------
# build_memory_engine
# ---------------------------------------------------------------------------


class TestBuildMemoryEngine:
    def test_returns_memory_engine(self, tmp_path):
        engine = build_memory_engine(data_dir=tmp_path / "memory")
        try:
            assert isinstance(engine, MemoryEngine)
            assert engine.store is not None
        finally:
            engine.store.close()

    def test_lexical_skips_backfill(self, tmp_path):
        engine = build_memory_engine(data_dir=tmp_path / "memory")
        try:
            # Internal field check is acceptable at the facade layer:
            # lifecycle wiring is part of the contract.
            assert engine._backfill_worker is None
        finally:
            engine.store.close()

    def test_no_llm_skips_extractor(self, tmp_path):
        engine = build_memory_engine(data_dir=tmp_path / "memory")
        try:
            assert engine._extractor_worker is None
        finally:
            engine.store.close()

    def test_embedding_wires_backfill(self, tmp_path):
        engine = build_memory_engine(
            data_dir=tmp_path / "memory",
            embedding_provider=NoOpEmbeddingProvider(),
        )
        try:
            assert engine._backfill_worker is not None
        finally:
            engine.store.close()


class TestBuildMemoryEngineFromEnv:
    def test_lexical_when_no_provider(self, tmp_path, monkeypatch):
        # Force the embedding factory to fail so the env-builder falls
        # through to lexical-only mode. Any non-zero key combination
        # missing the matching API key will do.
        from houyi.infrastructure.config.env_config import EnvConfig

        EnvConfig._reset()
        monkeypatch.setenv("EMBEDDING_PROVIDER", "siliconflow")
        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

        engine = build_memory_engine_from_env(data_dir=tmp_path / "memory")
        try:
            assert isinstance(engine, MemoryEngine)
            assert engine._backfill_worker is None
        finally:
            engine.store.close()

    def test_env_resolves_provider(self, tmp_path, monkeypatch):
        from houyi.infrastructure.config.env_config import EnvConfig

        EnvConfig._reset()
        monkeypatch.setenv("EMBEDDING_PROVIDER", "noop")

        engine = build_memory_engine_from_env(data_dir=tmp_path / "memory")
        try:
            assert engine._backfill_worker is not None
        finally:
            engine.store.close()


# ---------------------------------------------------------------------------
# MemoryEngine facade
# ---------------------------------------------------------------------------


class TestMemoryEngineFacade:
    @pytest.mark.asyncio
    async def test_async_with_lifecycle(self, tmp_path):
        engine = build_memory_engine(
            data_dir=tmp_path / "memory",
            embedding_provider=NoOpEmbeddingProvider(),
        )
        try:
            async with engine:
                # Inside the context, the backfill worker task is running.
                assert len(engine._worker_tasks) == 1
                assert not engine._worker_tasks[0].done()
            # After exit, the task list is drained.
            assert engine._worker_tasks == []
        finally:
            engine.store.close()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, tmp_path):
        engine = build_memory_engine(
            data_dir=tmp_path / "memory",
            embedding_provider=NoOpEmbeddingProvider(),
        )
        try:
            await engine.start()
            await engine.start()  # second call is a no-op
            assert len(engine._worker_tasks) == 1
            await engine.stop()
        finally:
            engine.store.close()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, tmp_path):
        engine = build_memory_engine(data_dir=tmp_path / "memory")
        try:
            # No workers wired -> stop is a pure no-op.
            await engine.stop()
        finally:
            engine.store.close()

    @pytest.mark.asyncio
    async def test_write_turn_persists(self, tmp_path):
        engine = build_memory_engine(data_dir=tmp_path / "memory")
        try:
            result = await engine.write_turn(_make_turn(), schedule_extract=False)
            assert result.turn.turn_index >= 0
        finally:
            engine.store.close()

    @pytest.mark.asyncio
    async def test_no_writer_raises(self, tmp_path):
        # Direct MemoryEngine construction (bypassing the factory) leaves
        # turn_writer unwired; write_turn must surface a clear error.
        from houyi.adapters.memory.store import MemoryStore

        store = MemoryStore(backend=SQLiteMemoryBackend(db_path=":memory:"))
        engine = MemoryEngine(store)
        try:
            with pytest.raises(RuntimeError, match="no TurnWriter"):
                await engine.write_turn(_make_turn())
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_flush_zero_when_idle(self, tmp_path):
        engine = build_memory_engine(data_dir=tmp_path / "memory")
        try:
            counts = await engine.flush(timeout=1.0)
            assert counts == {"extracted": 0, "backfilled": 0}
        finally:
            engine.store.close()

    @pytest.mark.asyncio
    async def test_flush_drains_extract_queue(self, tmp_path):
        engine = build_memory_engine(data_dir=tmp_path / "memory")
        try:
            # Write a turn with schedule_extract=True so the extract queue
            # carries one pending row. No extractor worker is wired (we
            # passed llm_adapter=None), so the row stays pending; flush
            # therefore must time out.
            await engine.write_turn(_make_turn(), schedule_extract=True)
            with pytest.raises(asyncio.TimeoutError):
                await engine.flush(timeout=0.2)
        finally:
            engine.store.close()


# ---------------------------------------------------------------------------
# RecallOrchestrator factory
# ---------------------------------------------------------------------------


class TestRecallOrchestratorFactory:
    def test_requires_backend(self):
        with pytest.raises(ValueError, match="backend is required"):
            _build_default_recall_orchestrator(
                backend=None,  # type: ignore[arg-type]
                entity_state=None,  # type: ignore[arg-type]
            )

    def test_requires_entity_state(self):
        backend = SQLiteMemoryBackend(db_path=":memory:")
        try:
            with pytest.raises(ValueError, match="entity_state is required"):
                _build_default_recall_orchestrator(
                    backend=backend,
                    entity_state=None,  # type: ignore[arg-type]
                )
        finally:
            backend.close()

    def test_lexical_omits_vector(self, tmp_path):
        backend = SQLiteMemoryBackend(data_dir=tmp_path)
        try:
            view = SQLiteEntityStateView(backend)
            orchestrator = _build_default_recall_orchestrator(
                backend=backend,
                entity_state=view,
                embedding_provider=None,
            )
            # Internal field is the only honest contract surface here:
            # the orchestrator does not advertise its retriever set.
            assert "vector" not in orchestrator._retrievers
            assert "entity_state" in orchestrator._retrievers
        finally:
            backend.close()

    def test_embedding_adds_vector(self, tmp_path):
        backend = SQLiteMemoryBackend(data_dir=tmp_path)
        try:
            view = SQLiteEntityStateView(backend)
            orchestrator = _build_default_recall_orchestrator(
                backend=backend,
                entity_state=view,
                embedding_provider=NoOpEmbeddingProvider(),
            )
            assert "vector" in orchestrator._retrievers
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# RecallCandidate -> MemoryRecall adapter
# ---------------------------------------------------------------------------


class TestCandidateToMemoryRecall:
    def test_propagates_score_and_explanation(self):
        fact = _make_fact()
        cand = RecallCandidate(
            fact=fact,
            score=0.42,
            matched_by=RetrieverKind.ENTITY_STATE,
            explanation="entity_state exact match",
        )
        recall = _candidate_to_memory_recall(cand)
        assert recall.score == pytest.approx(0.42)
        assert recall.explanation == "entity_state exact match"
        assert recall.matched_by == RecallMatchMethod.RULE
        assert recall.memory_id.startswith("fact:")

    def test_explanation_uses_triple(self):
        fact = _make_fact(subject="bob", predicate="bought", obj="bread")
        cand = RecallCandidate(
            fact=fact,
            score=0.1,
            matched_by=RetrieverKind.VECTOR,
        )
        recall = _candidate_to_memory_recall(cand)
        # When the orchestrator did not set an explanation, the adapter
        # synthesizes one from the triple.
        assert recall.explanation == "bob bought bread"

    def test_unknown_kind_hybrid(self):
        # All known RetrieverKind values are mapped explicitly inside
        # the engine's lookup table; we exercise the fallback by feeding
        # a candidate whose matched_by does not appear in the table.
        # Since the table is keyed on RetrieverKind itself, every member
        # maps; the fallback only fires under future enum extension.
        # This test guards that behavior by patching the lookup.
        import houyi.adapters.memory.engine as engine_mod

        original = engine_mod._RETRIEVER_KIND_TO_MATCH_METHOD
        engine_mod._RETRIEVER_KIND_TO_MATCH_METHOD = {}
        try:
            cand = RecallCandidate(
                fact=_make_fact(),
                score=0.5,
                matched_by=RetrieverKind.RAW_TURN,
            )
            recall = _candidate_to_memory_recall(cand)
            assert recall.matched_by == RecallMatchMethod.HYBRID
        finally:
            engine_mod._RETRIEVER_KIND_TO_MATCH_METHOD = original

    def test_memory_id_stable(self):
        # The synthesized memory_id is deterministic so logs that
        # de-dup on memory_id behave consistently across recall calls.
        fact = _make_fact()
        cand_a = RecallCandidate(fact=fact, score=0.1, matched_by=RetrieverKind.ENTITY_STATE)
        cand_b = RecallCandidate(fact=fact, score=0.9, matched_by=RetrieverKind.ENTITY_STATE)
        assert _candidate_to_memory_recall(cand_a).memory_id == (
            _candidate_to_memory_recall(cand_b).memory_id
        )
