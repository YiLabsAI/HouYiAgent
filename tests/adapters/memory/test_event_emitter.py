"""Tests for the hot-path memory event emitter and its 5 callsite hooks.

Each callsite gets a focused assertion that the right event_type lands
in a recording double when the failure condition fires, and that no
event is published on the healthy path. The recording double mimics the
EvolutionClient.emit_event signature via duck typing so the production
client never has to be constructed in test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.event_emitter import MemoryEventEmitter, disabled_emitter
from houyi.adapters.memory.extractor import AtomicFactExtractor
from houyi.adapters.memory.ingestor import MemoryIngestor
from houyi.adapters.memory.recall.idk_guard import IDKGuard
from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator
from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.router import QueryRouter, RouteDecision
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallQuery,
    RecallReason,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.resolver import MemoryWriterTools
from houyi.adapters.memory.retraction import (
    RetractionDetector,
    RetractionOrchestrator,
    RetractionTarget,
)
from houyi.adapters.memory.types import AtomicFact, Certainty
from houyi.application.evolution.events import EvolutionEvent, EvolutionEventType

# ---------------------------------------------------------------------------
# Recording double
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Stands in for EvolutionClient — only emit_event is exercised."""

    def __init__(self) -> None:
        self.events: list[EvolutionEvent] = []

    def emit_event(self, event: EvolutionEvent) -> None:
        self.events.append(event)


class _RaisingClient:
    """emit_event always raises; used to verify the wrapper swallows it."""

    def emit_event(self, event: EvolutionEvent) -> None:
        raise RuntimeError("backend down")


def _types(events: list[EvolutionEvent]) -> list[EvolutionEventType]:
    return [e.event_type for e in events]


# ---------------------------------------------------------------------------
# MemoryEventEmitter wrapper
# ---------------------------------------------------------------------------


class TestEventEmitter:
    def test_disabled_no_op(self) -> None:
        # disabled_emitter() must accept emit calls without touching anything.
        emitter = disabled_emitter()
        assert emitter.enabled is False
        emitter.emit(
            EvolutionEventType.RECALL_FAILURE,
            target="t",
            payload={"k": "v"},
            metrics={"x": 1.0},
        )
        # Nothing to assert beyond "did not raise".

    def test_enabled_forwards(self) -> None:
        client = _RecordingClient()
        emitter = MemoryEventEmitter(client)
        assert emitter.enabled is True
        emitter.emit(
            EvolutionEventType.IDK_DECISION,
            target="recall_idk_guard",
            payload={"reason": "no_candidates"},
            metrics={"top_score": 0.0},
            namespace="ns1",
        )
        assert len(client.events) == 1
        ev = client.events[0]
        assert ev.event_type is EvolutionEventType.IDK_DECISION
        assert ev.target == "recall_idk_guard"
        assert ev.namespace == "ns1"
        assert ev.payload == {"reason": "no_candidates"}
        assert ev.metrics == {"top_score": 0.0}

    def test_swallows_client_errors(self) -> None:
        # Hot path must never see EvolutionClient exceptions.
        emitter = MemoryEventEmitter(_RaisingClient())
        # Should not raise.
        emitter.emit(EvolutionEventType.RECALL_FAILURE, target="t")


# ---------------------------------------------------------------------------
# IDKGuard hook
# ---------------------------------------------------------------------------


def _candidate(score: float, signals: dict[str, object] | None = None) -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject="user",
            predicate="city",
            object="Beijing",
            certainty=Certainty.CERTAIN,
            source_anchor="s1",
        ),
        score=score,
        matched_by=RetrieverKind.ENTITY_STATE,
        retriever_name="fake",
        signals=signals or {},
    )


class TestIDKGuardEmits:
    def test_no_candidates_emits_idk(self) -> None:
        client = _RecordingClient()
        guard = IDKGuard(emitter=MemoryEventEmitter(client))
        guard.evaluate(query_type=QueryType.FACTUAL_LOOKUP, candidates=[])
        assert _types(client.events) == [EvolutionEventType.IDK_DECISION]
        ev = client.events[0]
        assert ev.payload["reason"] == RecallReason.NO_CANDIDATES.value
        assert ev.metrics["candidate_count"] == 0.0

    def test_low_evidence_emits_idk(self) -> None:
        client = _RecordingClient()
        guard = IDKGuard(emitter=MemoryEventEmitter(client))
        # rerank_score below default 0.5 threshold.
        guard.evaluate(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate(0.4)],
        )
        assert _types(client.events) == [EvolutionEventType.IDK_DECISION]
        assert client.events[0].payload["reason"] == RecallReason.LOW_EVIDENCE.value

    def test_sufficient_no_emit(self) -> None:
        client = _RecordingClient()
        guard = IDKGuard(emitter=MemoryEventEmitter(client))
        # Healthy outcome — guard must stay silent.
        guard.evaluate(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate(0.9)],
        )
        assert client.events == []


# ---------------------------------------------------------------------------
# RetractionOrchestrator hook
# ---------------------------------------------------------------------------


class _StubWriterTools:
    namespace = "ns"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def invalidate_unit(self, entity: str, attribute: str, *, valid_to=None) -> bool:
        self.calls.append((entity, attribute))
        return True


class TestRetractionEmits:
    def test_signal_emits_retraction_fired(self) -> None:
        client = _RecordingClient()
        tools = _StubWriterTools()
        orch = RetractionOrchestrator(
            RetractionDetector(),
            tools,  # type: ignore[arg-type]
            emitter=MemoryEventEmitter(client),
        )
        outcome = orch.process(
            "Actually, that's not right",
            [RetractionTarget("user", "city")],
        )
        assert outcome.signal is not None
        assert _types(client.events) == [EvolutionEventType.RETRACTION_FIRED]
        ev = client.events[0]
        assert ev.namespace == "ns"
        assert ev.metrics["invalidated"] == 1.0

    def test_no_signal_no_emit(self) -> None:
        client = _RecordingClient()
        tools = _StubWriterTools()
        orch = RetractionOrchestrator(
            RetractionDetector(),
            tools,  # type: ignore[arg-type]
            emitter=MemoryEventEmitter(client),
        )
        orch.process("I love Beijing food", [RetractionTarget("user", "city")])
        assert client.events == []


# ---------------------------------------------------------------------------
# MemoryIngestor hook
# ---------------------------------------------------------------------------


@dataclass
class _Resp:
    content: str


class _ScriptedLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def chat(self, messages, *, temperature: float, max_tokens: int):
        return _Resp(self._payload)


def _build_ingestor(tmp_path, payload: str, client: _RecordingClient) -> MemoryIngestor:
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    view = SQLiteEntityStateView(backend)
    inbox = SQLiteCandidateInbox(backend)
    tools = MemoryWriterTools(view, inbox, namespace="ns")
    extractor = AtomicFactExtractor(_ScriptedLLM(payload))
    retraction = RetractionOrchestrator(RetractionDetector(), tools)
    return MemoryIngestor(extractor, retraction, tools, inbox, emitter=MemoryEventEmitter(client))


class TestIngestorEmits:
    @pytest.mark.asyncio
    async def test_vague_only_emits(self, tmp_path) -> None:
        client = _RecordingClient()
        # All vague — no certain fact admitted.
        payload = json.dumps(
            [
                {
                    "subject": "project",
                    "predicate": "status",
                    "object": "stuck",
                    "certainty": "vague",
                }
            ]
        )
        ing = _build_ingestor(tmp_path, payload, client)
        await ing.ingest_turn("kind of stuck", source_anchor="m1")
        assert _types(client.events) == [EvolutionEventType.EXTRACTOR_LOW_CERTAINTY]
        ev = client.events[0]
        assert ev.namespace == "ns"
        # One produced fact, all non-certain.
        assert ev.metrics["facts_total"] == 1.0
        assert ev.metrics["vague_or_probable"] == 1.0

    @pytest.mark.asyncio
    async def test_certain_no_emit(self, tmp_path) -> None:
        client = _RecordingClient()
        payload = json.dumps(
            [
                {
                    "subject": "user",
                    "predicate": "name",
                    "object": "Alice",
                    "certainty": "certain",
                }
            ]
        )
        ing = _build_ingestor(tmp_path, payload, client)
        await ing.ingest_turn("My name is Alice", source_anchor="m1")
        assert client.events == []


# ---------------------------------------------------------------------------
# RecallOrchestrator hook
# ---------------------------------------------------------------------------


class _StaticRouter(QueryRouter):
    """Router that always returns FACTUAL_LOOKUP — bypasses LLM tier."""

    def __init__(self) -> None:
        pass

    async def classify(self, query: RecallQuery) -> RouteDecision:
        return RouteDecision(query_type=QueryType.FACTUAL_LOOKUP, confidence=1.0, source="rule")


class _EmptyRetriever(Retriever):
    name = "entity_state"
    kind = RetrieverKind.ENTITY_STATE

    async def retrieve(self, query: RecallQuery, ctx: RetrieverContext) -> list[RecallCandidate]:
        return []


class TestRecallOrchestratorEmits:
    @pytest.mark.asyncio
    async def test_dry_recall_emits_failure(self) -> None:
        client = _RecordingClient()
        orch = RecallOrchestrator(
            router=_StaticRouter(),
            retrievers={"entity_state": _EmptyRetriever()},
            emitter=MemoryEventEmitter(client),
        )
        result = await orch.recall(RecallQuery(text="who is Alice", namespace="ns1", top_k=3))
        assert result.reason is RecallReason.NO_CANDIDATES
        assert _types(client.events) == [EvolutionEventType.RECALL_FAILURE]
        ev = client.events[0]
        assert ev.namespace == "ns1"
        assert ev.payload["reason"] == RecallReason.NO_CANDIDATES.value


# ---------------------------------------------------------------------------
# MemoryEngine hook
# ---------------------------------------------------------------------------


class _StubStore:
    def all_records(self) -> list:
        return []


class _EmptyMemoryRetriever:
    async def retrieve(self, query, session_context, top_k):
        return []


class TestMemoryEngineEmits:
    @pytest.mark.asyncio
    async def test_empty_recall_emits_failure(self) -> None:
        from houyi.adapters.memory.engine import MemoryEngine

        client = _RecordingClient()
        engine = MemoryEngine(
            _StubStore(),  # type: ignore[arg-type]
            retriever=_EmptyMemoryRetriever(),  # type: ignore[arg-type]
            emitter=MemoryEventEmitter(client),
        )
        result = await engine.recall("anything")
        assert result == []
        assert _types(client.events) == [EvolutionEventType.RECALL_FAILURE]
        assert client.events[0].payload["reason"] == "no_candidates"
