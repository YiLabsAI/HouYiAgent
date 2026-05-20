"""Tests for the async L1 extractor worker that drains the extraction queue."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import AtomicFact, Certainty, RawTurn
from houyi.adapters.memory.workers import ExtractorWorker, ExtractorWorkerConfig


@dataclass
class _FakeResult:
    facts: list = None
    raw_sourceless: list = None
    invalid_dropped: int = 0


class _FakeExtractor:
    """Returns canned ExtractionResult-like objects per call."""

    def __init__(self, results=None, raise_for: str | None = None):
        self.results = list(results or [])
        self.calls: list[tuple[str, str | None]] = []
        self.raise_for = raise_for

    async def extract(self, text, source_anchor):
        self.calls.append((text, source_anchor))
        if self.raise_for and self.raise_for in text:
            raise RuntimeError("simulated LLM outage")
        if not self.results:
            return _FakeResult(facts=[], raw_sourceless=[])
        return self.results.pop(0)


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteMemoryBackend(db_path=tmp_path / "ext.db")
    yield b
    b.close()


@pytest.fixture()
def state_view(backend):
    return SQLiteEntityStateView(backend)


@pytest.fixture()
def inbox(backend):
    return SQLiteCandidateInbox(backend)


def _fact(
    subject="alice", predicate="likes", obj="tea", *, certainty=Certainty.CERTAIN, anchor="anchor"
):
    return AtomicFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        certainty=certainty,
        source_anchor=anchor,
    )


def _make_tw(backend) -> TurnWriter:
    """Permissive turn-writer that bypasses the default L1 trigger."""
    return TurnWriter(backend, extract_trigger=all_of())


def _enqueue(tw: TurnWriter, content: str, *, session: str = "s") -> RawTurn:
    return tw.fast_path(RawTurn(session_id=session, role="user", content=content)).turn


class TestProcessOnce:
    async def test_empty_queue_returns_zero(self, backend, state_view, inbox):
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(),
            entity_state=state_view,
            candidate_inbox=inbox,
        )
        assert await worker.process_once() == 0

    async def test_certain_fact_entity_state(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        turn = _enqueue(wp, "alice likes tea")
        fact = _fact(anchor=turn.turn_id)
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(results=[_FakeResult(facts=[fact])]),
            entity_state=state_view,
            candidate_inbox=inbox,
        )
        processed = await worker.process_once()
        assert processed == 1
        rows = state_view.get_active("default", "alice")
        assert len(rows) == 1
        assert rows[0].value == "tea"
        assert backend.extract_queue_stats() == {"done": 1}

    async def test_vague_fact_to_inbox(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        turn = _enqueue(wp, "maybe alice likes tea")
        vague = _fact(certainty=Certainty.VAGUE, anchor=turn.turn_id)
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(results=[_FakeResult(facts=[vague])]),
            entity_state=state_view,
            candidate_inbox=inbox,
        )
        await worker.process_once()
        # No entity_state row; inbox holds the vague candidate.
        assert state_view.get_active("default", "alice") == []
        assert len(inbox.list_for("default")) == 1
        assert backend.extract_queue_stats() == {"done": 1}

    async def test_sourceless_routes_to_inbox(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        _enqueue(wp, "noisy line")
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(results=[_FakeResult(raw_sourceless=[{"foo": "bar"}])]),
            entity_state=state_view,
            candidate_inbox=inbox,
        )
        await worker.process_once()
        assert inbox.list_sourceless("default") == [{"foo": "bar"}]
        assert backend.extract_queue_stats() == {"done": 1}


class TestFailureHandling:
    async def test_extractor_failure_retries(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        _enqueue(wp, "boom payload")
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(raise_for="boom"),
            entity_state=state_view,
            candidate_inbox=inbox,
            config=ExtractorWorkerConfig(max_attempts=2),
        )
        # First failure → re-queued.
        await worker.process_once()
        assert backend.extract_queue_stats() == {"pending": 1}
        # Second failure → terminal.
        await worker.process_once()
        assert backend.extract_queue_stats() == {"failed": 1}

    async def test_projection_failure_marks_failed(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        turn = _enqueue(wp, "x")

        class _BoomState:
            def upsert(self, *a, **kw):
                raise RuntimeError("upsert exploded")

        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(results=[_FakeResult(facts=[_fact(anchor=turn.turn_id)])]),
            entity_state=_BoomState(),
            candidate_inbox=inbox,
            config=ExtractorWorkerConfig(max_attempts=1),
        )
        await worker.process_once()
        assert backend.extract_queue_stats() == {"failed": 1}


class TestRunForever:
    async def test_stop_event_terminates_loop(self, backend, state_view, inbox):
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(),
            entity_state=state_view,
            candidate_inbox=inbox,
            config=ExtractorWorkerConfig(idle_sleep_s=0.05),
        )
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run_forever(stop))
        await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

    async def test_processes_queue_until_empty(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        # Three jobs.
        for i in range(3):
            t = _enqueue(wp, f"msg{i}")
            del t
        # Worker drains all three across batches.
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(),  # empty results = no facts
            entity_state=state_view,
            candidate_inbox=inbox,
            config=ExtractorWorkerConfig(batch_size=2),
        )
        await worker.process_once()
        await worker.process_once()
        assert backend.extract_queue_stats() == {"done": 3}


class TestConstruction:
    def test_requires_dependencies(self, backend, state_view, inbox):
        with pytest.raises(ValueError):
            ExtractorWorker(
                backend=None,
                extractor=_FakeExtractor(),
                entity_state=state_view,
                candidate_inbox=inbox,
            )
            with pytest.raises(ValueError):
                ExtractorWorker(
                    backend=backend,
                    extractor=None,
                    entity_state=state_view,
                    candidate_inbox=inbox,
                )
                with pytest.raises(ValueError):
                    ExtractorWorker(
                        backend=backend,
                        extractor=_FakeExtractor(),
                        entity_state=None,
                        candidate_inbox=inbox,
                    )
