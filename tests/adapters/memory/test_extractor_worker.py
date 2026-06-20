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
from houyi.adapters.memory.types import AtomicFact, Certainty, MemoryEvent, RawTurn
from houyi.adapters.memory.workers import ExtractorWorker, ExtractorWorkerConfig


@dataclass
class _FakeResult:
    facts: list = None
    events: list = None
    edges: list = None
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


class _BatchFakeExtractor(_FakeExtractor):
    def __init__(self, results=None):
        super().__init__(results=results)
        self.batch_calls: list[list[tuple[str, str | None]]] = []

    async def extract_batch(self, turns: list[tuple[str, str | None]], namespace: str = "default"):
        self.batch_calls.append(list(turns))
        if not self.results:
            return [_FakeResult(facts=[], raw_sourceless=[]) for _ in turns]
        out = []
        for _ in turns:
            if self.results:
                out.append(self.results.pop(0))
            else:
                out.append(_FakeResult(facts=[], raw_sourceless=[]))
        return out


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

    async def test_verb_echo_fact_dropped(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        turn = _enqueue(wp, "Evan started painting a forest scene")
        echo = _fact(
            subject="Evan", predicate="started_painting", obj="painting", anchor=turn.turn_id
        )
        real = _fact(subject="Evan", predicate="painted", obj="forest scene", anchor=turn.turn_id)
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(results=[_FakeResult(facts=[echo, real])]),
            entity_state=state_view,
            candidate_inbox=inbox,
        )
        processed = await worker.process_once()
        assert processed == 1
        rows = state_view.get_active("default", "Evan")
        values = {r.value for r in rows}
        # The verb-echo 'painting' must be dropped; the concrete fact kept.
        assert "forest scene" in values
        assert "painting" not in values

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

    async def test_batch_extractor(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        t1 = _enqueue(wp, "alice likes tea")
        t2 = _enqueue(wp, "alice likes coffee")
        batch = _BatchFakeExtractor(
            results=[
                _FakeResult(facts=[_fact(obj="tea", anchor=t1.turn_id)]),
                _FakeResult(facts=[_fact(obj="coffee", anchor=t2.turn_id)]),
            ]
        )
        worker = ExtractorWorker(
            backend=backend,
            extractor=batch,
            entity_state=state_view,
            candidate_inbox=inbox,
            config=ExtractorWorkerConfig(batch_size=8),
        )
        processed = await worker.process_once()
        assert processed == 2
        assert len(batch.batch_calls) == 1
        payload = batch.batch_calls[0]
        assert [anchor for _, anchor in payload] == [t1.turn_id, t2.turn_id]
        rows = state_view.get_active("default", "alice", "likes")
        assert len(rows) == 2
        assert {r.value for r in rows} == {"tea", "coffee"}
        assert backend.extract_queue_stats() == {"done": 2}

    async def test_auto_derive_edges(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        turn = _enqueue(wp, "Caroline works as a Teacher. Joanna knows Bob.")

        extractor = _FakeExtractor(
            results=[
                _FakeResult(
                    facts=[
                        _fact(
                            subject="Caroline", predicate="job", obj="Teacher", anchor=turn.turn_id
                        ),
                        _fact(subject="Joanna", predicate="knows", obj="Bob", anchor=turn.turn_id),
                    ],
                    edges=[],
                )
            ]
        )
        worker = ExtractorWorker(
            backend=backend,
            extractor=extractor,
            entity_state=state_view,
            candidate_inbox=inbox,
        )
        processed = await worker.process_once()
        assert processed == 1

        # Verify state records were inserted
        c_rows = state_view.get_active("default", "Caroline", "job")
        j_rows = state_view.get_active("default", "Joanna", "knows")
        assert len(c_rows) == 1
        assert len(j_rows) == 1

        # Verify auto-derived edges (knows -> RELATED_TO, plus identity anchors)
        conn = backend._conn()
        edges = conn.execute("SELECT * FROM memory_edges").fetchall()
        edge_relations = {dict(e)["relation"] for e in edges}
        assert "related_to" in edge_relations

    async def test_event_wiring(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        turn = _enqueue(wp, "Joanna watched Eternal Sunshine in 2019")

        event = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor=turn.turn_id,
        )
        extractor = _FakeExtractor(
            results=[
                _FakeResult(
                    facts=[
                        _fact(
                            subject="Joanna",
                            predicate="watched",
                            obj="Eternal Sunshine",
                            anchor=turn.turn_id,
                        ),
                    ],
                    events=[event],
                    edges=[],
                )
            ]
        )
        worker = ExtractorWorker(
            backend=backend,
            extractor=extractor,
            entity_state=state_view,
            candidate_inbox=inbox,
            event_view=backend,
        )
        processed = await worker.process_once()
        assert processed == 1

        # Verify event stored
        stored_event = backend.get_event(event.event_id)
        assert stored_event is not None
        assert stored_event.action == "watched"
        assert stored_event.timestamp == "2019"

        # Verify PARTICIPATES_IN edge (state -> event)
        conn = backend._conn()
        edges = conn.execute("SELECT * FROM memory_edges").fetchall()
        participates_edges = [dict(e) for e in edges if dict(e)["relation"] == "participates_in"]
        assert len(participates_edges) >= 1
        pe = participates_edges[0]
        assert pe["source_type"] == "state"
        assert pe["target_type"] == "event"
        assert pe["target_unit_id"] == event.event_id


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


class TestWriteLock:
    async def test_lock_preserves_projection(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        turn = _enqueue(wp, "alice likes tea")
        fact = _fact(anchor=turn.turn_id)
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractor(results=[_FakeResult(facts=[fact])]),
            entity_state=state_view,
            candidate_inbox=inbox,
            write_lock=asyncio.Lock(),
        )
        assert await worker.process_once() == 1
        rows = state_view.get_active("default", "alice")
        assert len(rows) == 1
        assert rows[0].value == "tea"
        assert backend.extract_queue_stats() == {"done": 1}

    async def test_concurrent_drain_safe(self, backend, state_view, inbox):
        wp = _make_tw(backend)
        results = []
        for i in range(12):
            turn = _enqueue(wp, f"person{i} likes item{i}")
            results.append(
                _FakeResult(
                    facts=[_fact(subject=f"person{i}", obj=f"item{i}", anchor=turn.turn_id)]
                )
            )
        worker = ExtractorWorker(
            backend=backend,
            extractor=_BatchFakeExtractor(results=results),
            entity_state=state_view,
            candidate_inbox=inbox,
            config=ExtractorWorkerConfig(batch_size=3),
            write_lock=asyncio.Lock(),
        )

        async def drain() -> None:
            while await worker.process_once() > 0:
                pass

        await asyncio.gather(*[drain() for _ in range(4)])
        stats = backend.extract_queue_stats()
        assert stats.get("done") == 12
        assert "failed" not in stats


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
