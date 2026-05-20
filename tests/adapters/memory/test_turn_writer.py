"""Tests for the synchronous turn writer (fast_path) and extract queue."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import RawTurn


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteMemoryBackend(db_path=tmp_path / "wp.db")
    yield b
    b.close()


@pytest.fixture()
def turn_writer(backend):
    # Empty composite (all([]) == True) bypasses the default
    # min-length / role policy so legacy tests can use short payloads.
    return TurnWriter(backend, extract_trigger=all_of())


def _turn(content: str, *, session: str = "s1", role: str = "user") -> RawTurn:
    return RawTurn(session_id=session, role=role, content=content)


class TestFastPath:
    def test_appends_l0_enqueues_l1(self, backend, turn_writer):
        result = turn_writer.fast_path(_turn("hello"))
        assert result.turn.turn_index == 0
        assert result.queue_id is not None
        # L0 visible; L1 queue has one pending row.
        assert backend.list_raw_turns("default", "s1")[0].content == "hello"
        assert backend.extract_queue_stats() == {"pending": 1}

    def test_skip_extract_only_l0(self, backend, turn_writer):
        result = turn_writer.fast_path(_turn("hi"), schedule_extract=False)
        assert result.queue_id is None
        assert backend.list_raw_turns("default", "s1") != []
        assert backend.extract_queue_stats() == {}

    def test_idempotent_schedule(self, backend, turn_writer):
        # Same turn enqueued twice -> only one queue row.
        result = turn_writer.fast_path(_turn("once"))
        again = turn_writer.schedule_extract(result.turn)
        assert again == result.queue_id
        assert backend.extract_queue_stats() == {"pending": 1}


class _RecordingDetector:
    """Test-only detector that records every turn it sees."""

    def __init__(self, *, raise_for: str | None = None):
        self.seen: list[RawTurn] = []
        self.raise_for = raise_for

    def detect(self, turn):
        if self.raise_for and self.raise_for in turn.content:
            raise RuntimeError("simulated detector failure")
        self.seen.append(turn)


class TestDetectors:
    def test_detector_after_l0(self, backend):
        det = _RecordingDetector()
        tw = TurnWriter(backend, detectors=[det], extract_trigger=all_of())
        result = tw.fast_path(_turn("payload"))
        assert det.seen == [result.turn]
        assert "_RecordingDetector" in result.detectors_fired

    def test_detector_failure_l1(self, backend):
        det = _RecordingDetector(raise_for="bad")
        tw = TurnWriter(backend, detectors=[det], extract_trigger=all_of())
        result = tw.fast_path(_turn("bad input"))
        # L0 row + L1 queue row both persisted despite detector failure.
        assert backend.extract_queue_stats() == {"pending": 1}
        assert result.queue_id is not None
        assert "_RecordingDetector" not in result.detectors_fired


class TestExtractQueue:
    def test_claim_marks_in_progress(self, backend, turn_writer):
        turn_writer.fast_path(_turn("a"))
        turn_writer.fast_path(_turn("b"))
        claimed = backend.claim_extract_jobs(limit=10)
        assert len(claimed) == 2
        stats = backend.extract_queue_stats()
        assert stats == {"in_progress": 2}

    def test_claim_limit_order(self, backend, turn_writer):
        for i in range(5):
            turn_writer.fast_path(_turn(str(i)))
        claimed = backend.claim_extract_jobs(limit=2)
        assert len(claimed) == 2
        # Earliest enqueued first.
        assert [t.content for _, t in claimed] == ["0", "1"]

    def test_claim_namespace_filter(self, backend):
        tw = TurnWriter(backend, extract_trigger=all_of())
        tw.fast_path(RawTurn(namespace="t1", session_id="s", role="u", content="x"))
        tw.fast_path(RawTurn(namespace="t2", session_id="s", role="u", content="y"))
        only_t1 = backend.claim_extract_jobs(limit=10, namespace="t1")
        assert [t.namespace for _, t in only_t1] == ["t1"]
        assert backend.extract_queue_stats() == {"in_progress": 1, "pending": 1}

    def test_mark_done_finalizes(self, backend, turn_writer):
        turn_writer.fast_path(_turn("a"))
        [(qid, _)] = backend.claim_extract_jobs(limit=1)
        backend.mark_extract_done(qid)
        assert backend.extract_queue_stats() == {"done": 1}

    def test_mark_failed_with_retry(self, backend, turn_writer):
        turn_writer.fast_path(_turn("a"))
        [(qid, _)] = backend.claim_extract_jobs(limit=1)
        backend.mark_extract_failed(qid, "boom", retry=True, max_attempts=3)
        # Re-queued for another attempt.
        assert backend.extract_queue_stats() == {"pending": 1}
        # Subsequent claim picks it up again with attempts=2.
        claimed = backend.claim_extract_jobs(limit=1)
        assert len(claimed) == 1

    def test_mark_failed_terminal(self, backend, turn_writer):
        turn_writer.fast_path(_turn("a"))
        [(qid, _)] = backend.claim_extract_jobs(limit=1)
        backend.mark_extract_failed(qid, "boom", retry=False)
        assert backend.extract_queue_stats() == {"failed": 1}

    def test_stale_lease_reclaims(self, backend, turn_writer):
        turn_writer.fast_path(_turn("stuck"))
        # First worker claims and dies.
        backend.claim_extract_jobs(limit=1, lease_seconds=10.0, now=1000.0)
        # Second worker arrives much later; lease has expired.
        recovered = backend.claim_extract_jobs(limit=1, lease_seconds=10.0, now=1100.0)
        assert len(recovered) == 1

    def test_orphan_turn_marked_failed(self, backend):
        tw = TurnWriter(backend)
        # Manually craft a queue row whose turn_id does not exist.
        ghost = RawTurn(session_id="s", role="u", content="ghost")
        backend.append_raw_turn(ghost)
        qid = backend.enqueue_extract(ghost)
        # Drop the underlying turn behind the queue's back.
        backend._conn().execute("DELETE FROM raw_turn_log WHERE turn_id = ?", (ghost.turn_id,))
        backend._conn().commit()
        claimed = backend.claim_extract_jobs(limit=1)
        assert claimed == []
        assert backend.extract_queue_stats() == {"failed": 1}
        del qid  # silence unused-var


class TestTurnWriterConstruction:
    def test_requires_backend(self):
        with pytest.raises(ValueError):
            TurnWriter(None)
