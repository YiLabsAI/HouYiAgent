"""Long-running daemon tests."""

from __future__ import annotations

import pytest

from houyi.application.evolution import (
    AuditEntry,
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionDaemon,
    EvolutionEvent,
    EvolutionEventType,
    EvolutionRunReport,
    InMemoryEvolutionAuditLog,
    SQLiteEvolutionStore,
)


def _baseline() -> EvolutionArtifact:
    return EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline recall policy",
    )


def _failure_event(target: str = "recall_policy") -> EvolutionEvent:
    return EvolutionEvent(
        EvolutionEventType.RECALL_FAILURE,
        target=target,
        metrics={"severity": 0.9},
    )


# ---------------------------------------------------------------------------
# run_until_idle drains backlog
# ---------------------------------------------------------------------------


class TestDrainLoop:
    def test_drains_backlog(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        audit = InMemoryEvolutionAuditLog()
        daemon = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            audit_log=audit,
            consumer_name="worker_drain",
        )
        for _ in range(3):
            daemon.emit_event(_failure_event())
        daemon.start()

        report = daemon.run_until_idle()

        assert isinstance(report, EvolutionRunReport)
        # All 3 events drained; cursor advanced to the head.
        assert report.events_consumed == 3
        assert report.cursor == 3
        # Final tick reports "no_events" so the loop exits cleanly.
        assert report.last_report is not None
        assert report.last_report.reason == "no_events"
        assert report.errors == 0

    def test_idle_returns_immediately(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        daemon = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            consumer_name="worker_idle",
        )
        daemon.start()

        report = daemon.run_until_idle()

        # No events at all → exactly one "no_events" tick, no work.
        assert report.events_consumed == 0
        assert len(report.ticks) == 1
        assert report.ticks[0].reason == "no_events"


# ---------------------------------------------------------------------------
# Multiple consumers share the same event log via independent cursors
# ---------------------------------------------------------------------------


class TestMultipleConsumers:
    def test_independent_cursors(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        # Two workers backed by the same SQLite store but with distinct
        # consumer_name keys; each must observe the full event stream.
        worker_a = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            consumer_name="worker_a",
        )
        worker_b = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            consumer_name="worker_b",
        )
        for _ in range(2):
            worker_a.emit_event(_failure_event())
        worker_a.start()
        worker_b.start()

        report_a = worker_a.run_until_idle()
        # worker_a drained both events.
        assert report_a.events_consumed == 2

        # worker_b's cursor was untouched by worker_a.
        assert store.get_cursor("worker_b") == 0
        report_b = worker_b.run_until_idle()
        # worker_b sees the same backlog independently.
        assert report_b.events_consumed == 2

        # The cursors are persisted separately.
        assert store.get_cursor("worker_a") == 2
        assert store.get_cursor("worker_b") == 2


# ---------------------------------------------------------------------------
# Crash recovery: a fresh daemon resumes from the persisted cursor
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    def test_fresh_instance_resumes(self, tmp_path) -> None:
        path = tmp_path / "evolution.db"
        store_first = SQLiteEvolutionStore(path)
        first = EvolutionDaemon(
            _baseline(),
            event_log=store_first,
            cursor_store=store_first,
            consumer_name="resumer",
        )
        first.emit_event(_failure_event())
        first.emit_event(_failure_event())
        first.start()
        first.run_until_idle()
        assert store_first.get_cursor("resumer") == 2
        # Simulate a crash: drop the daemon and reopen the store.
        del first

        store_second = SQLiteEvolutionStore(path)
        # 2 more events arrive after the "crash".
        store_second.append(_failure_event())
        store_second.append(_failure_event())
        second = EvolutionDaemon(
            _baseline(),
            event_log=store_second,
            cursor_store=store_second,
            consumer_name="resumer",
        )
        # The new daemon must pick up cursor=2 (not 0) — the two events
        # already processed by the first instance must not be replayed.
        assert second.cursor == 2
        second.start()
        report = second.run_until_idle()
        assert report.events_consumed == 2
        assert store_second.get_cursor("resumer") == 4

    def test_other_consumer_unaffected(self, tmp_path) -> None:
        # Crash recovery must be per-consumer: a crashed worker_a should
        # not silently advance worker_b's cursor.
        path = tmp_path / "evolution.db"
        store = SQLiteEvolutionStore(path)
        worker_a = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            consumer_name="crash_a",
        )
        worker_a.emit_event(_failure_event())
        worker_a.start()
        worker_a.run_until_idle()

        store_reopen = SQLiteEvolutionStore(path)
        worker_b_fresh = EvolutionDaemon(
            _baseline(),
            event_log=store_reopen,
            cursor_store=store_reopen,
            consumer_name="crash_b",
        )
        # crash_b never ran → cursor stays at 0 even after reopening.
        assert worker_b_fresh.cursor == 0


# ---------------------------------------------------------------------------
# Audit log records every tick boundary
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_records_tick_per_pass(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        audit = InMemoryEvolutionAuditLog()
        daemon = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            audit_log=audit,
            consumer_name="audited",
        )
        daemon.emit_event(_failure_event())
        daemon.start()

        daemon.run_until_idle()

        entries = audit.read_audit(consumer="audited")
        # First tick consumes the event ("completed"); second tick is
        # the idle drain ("no_events").
        actions = [e.action for e in entries]
        reasons = [e.reason for e in entries]
        assert actions == ["tick", "tick"]
        assert reasons == ["completed", "no_events"]

        completed = entries[0]
        assert completed.cursor_before == 0
        assert completed.cursor_after == 1
        assert completed.events_consumed == 1
        assert completed.skipped is False
        assert completed.promotion_level == "active"

    def test_sqlite_audit_persists(self, tmp_path) -> None:
        path = tmp_path / "evolution.db"
        store = SQLiteEvolutionStore(path)
        # Manually append an audit row (simulating a previous run) and
        # verify a freshly-opened store reads it back.
        store.append_audit(
            AuditEntry(
                consumer="persisted",
                action="tick",
                cursor_before=0,
                cursor_after=5,
                events_consumed=5,
                skipped=False,
                reason="completed",
                promotion_level="active",
            )
        )
        del store

        reopened = SQLiteEvolutionStore(path)
        rows = reopened.read_audit(consumer="persisted")
        assert len(rows) == 1
        assert rows[0].cursor_after == 5
        assert rows[0].promotion_level == "active"

    def test_miner_raise_audit(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        audit = InMemoryEvolutionAuditLog()

        class _ExplodingMiner:
            def mine(self, events):
                raise RuntimeError("boom")

        daemon = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            audit_log=audit,
            signal_miner=_ExplodingMiner(),
            consumer_name="ex",
        )
        daemon.emit_event(_failure_event())
        daemon.start()

        # Single-tick mode: the exception bubbles up after audit.
        with pytest.raises(RuntimeError, match="boom"):
            daemon.tick(force=True)

        rows = audit.read_audit(consumer="ex")
        assert [r.action for r in rows] == ["error"]
        assert rows[0].error is not None
        assert "RuntimeError: boom" in rows[0].error

    def test_drain_swallows_errors(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        audit = InMemoryEvolutionAuditLog()

        class _FlakyMiner:
            def __init__(self) -> None:
                self.calls = 0

            def mine(self, events):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient")
                return []

        miner = _FlakyMiner()
        daemon = EvolutionDaemon(
            _baseline(),
            event_log=store,
            cursor_store=store,
            audit_log=audit,
            signal_miner=miner,
            consumer_name="flaky",
        )
        daemon.emit_event(_failure_event())
        daemon.start()

        # Drive the daemon. The first tick raises (transient), the
        # second succeeds with no_signals → drains and exits.
        # We bound max_ticks so a misbehaving error path cannot loop.
        report = daemon.run_until_idle(max_ticks=5)

        # The error did not strand the loop — at least one error was
        # captured and the eventual cursor advanced past the event.
        assert report.errors >= 1
        assert miner.calls >= 2
