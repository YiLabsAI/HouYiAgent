"""Evolution closeout pipeline: emit → daemon → optimize → rollback."""

from __future__ import annotations

from houyi.adapters.memory.event_emitter import MemoryEventEmitter
from houyi.application.evolution import (
    DeterministicEvolutionOptimizer,
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionClient,
    EvolutionDaemon,
    EvolutionEventType,
    InMemoryEvolutionPolicyStore,
    OptimizationRunner,
    SQLiteEvolutionStore,
)


def _seed_baseline(store: SQLiteEvolutionStore) -> EvolutionArtifact:
    baseline = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline recall policy",
    )
    store.set_active(baseline)
    return baseline


def _emit_recall_failures(emitter: MemoryEventEmitter, count: int) -> None:
    for index in range(count):
        emitter.emit(
            EvolutionEventType.RECALL_FAILURE,
            target="recall_orchestrator",
            payload={"query_preview": f"unknown query {index}"},
            metrics={"severity": 0.9},
        )


class TestClosePhase2Pipeline:
    def test_hotpath_to_audit(self, tmp_path) -> None:
        db_path = tmp_path / "evolution.db"
        store = SQLiteEvolutionStore(db_path)
        baseline = _seed_baseline(store)

        client = EvolutionClient(
            event_log=store,
            policy_store=InMemoryEvolutionPolicyStore(),
            artifact_type=EvolutionArtifactType.RECALL_POLICY,
        )
        emitter = MemoryEventEmitter(client=client)

        _emit_recall_failures(emitter, count=5)

        events, _ = store.read_since(0)
        assert len(events) == 5
        assert all(e.event_type == EvolutionEventType.RECALL_FAILURE for e in events)

        daemon = EvolutionDaemon(
            artifact=baseline,
            event_log=store,
            policy_store=store,
            cursor_store=store,
            consumer_name="closeout_e2e",
            optimizer=DeterministicEvolutionOptimizer(),
            audit_log=store,
        )
        daemon.start()
        run_report = daemon.run_until_idle(max_ticks=5, force=True)

        assert run_report.events_consumed == 5
        assert run_report.errors == 0
        audit = store.read_audit(consumer="closeout_e2e")
        assert len(audit) >= 1
        assert any(entry.action == "tick" for entry in audit)

    def test_runner_emits_report(self, tmp_path) -> None:
        db_path = tmp_path / "evolution.db"
        store = SQLiteEvolutionStore(db_path)
        baseline = _seed_baseline(store)

        client = EvolutionClient(
            event_log=store,
            policy_store=InMemoryEvolutionPolicyStore(),
            artifact_type=EvolutionArtifactType.RECALL_POLICY,
        )
        emitter = MemoryEventEmitter(client=client)
        _emit_recall_failures(emitter, count=3)

        events, _ = store.read_since(0)
        from houyi.application.evolution.signals import EvolutionSignalMiner

        signals = EvolutionSignalMiner().mine(events)
        assert signals, "miner should produce at least one signal from RECALL_FAILUREs"

        runner = OptimizationRunner(optimizer_name="closeout_e2e")
        report = runner.run(baseline, signals, run_id="closeout_e2e")

        assert report.signal_count == len(signals)
        assert report.run_id == "closeout_e2e"
        assert report.optimizer == "closeout_e2e"
        assert "baseline" in report.baseline_content

    def test_rollback_dual_audit(self, tmp_path) -> None:
        db_path = tmp_path / "evolution.db"
        store = SQLiteEvolutionStore(db_path)
        v1 = EvolutionArtifact(
            artifact_type=EvolutionArtifactType.RECALL_POLICY,
            content="v1",
        )
        v2 = EvolutionArtifact(
            artifact_type=EvolutionArtifactType.RECALL_POLICY,
            content="v2",
        )
        store.set_active(v1)
        store.set_active(v2)

        from houyi.application.evolution.__main__ import main

        rc = main(
            [
                "rollback",
                "--store-db",
                str(db_path),
                "--artifact-type",
                "recall_policy",
                "--reason",
                "e2e_smoke",
                "--operator",
                "closeout-bot",
            ]
        )
        assert rc == 0
        active_after = store.get_active(EvolutionArtifactType.RECALL_POLICY)
        assert active_after.artifact_id == v1.artifact_id
        events, _ = store.read_since(0)
        rollback_events = [
            e for e in events if e.event_type == EvolutionEventType.ROLLBACK_PERFORMED
        ]
        assert len(rollback_events) == 1
        assert rollback_events[0].payload["from_artifact_id"] == v2.artifact_id
        audit = store.read_audit(consumer="closeout-bot")
        assert any(entry.action == "rollback" for entry in audit)
