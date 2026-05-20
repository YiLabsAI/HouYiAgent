from __future__ import annotations

from houyi.application.evolution import (
    BasicConstraintGate,
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionClient,
    EvolutionDaemon,
    EvolutionEvent,
    EvolutionEventType,
    EvolutionScheduler,
    InMemoryEvolutionEventLog,
    InMemoryEvolutionPolicyStore,
    PromotionLevel,
)


def test_event_log_reads() -> None:
    event_log = InMemoryEvolutionEventLog()
    event = EvolutionEvent(EvolutionEventType.RECALL_FAILURE, target="recall_policy")

    event_log.append(event)
    events, cursor = event_log.read_since(0)

    assert events == [event]
    assert cursor == 1


def test_client_emits() -> None:
    event_log = InMemoryEvolutionEventLog()
    policy_store = InMemoryEvolutionPolicyStore()
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline recall policy",
    )
    policy_store.set_active(artifact)
    client = EvolutionClient(
        event_log,
        policy_store,
        EvolutionArtifactType.RECALL_POLICY,
    )
    event = EvolutionEvent(EvolutionEventType.RECALL_FAILURE, target="recall_policy")

    client.emit_event(event)
    events, cursor = event_log.read_since(0)

    assert events == [event]
    assert cursor == 1
    assert client.get_active_artifact() == artifact


def test_policy_store_stages() -> None:
    policy_store = InMemoryEvolutionPolicyStore()
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="candidate recall policy",
    )

    policy_store.stage(artifact)
    policy_store.set_active(artifact)

    assert policy_store.get_active(EvolutionArtifactType.RECALL_POLICY) == artifact
    assert policy_store.list_staged(EvolutionArtifactType.RECALL_POLICY) == [artifact]


def test_scheduler_interval() -> None:
    scheduler = EvolutionScheduler(interval_seconds=10.0)

    assert scheduler.should_run(now=100.0) is True
    scheduler.mark_run(now=100.0)

    assert scheduler.should_run(now=105.0) is False
    assert scheduler.should_run(now=111.0) is True


def test_daemon_promotes_via_shadow() -> None:
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline recall policy",
    )
    daemon = EvolutionDaemon(artifact)
    daemon.start()
    daemon.emit_event(
        EvolutionEvent(
            EvolutionEventType.RECALL_FAILURE,
            target="recall_policy",
            metrics={"severity": 0.9},
        )
    )

    report = daemon.tick(force=True)

    assert report.skipped is False
    assert report.signals_found == 1
    assert report.candidates_created == 1
    assert report.promotion is not None
    assert report.promotion.level == PromotionLevel.ACTIVE
    assert daemon.artifact.version == 2


def test_daemon_gates() -> None:
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline recall policy",
    )
    daemon = EvolutionDaemon(artifact, constraint_gate=BasicConstraintGate(max_content_size=1))
    daemon.start()
    daemon.emit_event(
        EvolutionEvent(
            EvolutionEventType.RECALL_FAILURE,
            target="recall_policy",
            metrics={"severity": 0.9},
        )
    )

    report = daemon.tick(force=True)

    assert report.promotion is not None
    assert report.promotion.level == PromotionLevel.REJECTED
    assert daemon.artifact.content == "baseline recall policy"


def test_daemon_requires_start() -> None:
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline recall policy",
    )
    daemon = EvolutionDaemon(artifact)

    report = daemon.tick(force=True)

    assert report.skipped is True
    assert report.reason == "not_running"


def test_daemon_skips_noise() -> None:
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline recall policy",
    )
    daemon = EvolutionDaemon(artifact)
    daemon.start()
    daemon.emit_event(EvolutionEvent(EvolutionEventType.RECALL_RESULT, target="recall_policy"))

    report = daemon.tick(force=True)

    assert report.skipped is True
    assert report.reason == "no_signals"
