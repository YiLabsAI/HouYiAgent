from __future__ import annotations

from houyi.application.evolution import (
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionEvent,
    EvolutionEventType,
    SQLiteEvolutionStore,
)


def test_sqlite_events(tmp_path) -> None:
    store = SQLiteEvolutionStore(tmp_path / "evolution.db")
    event = EvolutionEvent(
        EvolutionEventType.RECALL_FAILURE,
        target="recall_policy",
        metrics={"severity": 0.9},
    )

    store.append(event)
    events, cursor = store.read_since(0)

    assert events == [event]
    assert cursor == 1


def test_sqlite_cursor(tmp_path) -> None:
    store = SQLiteEvolutionStore(tmp_path / "evolution.db")

    store.set_cursor("worker", 7)

    assert store.get_cursor("worker") == 7
    assert store.get_cursor("other") == 0


def test_sqlite_policy(tmp_path) -> None:
    store = SQLiteEvolutionStore(tmp_path / "evolution.db")
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline",
    )
    candidate = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="candidate",
        version=2,
        parent_id=artifact.artifact_id,
    )

    store.set_active(artifact)
    store.stage(candidate)
    activated = store.activate(candidate.artifact_id)
    rolled_back = store.rollback(EvolutionArtifactType.RECALL_POLICY)

    assert activated == candidate
    assert store.get_active(EvolutionArtifactType.RECALL_POLICY) == rolled_back
    assert rolled_back == artifact
