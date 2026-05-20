"""Rollback / replay tests for policy stores + CLI."""

from __future__ import annotations

import json

import pytest

from houyi.application.evolution import (
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionEventType,
    InMemoryEvolutionPolicyStore,
    SQLiteEvolutionStore,
)
from houyi.application.evolution.__main__ import main


def _artifact(content: str) -> EvolutionArtifact:
    return EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content=content,
    )


class TestInMemoryHistory:
    def test_history_after_promotions(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        v1 = _artifact("v1")
        v2 = _artifact("v2")
        v3 = _artifact("v3")
        store.set_active(v1)
        store.set_active(v2)
        store.set_active(v3)
        history = store.list_history(EvolutionArtifactType.RECALL_POLICY)
        assert [item.artifact_id for item in history] == [v1.artifact_id, v2.artifact_id]
        assert store.get_active(EvolutionArtifactType.RECALL_POLICY).artifact_id == v3.artifact_id

    def test_revert_specific_id(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        v1 = _artifact("v1")
        v2 = _artifact("v2")
        v3 = _artifact("v3")
        store.set_active(v1)
        store.set_active(v2)
        store.set_active(v3)
        reverted = store.revert_to(v1.artifact_id)
        assert reverted.artifact_id == v1.artifact_id
        active = store.get_active(EvolutionArtifactType.RECALL_POLICY)
        assert active.artifact_id == v1.artifact_id
        history_ids = [
            item.artifact_id for item in store.list_history(EvolutionArtifactType.RECALL_POLICY)
        ]
        assert v1.artifact_id not in history_ids
        assert v3.artifact_id in history_ids

    def test_revert_unknown_id_raises(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        store.set_active(_artifact("v1"))
        with pytest.raises(LookupError):
            store.revert_to("does-not-exist")


class TestSqliteHistory:
    def test_list_history_and_revert(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        v1 = _artifact("v1")
        v2 = _artifact("v2")
        v3 = _artifact("v3")
        store.set_active(v1)
        store.set_active(v2)
        store.set_active(v3)
        history = store.list_history(EvolutionArtifactType.RECALL_POLICY)
        history_ids = {item.artifact_id for item in history}
        assert {v1.artifact_id, v2.artifact_id} <= history_ids
        reverted = store.revert_to(v1.artifact_id)
        assert reverted.artifact_id == v1.artifact_id
        active = store.get_active(EvolutionArtifactType.RECALL_POLICY)
        assert active.artifact_id == v1.artifact_id


class TestRollbackCli:
    def _seed(self, db_path) -> tuple[EvolutionArtifact, EvolutionArtifact, EvolutionArtifact]:
        store = SQLiteEvolutionStore(db_path)
        v1 = _artifact("v1")
        v2 = _artifact("v2")
        v3 = _artifact("v3")
        store.set_active(v1)
        store.set_active(v2)
        store.set_active(v3)
        return v1, v2, v3

    def test_history_cmd(self, tmp_path, capsys) -> None:
        db_path = tmp_path / "evolution.db"
        v1, v2, v3 = self._seed(db_path)
        rc = main(
            [
                "history",
                "--store-db",
                str(db_path),
                "--artifact-type",
                "recall_policy",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["active"]["artifact_id"] == v3.artifact_id
        history_ids = {item["artifact_id"] for item in payload["history"]}
        assert {v1.artifact_id, v2.artifact_id} <= history_ids

    def test_rollback_writes_audit(self, tmp_path, capsys) -> None:
        db_path = tmp_path / "evolution.db"
        v1, v2, v3 = self._seed(db_path)
        rc = main(
            [
                "rollback",
                "--store-db",
                str(db_path),
                "--artifact-type",
                "recall_policy",
                "--reason",
                "regression_detected",
                "--operator",
                "ops-bot",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["rolled_from"] == v3.artifact_id
        assert payload["rolled_to"] == v2.artifact_id
        store = SQLiteEvolutionStore(db_path)
        assert store.get_active(EvolutionArtifactType.RECALL_POLICY).artifact_id == v2.artifact_id
        events, _ = store.read_since(0)
        rollback_events = [
            e for e in events if e.event_type == EvolutionEventType.ROLLBACK_PERFORMED
        ]
        assert len(rollback_events) == 1
        assert rollback_events[0].payload["from_artifact_id"] == v3.artifact_id
        assert rollback_events[0].payload["to_artifact_id"] == v2.artifact_id
        audit = store.read_audit(consumer="ops-bot")
        assert any(
            entry.action == "rollback" and entry.reason == "regression_detected" for entry in audit
        )

    def test_revert_to_arbitrary_id(self, tmp_path, capsys) -> None:
        db_path = tmp_path / "evolution.db"
        v1, v2, v3 = self._seed(db_path)
        rc = main(
            [
                "revert",
                "--store-db",
                str(db_path),
                "--artifact-id",
                v1.artifact_id,
                "--reason",
                "manual_pin",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reverted_to"] == v1.artifact_id
        store = SQLiteEvolutionStore(db_path)
        assert store.get_active(EvolutionArtifactType.RECALL_POLICY).artifact_id == v1.artifact_id
