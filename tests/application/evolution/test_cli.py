"""CLI tests for python -m houyi.application.evolution."""

from __future__ import annotations

import json

import pytest

from houyi.application.evolution import EvolutionEventType, SQLiteEvolutionStore
from houyi.application.evolution.__main__ import main
from houyi.application.evolution.events import EvolutionEvent


def _fixture_payload() -> list[dict]:
    return [
        {
            "event_type": EvolutionEventType.RECALL_FAILURE.value,
            "target": "recall_policy",
            "metrics": {"severity": 0.9},
            "namespace": "default",
            "payload": {},
        },
        {
            "event_type": EvolutionEventType.RECALL_FAILURE.value,
            "target": "recall_policy",
            "metrics": {"severity": 0.7},
            "namespace": "default",
            "payload": {},
        },
    ]


class TestOptimizeCli:
    def test_fixture_end_to_end(self, tmp_path, capsys) -> None:
        fixture = tmp_path / "signals.json"
        fixture.write_text(json.dumps(_fixture_payload()), encoding="utf-8")
        report_root = tmp_path / "reports"
        rc = main(
            [
                "optimize",
                "--artifact-type",
                "recall_policy",
                "--baseline-content",
                "baseline recall policy",
                "--signal-fixture",
                str(fixture),
                "--report-root",
                str(report_root),
                "--run-id",
                "cli_fixture",
            ]
        )
        assert rc == 0
        md_path = report_root / "cli_fixture" / "before_after.md"
        json_path = report_root / "cli_fixture" / "before_after.json"
        assert md_path.exists() and json_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == "cli_fixture"
        assert payload["signal_count"] == 2
        captured = capsys.readouterr().out
        assert "verdict=" in captured

    def test_event_log_db(self, tmp_path) -> None:
        db_path = tmp_path / "evolution.db"
        store = SQLiteEvolutionStore(db_path)
        store.append(
            EvolutionEvent(
                event_type=EvolutionEventType.RECALL_FAILURE,
                target="recall_policy",
                metrics={"severity": 0.9},
            )
        )
        report_root = tmp_path / "reports"
        rc = main(
            [
                "optimize",
                "--artifact-type",
                "recall_policy",
                "--baseline-content",
                "baseline",
                "--event-log-db",
                str(db_path),
                "--report-root",
                str(report_root),
                "--run-id",
                "cli_db",
            ]
        )
        assert rc == 0
        payload = json.loads(
            (report_root / "cli_db" / "before_after.json").read_text(encoding="utf-8")
        )
        assert payload["signal_count"] == 1

    def test_missing_signal_source(self) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "optimize",
                    "--artifact-type",
                    "recall_policy",
                    "--baseline-content",
                    "baseline",
                ]
            )

    def test_dspy_gepa_unavailable(self, tmp_path) -> None:
        fixture = tmp_path / "signals.json"
        fixture.write_text(json.dumps(_fixture_payload()), encoding="utf-8")
        from houyi.application.evolution.dspy_gepa import DspyGepaUnavailableError

        with pytest.raises((DspyGepaUnavailableError, ImportError)):
            main(
                [
                    "optimize",
                    "--artifact-type",
                    "recall_policy",
                    "--baseline-content",
                    "baseline",
                    "--signal-fixture",
                    str(fixture),
                    "--optimizer",
                    "dspy_gepa",
                    "--report-root",
                    str(tmp_path / "reports"),
                ]
            )
