"""CLI entry point for evolution optimization runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from houyi.application.evolution.artifacts import (
    EvolutionArtifact,
    EvolutionArtifactType,
)
from houyi.application.evolution.audit_log import AuditEntry
from houyi.application.evolution.before_after import (
    BeforeAfterReport,
    make_run_id,
    write_report,
)
from houyi.application.evolution.events import EvolutionEvent, EvolutionEventType
from houyi.application.evolution.optimization_runner import OptimizationRunner
from houyi.application.evolution.optimizers import DeterministicEvolutionOptimizer
from houyi.application.evolution.signals import EvolutionSignalMiner
from houyi.application.evolution.sqlite_providers import SQLiteEvolutionStore

logger = logging.getLogger(__name__)

OPTIMIZER_DETERMINISTIC = "deterministic"
OPTIMIZER_DSPY_GEPA = "dspy_gepa"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m houyi.application.evolution")
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="Run one optimization pass and emit a report.")
    optimize.add_argument(
        "--artifact-type",
        required=True,
        choices=[item.value for item in EvolutionArtifactType],
    )
    optimize.add_argument("--baseline-content", required=True)
    optimize.add_argument(
        "--signal-fixture",
        help="JSON file with a list of EvolutionEvent payloads to mine signals from.",
    )
    optimize.add_argument(
        "--event-log-db",
        help="SQLite event log path; reads recent events via read_since(0).",
    )
    optimize.add_argument(
        "--optimizer",
        choices=(OPTIMIZER_DETERMINISTIC, OPTIMIZER_DSPY_GEPA),
        default=OPTIMIZER_DETERMINISTIC,
    )
    optimize.add_argument(
        "--report-root",
        default="reports/evolution",
    )
    optimize.add_argument("--run-id", default=None)

    history = sub.add_parser("history", help="Show artifact history for a SQLite store.")
    history.add_argument("--store-db", required=True)
    history.add_argument(
        "--artifact-type",
        required=True,
        choices=[item.value for item in EvolutionArtifactType],
    )

    rollback = sub.add_parser("rollback", help="Roll active artifact back one step (LIFO).")
    rollback.add_argument("--store-db", required=True)
    rollback.add_argument(
        "--artifact-type",
        required=True,
        choices=[item.value for item in EvolutionArtifactType],
    )
    rollback.add_argument(
        "--reason",
        default="manual",
        help="Reason recorded in the rollback event/audit row.",
    )
    rollback.add_argument(
        "--operator",
        default="cli",
        help="Operator name recorded as the audit consumer.",
    )

    revert = sub.add_parser("revert", help="Revert active artifact to a specific historical id.")
    revert.add_argument("--store-db", required=True)
    revert.add_argument("--artifact-id", required=True)
    revert.add_argument("--reason", default="manual")
    revert.add_argument("--operator", default="cli")

    return parser.parse_args(argv)


def _load_events_from_fixture(path: Path) -> list[EvolutionEvent]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"signal fixture must be a JSON list, got {type(payload).__name__}")
    events: list[EvolutionEvent] = []
    for raw in payload:
        events.append(
            EvolutionEvent(
                event_type=EvolutionEventType(raw["event_type"]),
                target=str(raw["target"]),
                payload=dict(raw.get("payload", {})),
                metrics={key: float(value) for key, value in dict(raw.get("metrics", {})).items()},
                namespace=str(raw.get("namespace", "default")),
            )
        )
    return events


def _load_events_from_db(path: Path) -> list[EvolutionEvent]:
    store = SQLiteEvolutionStore(path)
    events, _ = store.read_since(0)
    return events


def _build_optimizer(name: str):
    if name == OPTIMIZER_DETERMINISTIC:
        return DeterministicEvolutionOptimizer(), name
    if name == OPTIMIZER_DSPY_GEPA:
        from houyi.application.evolution.dspy_gepa import DspyGepaOptimizer

        return DspyGepaOptimizer(), name
    raise ValueError(f"unknown optimizer: {name!r}")


def _run_optimize(args: argparse.Namespace) -> int:
    if not args.signal_fixture and not args.event_log_db:
        raise SystemExit("must provide --signal-fixture or --event-log-db")
    if args.signal_fixture and args.event_log_db:
        raise SystemExit("provide only one of --signal-fixture / --event-log-db")
    if args.signal_fixture:
        events = _load_events_from_fixture(Path(args.signal_fixture))
    else:
        events = _load_events_from_db(Path(args.event_log_db))

    signals = EvolutionSignalMiner().mine(events)
    baseline = EvolutionArtifact(
        artifact_type=EvolutionArtifactType(args.artifact_type),
        content=args.baseline_content,
    )
    optimizer, name = _build_optimizer(args.optimizer)
    runner = OptimizationRunner(optimizer=optimizer, optimizer_name=name)
    run_id = args.run_id or make_run_id()
    report = runner.run(baseline, signals, run_id=run_id)
    out_dir = Path(args.report_root) / run_id
    md_path = write_report(report, out_dir)
    print(_summary_line(report, md_path))
    return 0


def _summary_line(report: BeforeAfterReport, md_path: Path) -> str:
    return (
        f"verdict={report.verdict} delta={report.delta:+.4f} "
        f"baseline={report.baseline_score:.4f} optimized={report.optimized_score:.4f} "
        f"signals={report.signal_count} report={md_path}"
    )


def _record_rollback(
    store: SQLiteEvolutionStore,
    *,
    operator: str,
    reason: str,
    artifact_type: str,
    from_id: str | None,
    to_id: str,
) -> None:
    payload = {
        "from_artifact_id": from_id or "",
        "to_artifact_id": to_id,
        "reason": reason,
        "operator": operator,
    }
    store.append(
        EvolutionEvent(
            event_type=EvolutionEventType.ROLLBACK_PERFORMED,
            target=artifact_type,
            payload=payload,
        )
    )
    cursor = store.get_cursor(operator)
    store.append_audit(
        AuditEntry(
            consumer=operator,
            action="rollback",
            cursor_before=cursor,
            cursor_after=cursor,
            events_consumed=0,
            skipped=False,
            reason=reason,
            promotion_level=None,
        )
    )


def _run_history(args: argparse.Namespace) -> int:
    store = SQLiteEvolutionStore(Path(args.store_db))
    artifact_type = EvolutionArtifactType(args.artifact_type)
    history = store.list_history(artifact_type)
    active = store.get_active(artifact_type)
    payload = {
        "artifact_type": artifact_type.value,
        "active": _artifact_summary(active),
        "history": [_artifact_summary(item) for item in history],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_rollback(args: argparse.Namespace) -> int:
    store = SQLiteEvolutionStore(Path(args.store_db))
    artifact_type = EvolutionArtifactType(args.artifact_type)
    active_before = store.get_active(artifact_type)
    previous = store.rollback(artifact_type)
    _record_rollback(
        store,
        operator=args.operator,
        reason=args.reason,
        artifact_type=artifact_type.value,
        from_id=active_before.artifact_id if active_before else None,
        to_id=previous.artifact_id,
    )
    print(
        json.dumps(
            {
                "rolled_from": active_before.artifact_id if active_before else None,
                "rolled_to": previous.artifact_id,
                "version": previous.version,
                "reason": args.reason,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _run_revert(args: argparse.Namespace) -> int:
    store = SQLiteEvolutionStore(Path(args.store_db))
    target_row = store.revert_to(args.artifact_id)
    _record_rollback(
        store,
        operator=args.operator,
        reason=args.reason,
        artifact_type=target_row.artifact_type.value,
        from_id=None,
        to_id=target_row.artifact_id,
    )
    print(
        json.dumps(
            {
                "reverted_to": target_row.artifact_id,
                "artifact_type": target_row.artifact_type.value,
                "version": target_row.version,
                "reason": args.reason,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _artifact_summary(artifact: EvolutionArtifact | None) -> dict[str, object] | None:
    if artifact is None:
        return None
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type.value,
        "version": artifact.version,
        "parent_id": artifact.parent_id,
        "content": artifact.content,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args(argv)
    if args.command == "optimize":
        return _run_optimize(args)
    if args.command == "history":
        return _run_history(args)
    if args.command == "rollback":
        return _run_rollback(args)
    if args.command == "revert":
        return _run_revert(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
