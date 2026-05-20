"""Command-line entry point for the memory benchmark.

Usage::

 python -m houyi.arena.memory_bench --dataset=fixture
 python -m houyi.arena.memory_bench --dataset=halumem-medium --sample=20

Wires the production components together (real LLM adapter, real
SQLite store) and prints a JSON summary of the resulting metrics. The
script is intentionally read-only with respect to git state — it
writes to --output only when the user requests it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from houyi.arena.memory_bench.cells import (
    CellRunner,
    write_cells_report,
)
from houyi.arena.memory_bench.dataset import (
    load_halumem_medium,
    load_synthetic_fixture,
)
from houyi.arena.memory_bench.judge import LLMMemoryJudge, MemoryJudge, StubMemoryJudge
from houyi.arena.memory_bench.metrics import BenchMetrics
from houyi.arena.memory_bench.runner import (
    Answerer,
    MemoryBenchReport,
    MemoryBenchRunner,
    MemoryReader,
    SubstringAnswerer,
)
from houyi.arena.memory_bench.types import BenchSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchRunConfig:
    extractor_model: str | None
    extractor_retries: int
    extractor_json_mode: bool
    debug_predictions: bool = False
    shared_memory_across_sessions: bool = False


# ---------------------------------------------------------------------------
# Bench wiring helpers
# ---------------------------------------------------------------------------


def _build_ingestor_and_reader(
    namespace: str,
    db_path: Path,
    *,
    extractor_model: str | None,
    extractor_retries: int = 1,
    extractor_json_mode: bool = True,
) -> tuple[Any, MemoryReader]:
    """Construct a real MemoryIngestor plus a matching reader.

    The reader exposes only the snapshot surface the runner needs and
    intentionally does not leak the SQLite handle so the bench loop
    cannot accidentally mutate state outside the ingestor's path.

    extractor_model overrides the default SiliconFlow model used by
    the atomic-fact extractor; this is the main lever for trading
    extraction quality against bench wall-clock (e.g. switching from
    Pro/zai-org/GLM-5.1 to Qwen/Qwen2.5-7B-Instruct cuts a 100-
    session run from ~8h to ~30-60min).
    """
    from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
    from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
    from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
    from houyi.adapters.memory.extractor import AtomicFactExtractor
    from houyi.adapters.memory.ingestor import MemoryIngestor
    from houyi.adapters.memory.resolver import MemoryWriterTools
    from houyi.adapters.memory.retraction import (
        RetractionDetector,
        RetractionOrchestrator,
    )

    backend = SQLiteMemoryBackend(db_path=db_path)
    view = SQLiteEntityStateView(backend)
    inbox = SQLiteCandidateInbox(backend)
    tools = MemoryWriterTools(view, inbox, namespace=namespace)

    llm = _resolve_llm_adapter(model_override=extractor_model)
    logger.info(
        "[bench] extractor LLM=%s model=%s retries=%d json_mode=%s",
        type(llm).__name__,
        getattr(llm, "default_model", "?"),
        extractor_retries,
        extractor_json_mode,
    )
    extractor = AtomicFactExtractor(
        llm,
        max_retries=extractor_retries,
        prefer_json_mode=extractor_json_mode,
    )
    orchestrator = RetractionOrchestrator(RetractionDetector(), tools)
    ingestor = MemoryIngestor(extractor, orchestrator, tools, inbox)

    class _Reader:
        def list_active_memories(self, ns: str) -> list[str]:
            rows = view.get_active(ns, entity="user")
            return [f"{r.attribute}: {r.value}" for r in rows]

    return ingestor, _Reader()


def _resolve_llm_adapter(*, model_override: str | None = None) -> Any:
    """Build an LLM adapter for ingestion or judging.

    When model_override is provided we instantiate the SiliconFlow
    adapter directly with the explicit default_model so the same
    process can run different stages on different models (e.g. fast
    extractor + heavier judge). When the override is absent we fall
    back to the project-default factory so other providers (Vertex,
    OpenAI, …) keep working.
    """
    if model_override:
        from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter

        return SiliconFlowAdapter(default_model=model_override)

    from houyi.adapters.llm.factory import LLMAdapterFactory

    return LLMAdapterFactory.create()


def _build_judge(kind: str, *, model_override: str | None = None) -> MemoryJudge:
    """Resolve the configured judge implementation.

    'auto' picks LLMMemoryJudge when an LLM adapter is reachable (i.e. an
    API key / local model is available), otherwise it falls back to the
    deterministic StubMemoryJudge so smoke tests and CI runs keep working
    without external credentials.
    """
    if kind == "stub":
        return StubMemoryJudge()
    if kind == "llm":
        return _build_llm_judge(model_override=model_override)
    if kind == "auto":
        try:
            return _build_llm_judge(model_override=model_override)
        except Exception as exc:
            logger.warning(
                "[bench] judge auto: LLM adapter unavailable (%s); falling back to stub",
                exc,
            )
            return StubMemoryJudge()
    raise ValueError(f"unknown judge kind: {kind!r}")


def _build_llm_judge(*, model_override: str | None = None) -> LLMMemoryJudge:
    llm = _resolve_llm_adapter(model_override=model_override)
    logger.info(
        "[bench] judge LLM=%s model=%s",
        type(llm).__name__,
        getattr(llm, "default_model", "?"),
    )
    return LLMMemoryJudge(llm)


def _build_answerer(kind: str) -> Answerer:
    if kind == "stub":
        return SubstringAnswerer()
    raise ValueError(f"unknown answerer kind: {kind!r}; LLM-backed answerer not yet wired")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m houyi.arena.memory_bench",
        description="Run the HaluMem-aligned memory benchmark.",
    )
    parser.add_argument(
        "--dataset",
        choices=("fixture", "halumem-medium"),
        default="fixture",
        help="Dataset to evaluate against (default: built-in fixture).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit the number of sessions (HaluMem only).",
    )
    parser.add_argument(
        "--judge",
        choices=("auto", "stub", "llm"),
        default="auto",
        help=(
            "Judge implementation. 'auto' (default) prefers LLMMemoryJudge "
            "and falls back to stub when no LLM adapter is reachable; "
            "'stub' forces deterministic offline mode; 'llm' fails fast "
            "if no adapter is available."
        ),
    )
    parser.add_argument(
        "--answerer",
        choices=("stub",),
        default="stub",
        help="Answer-generation strategy for the QA task.",
    )
    parser.add_argument(
        "--namespace",
        default="bench",
        help="Memory namespace to write into (default: bench).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite path (defaults to a temp file deleted on exit).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the JSON report to this path (default: stdout).",
    )
    parser.add_argument(
        "--extractor-model",
        default=None,
        help=(
            "Override the SiliconFlow model used by the atomic-fact "
            "extractor. Default: project default (resolved from "
            "SILICONFLOW_MODEL / DEEPSEEK_MODEL env). Recommended for "
            "fast bench runs: Qwen/Qwen2.5-7B-Instruct."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Override the SiliconFlow model used by the LLM judge "
            "(only consulted when --judge=llm). Default: project default. "
            "Recommended: Qwen/Qwen2.5-7B-Instruct (binary verdicts do "
            "not need a 32B reasoner)."
        ),
    )
    parser.add_argument(
        "--extractor-retries",
        type=int,
        default=1,
        help="Retry count for invalid extractor JSON before dropping the turn.",
    )
    parser.add_argument(
        "--disable-extractor-json-mode",
        action="store_true",
        help="Disable response_format=json_schema for extractor calls.",
    )
    parser.add_argument(
        "--debug-predictions",
        action="store_true",
        help="Include per-session prediction and QA judgment details in the JSON report.",
    )
    parser.add_argument(
        "--shared-memory-across-sessions",
        action="store_true",
        help="Reuse one memory namespace across all sessions instead of isolating each session.",
    )
    parser.add_argument(
        "--cells",
        action="store_true",
        help=(
            "Run the full cell matrix and emit benchmark/output/memory/{run_id}/"
            "{cells.json,summary.md} instead of a HaluMem session run."
        ),
    )
    parser.add_argument(
        "--report-root",
        default="benchmark/output/memory",
        help="Root directory for cell reports (used with --cells).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id; defaults to a UTC timestamp.",
    )
    return parser.parse_args(argv)


def _load_sessions(args: argparse.Namespace) -> list[BenchSession]:
    if args.dataset == "fixture":
        return load_synthetic_fixture()
    return load_halumem_medium(sample=args.sample)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    db_path = Path(args.db_path) if args.db_path else Path(tempfile.mkstemp(suffix=".db")[1])
    run_config = BenchRunConfig(
        extractor_model=args.extractor_model,
        extractor_retries=args.extractor_retries,
        extractor_json_mode=not args.disable_extractor_json_mode,
        debug_predictions=args.debug_predictions,
        shared_memory_across_sessions=args.shared_memory_across_sessions,
    )
    try:
        judge = _build_judge(args.judge, model_override=args.judge_model)
        answerer = _build_answerer(args.answerer)
        sessions = _load_sessions(args)
        if args.shared_memory_across_sessions:
            ingestor, reader = _build_ingestor_and_reader(
                args.namespace,
                db_path,
                extractor_model=args.extractor_model,
                extractor_retries=args.extractor_retries,
                extractor_json_mode=not args.disable_extractor_json_mode,
            )
            runner = MemoryBenchRunner(
                ingestor,
                reader,
                judge=judge,
                answerer=answerer,
                namespace=args.namespace,
            )
            report = await runner.run(sessions)
            return _report_to_dict(report, run_config=run_config)
        report = await _run_isolated_sessions(
            args,
            sessions=sessions,
            db_path=db_path,
            judge=judge,
            answerer=answerer,
        )
        return _report_to_dict(report, run_config=run_config)
    finally:
        if args.db_path is None and db_path.exists():
            db_path.unlink(missing_ok=True)


async def _run_isolated_sessions(
    args: argparse.Namespace,
    *,
    sessions: list[BenchSession],
    db_path: Path,
    judge: Any,
    answerer: Answerer,
) -> MemoryBenchReport:
    reports = []
    aggregate = BenchMetrics.empty()
    for session in sessions:
        namespace = f"{args.namespace}:{session.session_id}"
        ingestor, reader = _build_ingestor_and_reader(
            namespace,
            db_path,
            extractor_model=args.extractor_model,
            extractor_retries=args.extractor_retries,
            extractor_json_mode=not args.disable_extractor_json_mode,
        )
        runner = MemoryBenchRunner(
            ingestor,
            reader,
            judge=judge,
            answerer=answerer,
            namespace=namespace,
        )
        report = await runner.run([session])
        reports.extend(report.sessions)
        aggregate = aggregate.merge(report.aggregate)
    return MemoryBenchReport(sessions=tuple(reports), aggregate=aggregate)


def _report_to_dict(report: Any, *, run_config: BenchRunConfig | None = None) -> dict[str, Any]:
    agg = report.aggregate
    payload = {
        "sessions_evaluated": len(report.sessions),
        "extraction": {
            "memory_recall": round(agg.extraction.memory_recall, 4),
            "memory_accuracy": round(agg.extraction.memory_accuracy, 4),
            "weighted_memory_recall": round(agg.extraction.weighted_memory_recall, 4),
            "f1": round(agg.extraction.f1, 4),
            "gold_total": agg.extraction.gold_total,
            "predicted_total": agg.extraction.predicted_total,
        },
        "update": {
            "upd_acc": round(agg.update.upd_acc, 4),
            "upd_hall": round(agg.update.upd_hall, 4),
            "upd_omit": round(agg.update.upd_omit, 4),
            "target_total": agg.update.target_total,
        },
        "qa": {
            "qa_acc": round(agg.qa.qa_acc, 4),
            "qa_hall": round(agg.qa.qa_hall, 4),
            "qa_omit": round(agg.qa.qa_omit, 4),
            "total": agg.qa.total,
        },
        "error_propagation": {
            "extraction_error": round(agg.extraction_error, 4),
            "update_error": round(agg.update_error, 4),
        },
    }
    if run_config is not None:
        payload["config"] = {
            "extractor_model": run_config.extractor_model,
            "extractor_retries": run_config.extractor_retries,
            "extractor_json_mode": run_config.extractor_json_mode,
            "shared_memory_across_sessions": run_config.shared_memory_across_sessions,
        }
        if run_config.debug_predictions:
            payload["sessions"] = [_session_debug_to_dict(session) for session in report.sessions]
    return payload


def _verdict_to_dict(verdict: Any) -> dict[str, str]:
    return {"kind": verdict.kind, "reason": verdict.reason}


def _session_debug_to_dict(session: Any) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "metrics": {
            "memory_recall": round(session.metrics.extraction.memory_recall, 4),
            "memory_accuracy": round(session.metrics.extraction.memory_accuracy, 4),
            "predicted_total": session.metrics.extraction.predicted_total,
            "gold_total": session.metrics.extraction.gold_total,
        },
        "active_memories": list(session.active_memories),
        "gold_recall_judgments": [
            {
                "point_id": judgment.point_id,
                "gold": judgment.gold,
                "best_predicted": judgment.best_predicted,
                "verdict": _verdict_to_dict(judgment.verdict),
            }
            for judgment in session.gold_recall_judgments
        ],
        "missing_gold": [
            {
                "point_id": judgment.point_id,
                "gold": judgment.gold,
                "best_predicted": judgment.best_predicted,
                "verdict": _verdict_to_dict(judgment.verdict),
            }
            for judgment in session.gold_recall_judgments
            if judgment.verdict.kind != "correct"
        ],
        "prediction_judgments": [
            {
                "predicted": judgment.predicted,
                "best_gold": judgment.best_gold,
                "verdict": _verdict_to_dict(judgment.verdict),
            }
            for judgment in session.prediction_judgments
        ],
        "wrong_predictions": [
            {
                "predicted": judgment.predicted,
                "best_gold": judgment.best_gold,
                "verdict": _verdict_to_dict(judgment.verdict),
            }
            for judgment in session.prediction_judgments
            if judgment.verdict.kind != "correct"
        ],
        "qa_judgments": [
            {
                "question": judgment.question,
                "gold_answer": judgment.gold_answer,
                "predicted_answer": judgment.predicted_answer,
                "verdict": _verdict_to_dict(judgment.verdict),
            }
            for judgment in session.qa_judgments
        ],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env load")
    args = _parse_args(argv)
    if args.cells:
        return _run_cells(args)
    summary = asyncio.run(_run(args))
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"wrote report to {args.output}")
    else:
        print(payload)
    return 0


def _run_cells(args: argparse.Namespace) -> int:
    runner = CellRunner()
    report = runner.run(run_id=args.run_id)
    out_dir = write_cells_report(report, args.report_root)
    print(
        f"wrote cells report to {out_dir} "
        f"({report.passed}/{report.total} passed, "
        f"{report.pass_rate * 100:.1f}%)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
