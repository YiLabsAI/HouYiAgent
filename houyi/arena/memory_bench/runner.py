"""Orchestrates one HaluMem-style benchmark run end-to-end.

The runner is intentionally agnostic of judge / answerer / dataset
sources: each is injected. Smoke tests pass stubs; the CLI binds the
real LLM-backed implementations. Per-session work is:

1. Ingest each user turn through MemoryIngestor.
2. Score the extraction task by reading active entity-state rows
 for the session's namespace and asking the judge whether each gold
 memory was covered (and whether each predicted memory was a
 fabrication).
3. Score the update task by checking, for each gold (old → new)
 pair, whether the active row now matches new.
4. Score the QA task by calling Answerer with the question
 + the available memories, then asking the judge to grade the
 generated answer against the gold reference.

The result rolls up to a MemoryBenchReport containing the
aggregate BenchMetrics plus per-session detail rows so failure
analysis can pick out specific cells without re-running the bench.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from houyi.arena.memory_bench.judge import JudgeVerdict, MemoryJudge, StubMemoryJudge
from houyi.arena.memory_bench.metrics import (
    BenchMetrics,
    ExtractionMetrics,
    QAMetrics,
    UpdateMetrics,
)
from houyi.arena.memory_bench.timing import (
    PATH_KIND_SYNC_INLINE,
    BenchTimings,
    SessionTimingSamples,
)
from houyi.arena.memory_bench.types import BenchSession, MemoryPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pluggable surfaces
# ---------------------------------------------------------------------------


class Answerer(Protocol):
    """Generate an answer given a question and the available memories.

    Implementations vary from a trivial substring lookup (smoke) to
    "build a prompt and call the LLM" (real evaluation). Returning an
    empty string is interpreted as a refusal / IDK and leads to a QA
    omission verdict.
    """

    def __call__(self, question: str, memories: list[str]) -> str: ...


class IngestorLike(Protocol):
    """Protocol matching MemoryIngestor.ingest_turn.

    Stated as a structural type to avoid hard-coupling the bench runner
    to the concrete implementation; tests can pass any object exposing
    the same async surface.
    """

    async def ingest_turn(
        self,
        text: str,
        *,
        source_anchor: str | None,
        recent_targets: Any = (),
    ) -> Any: ...


class MemoryReader(Protocol):
    """Read currently-active memories for a given session namespace.

    The runner needs a uniform way to ask "what does the SUT now
    believe about this user?" without hard-coding entity/attribute
    semantics. Implementations typically wrap an
    EntityStateView and return one string per active row.
    """

    def list_active_memories(self, namespace: str) -> list[str]: ...


# ---------------------------------------------------------------------------
# Built-in default Answerer
# ---------------------------------------------------------------------------


class SubstringAnswerer:
    """Returns the first memory string containing a question keyword.

    Trivially deterministic. Used by the smoke test so the entire QA
    plumbing exercises without an LLM. Real evaluation should replace
    this with an LLM-backed answerer (typically reusing the same
    chat model used as the judge, but with a generation prompt).
    """

    def __call__(self, question: str, memories: list[str]) -> str:
        question_tokens = {tok.lower().strip("?.,!") for tok in question.split() if len(tok) >= 3}
        for memory in memories:
            mem_l = memory.lower()
            if any(tok in mem_l for tok in question_tokens):
                return memory
        return ""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldRecallJudgment:
    point_id: str
    gold: str
    best_predicted: str | None
    verdict: JudgeVerdict


@dataclass(frozen=True, slots=True)
class PredictionJudgment:
    predicted: str
    best_gold: str | None
    verdict: JudgeVerdict


@dataclass(frozen=True, slots=True)
class QAJudgment:
    question: str
    gold_answer: str
    predicted_answer: str
    verdict: JudgeVerdict


@dataclass(frozen=True, slots=True)
class SessionReport:
    """Per-session detail (kept verbose for failure analysis)."""

    session_id: str
    metrics: BenchMetrics
    active_memories: tuple[str, ...] = ()
    extraction_verdicts: tuple[JudgeVerdict, ...] = ()
    gold_recall_judgments: tuple[GoldRecallJudgment, ...] = ()
    prediction_judgments: tuple[PredictionJudgment, ...] = ()
    update_verdicts: tuple[JudgeVerdict, ...] = ()
    qa_verdicts: tuple[JudgeVerdict, ...] = ()
    qa_judgments: tuple[QAJudgment, ...] = ()
    timings: BenchTimings = field(default_factory=BenchTimings.empty)
    timing_samples: SessionTimingSamples | None = None


@dataclass(frozen=True, slots=True)
class MemoryBenchReport:
    """Aggregate result of a benchmark run."""

    sessions: tuple[SessionReport, ...] = ()
    aggregate: BenchMetrics = field(default_factory=BenchMetrics.empty)
    timings: BenchTimings = field(default_factory=BenchTimings.empty)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MemoryBenchRunner:
    """Drive one or more BenchSession through the SUT and score."""

    def __init__(
        self,
        ingestor: IngestorLike,
        reader: MemoryReader,
        *,
        judge: MemoryJudge | None = None,
        answerer: Answerer | None = None,
        namespace: str = "bench",
        path_kind: str = PATH_KIND_SYNC_INLINE,
        drain_callback: Callable[[], Awaitable[None]] | None = None,
        cost_probe: Callable[[], int] | None = None,
    ) -> None:
        """Construct the bench runner.

        Args:
            ingestor: object satisfying IngestorLike.
            reader: snapshots active memories per session.
            judge / answerer / namespace: see existing semantics.
            path_kind: label that flows through to the BenchTimings of every
                report. Use PATH_KIND_SYNC_INLINE for the legacy
                MemoryIngestor (extraction LLM runs inside ingest_turn) and
                PATH_KIND_TIERED_ASYNC for the TurnWriter + ExtractorWorker
                pair where ingest_turn returns after L0 enqueue.
            drain_callback: optional async hook invoked after the ingest
                stage and before active-memory readout. The tiered path
                points this at ExtractorWorker drainage so its work is
                counted into `timings.drain_total_ms` rather than masked
                across the next session boundary.
            cost_probe: optional callable returning a cumulative
                "extractor calls" counter. The runner snapshots it before
                and after each session and records the delta into
                `timings.extractor_calls` as a cheap cost proxy.
        """
        self._ingestor = ingestor
        self._reader = reader
        self._judge = judge or StubMemoryJudge()
        self._answerer = answerer or SubstringAnswerer()
        self._namespace = namespace
        self._path_kind = path_kind
        self._drain_callback = drain_callback
        self._cost_probe = cost_probe

    async def run(self, sessions: list[BenchSession]) -> MemoryBenchReport:
        """Run all sessions and return the aggregate report.

        Emits one INFO log per session start/finish with cumulative
        wall-clock so long evaluations don't look hung. The line shape
        is stable: "[bench] session N/M ..." so downstream tooling
        can grep for progress.
        """

        per_session: list[SessionReport] = []
        aggregate = BenchMetrics.empty()
        aggregate_timings = BenchTimings.empty(self._path_kind)
        total = len(sessions)
        run_started = time.perf_counter()
        for idx, session in enumerate(sessions, start=1):
            user_turns = sum(1 for t in session.dialogue if t.role == "user")
            logger.info(
                "[bench] session %d/%d start id=%s user_turns=%d gold_mems=%d qa=%d",
                idx,
                total,
                session.session_id,
                user_turns,
                len(session.gold_memories),
                len(session.qa_items),
            )
            sess_started = time.perf_counter()
            report = await self._run_one(session)
            per_session.append(report)
            aggregate = aggregate.merge(report.metrics)
            aggregate_timings = aggregate_timings.merge(report.timings)
            elapsed = time.perf_counter() - sess_started
            cumulative = time.perf_counter() - run_started
            logger.info(
                "[bench] session %d/%d done in %.1fs (cum %.1fs) "
                "recall=%.2f acc=%.2f upd_acc=%.2f qa_acc=%.2f",
                idx,
                total,
                elapsed,
                cumulative,
                report.metrics.extraction.memory_recall,
                report.metrics.extraction.memory_accuracy,
                report.metrics.update.upd_acc,
                report.metrics.qa.qa_acc,
            )

        return MemoryBenchReport(
            sessions=tuple(per_session),
            aggregate=aggregate,
            timings=aggregate_timings,
        )

    # ------------------------------------------------------------------
    # Per-session work
    # ------------------------------------------------------------------

    async def _run_one(self, session: BenchSession) -> SessionReport:
        # Stage 1: feed user turns into the ingestor. Each turn is one
        # extractor LLM call on the sync_inline path; on tiered_async it
        # is just an L0 + enqueue. The runner times each call so the
        # comparison report can show p50/p95/max ingest latency per path.
        cost_before = self._cost_probe() if self._cost_probe is not None else 0
        ingest_samples_ms: list[float] = []
        for idx, turn in enumerate(session.dialogue):
            if turn.role != "user":
                continue
            anchor = f"{session.session_id}:turn-{idx}"
            turn_started = time.perf_counter()
            try:
                await self._ingestor.ingest_turn(
                    turn.content,
                    source_anchor=anchor,
                    recent_targets=(),
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("ingest_turn crashed on session=%s", session.session_id)
                continue
            ingest_ms = (time.perf_counter() - turn_started) * 1000.0
            ingest_samples_ms.append(ingest_ms)
            logger.debug(
                "[bench] turn-%d ingested in %.2fms (%d chars)",
                idx,
                ingest_ms,
                len(turn.content),
            )

        # Stage 2: drain async work (no-op for sync_inline).
        # Runs after ingest so its wall-clock is attributed to the path's
        # background cost rather than crossing into the next session.
        drain_total_ms = 0.0
        if self._drain_callback is not None:
            drain_started = time.perf_counter()
            try:
                await self._drain_callback()
            except Exception:  # pragma: no cover - defensive
                logger.exception("drain_callback crashed on session=%s", session.session_id)
            drain_total_ms = (time.perf_counter() - drain_started) * 1000.0

        cost_after = self._cost_probe() if self._cost_probe is not None else 0
        extractor_calls = max(0, cost_after - cost_before)

        # Stage 3: snapshot the resulting memory state.
        active_memories = self._reader.list_active_memories(self._namespace)

        extraction_metrics, ext_verdicts, gold_judgments, pred_judgments = self._score_extraction(
            session.gold_memories, active_memories
        )
        update_metrics, upd_verdicts = self._score_updates(session, active_memories)
        qa_metrics, qa_verdicts, qa_judgments = self._score_qa(session, active_memories)

        per_session_metrics = BenchMetrics(
            extraction=extraction_metrics,
            update=update_metrics,
            qa=qa_metrics,
        )
        timings = BenchTimings.from_samples(
            self._path_kind,
            ingest_samples_ms,
            drain_total_ms=drain_total_ms,
            extractor_calls=extractor_calls,
        )
        timing_samples = SessionTimingSamples(
            path_kind=self._path_kind,
            ingest_ms=tuple(ingest_samples_ms),
            drain_total_ms=drain_total_ms,
            extractor_calls=extractor_calls,
        )
        return SessionReport(
            session_id=session.session_id,
            metrics=per_session_metrics,
            active_memories=tuple(active_memories),
            extraction_verdicts=ext_verdicts,
            gold_recall_judgments=gold_judgments,
            prediction_judgments=pred_judgments,
            update_verdicts=upd_verdicts,
            qa_verdicts=qa_verdicts,
            qa_judgments=qa_judgments,
            timings=timings,
            timing_samples=timing_samples,
        )

        # ------------------------------------------------------------------
        # Scoring helpers
        # ------------------------------------------------------------------

    def _score_extraction(
        self,
        gold: tuple[MemoryPoint, ...],
        predicted: list[str],
    ) -> tuple[
        ExtractionMetrics,
        tuple[JudgeVerdict, ...],
        tuple[GoldRecallJudgment, ...],
        tuple[PredictionJudgment, ...],
    ]:
        recalled = 0
        weighted_recalled = 0.0
        salience_total = sum(point.salience for point in gold)
        verdicts: list[JudgeVerdict] = []
        gold_judgments: list[GoldRecallJudgment] = []
        for point in gold:
            best_predicted, verdict = _best_recall_detail(self._judge, point.text, predicted)
            verdicts.append(verdict)
            gold_judgments.append(
                GoldRecallJudgment(point.point_id, point.text, best_predicted, verdict)
            )
            if verdict.kind == "correct":
                recalled += 1
                weighted_recalled += point.salience

        gold_texts = [point.text for point in gold]
        accurate = 0
        prediction_judgments: list[PredictionJudgment] = []
        for pred in predicted:
            if not gold_texts:
                prediction_judgments.append(
                    PredictionJudgment(pred, None, JudgeVerdict("wrong", "no gold memories"))
                )
                continue
            best_gold, verdict = _best_match_detail(self._judge, pred, gold_texts)
            prediction_judgments.append(PredictionJudgment(pred, best_gold, verdict))
            if verdict.kind == "correct":
                accurate += 1

        metrics = ExtractionMetrics(
            gold_total=len(gold),
            predicted_total=len(predicted),
            recalled=recalled,
            accurate=accurate,
            weighted_recalled=weighted_recalled,
            salience_total=salience_total,
        )
        return (
            metrics,
            tuple(verdicts),
            tuple(gold_judgments),
            tuple(prediction_judgments),
        )

    def _score_updates(
        self,
        session: BenchSession,
        predicted: list[str],
    ) -> tuple[UpdateMetrics, tuple[JudgeVerdict, ...]]:
        if not session.gold_updates:
            return UpdateMetrics(0, 0, 0, 0), ()

        gold_old_by_id = {point.point_id: point for point in session.gold_memories}
        verdicts: list[JudgeVerdict] = []
        correct = wrong = missed = 0
        for upd in session.gold_updates:
            old_point = gold_old_by_id.get(upd.old_point_id)
            old_text = old_point.text if old_point else ""
            verdict = _best_update_verdict(
                self._judge,
                old_text=old_text,
                new_gold=upd.new_text,
                predicted=predicted,
            )
            verdicts.append(verdict)
            if verdict.kind == "correct":
                correct += 1
            elif verdict.kind == "wrong":
                wrong += 1
        else:
            missed += 1

            metrics = UpdateMetrics(
                target_total=len(session.gold_updates),
                correct=correct,
                wrong=wrong,
                missed=missed,
            )
            return metrics, tuple(verdicts)

    def _score_qa(
        self,
        session: BenchSession,
        memories: list[str],
    ) -> tuple[QAMetrics, tuple[JudgeVerdict, ...], tuple[QAJudgment, ...]]:
        verdicts: list[JudgeVerdict] = []
        qa_judgments: list[QAJudgment] = []
        correct = halluc = omit = 0
        for qa in session.qa_items:
            generated = self._answerer(qa.question, memories) or ""
            verdict = self._judge.judge_qa(qa.question, qa.answer, generated)
            verdicts.append(verdict)
            qa_judgments.append(QAJudgment(qa.question, qa.answer, generated, verdict))
            if verdict.kind == "correct":
                correct += 1
            elif verdict.kind == "wrong":
                halluc += 1
        else:
            omit += 1
            metrics = QAMetrics(
                total=len(session.qa_items),
                correct=correct,
                hallucinated=halluc,
                omitted=omit,
            )
            return metrics, tuple(verdicts), tuple(qa_judgments)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _best_match_verdict(
    judge: MemoryJudge,
    gold_text: str,
    candidates: list[str],
) -> JudgeVerdict:
    """Return the strongest verdict across candidates for one gold.

    "correct" beats "wrong" beats "missing"; this lets the runner
    short-circuit on the first positive match while still preserving
    the worst-case verdict if nothing matches.
    """
    if not candidates:
        return JudgeVerdict("missing", "no candidate memories available")

    fallback = JudgeVerdict("missing", "no candidate matched")
    saw_wrong = False
    last_wrong: JudgeVerdict | None = None
    for candidate in candidates:
        verdict = judge.judge_extraction(gold_text, candidate)
        if verdict.kind == "correct":
            return verdict
        if verdict.kind == "wrong":
            saw_wrong = True
            last_wrong = verdict
            if saw_wrong and last_wrong is not None:
                return last_wrong
            return fallback


def _best_recall_detail(
    judge: MemoryJudge,
    gold_text: str,
    candidates: list[str],
) -> tuple[str | None, JudgeVerdict]:
    if not candidates:
        return None, JudgeVerdict("missing", "no candidate memories available")

    fallback = JudgeVerdict("missing", "no candidate matched")
    last_wrong: tuple[str, JudgeVerdict] | None = None
    for candidate in candidates:
        verdict = judge.judge_extraction(gold_text, candidate)
        if verdict.kind == "correct":
            return candidate, verdict
        if verdict.kind == "wrong":
            last_wrong = (candidate, verdict)
            if last_wrong is not None:
                return last_wrong
            return None, fallback


def _best_match_detail(
    judge: MemoryJudge,
    predicted_text: str,
    gold_texts: list[str],
) -> tuple[str | None, JudgeVerdict]:
    if not gold_texts:
        return None, JudgeVerdict("missing", "no gold memories available")

    last_wrong: tuple[str, JudgeVerdict] | None = None
    for gold_text in gold_texts:
        verdict = judge.judge_extraction(gold_text, predicted_text)
        if verdict.kind == "correct":
            return gold_text, verdict
        if verdict.kind == "wrong":
            last_wrong = (gold_text, verdict)
            if last_wrong is not None:
                return last_wrong
            return None, JudgeVerdict("missing", "no gold matched prediction")


def _best_update_verdict(
    judge: MemoryJudge,
    *,
    old_text: str,
    new_gold: str,
    predicted: list[str],
) -> JudgeVerdict:
    """Best-of judge across predicted memories for one update target."""

    if not predicted:
        return JudgeVerdict("missing", "no candidate memories available")

    last_wrong: JudgeVerdict | None = None
    for candidate in predicted:
        verdict = judge.judge_update(old_text, new_gold, candidate)
        if verdict.kind == "correct":
            return verdict
        if verdict.kind == "wrong":
            last_wrong = verdict
            if last_wrong is not None:
                return last_wrong
            return JudgeVerdict("missing", "no successor recorded")
