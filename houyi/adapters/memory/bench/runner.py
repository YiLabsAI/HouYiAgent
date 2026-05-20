"""Bench runner skeleton — drive cases through judge + collect a report.

 provides the loop only. Concrete ingestion + recall glue is
injected via callbacks so the same runner serves adversarial,
LoCoMo, and HaluMem cases. The end-to-end wiring lives in /
T10; this module just guarantees one place that knows how to score
N cases and emit a JSON report.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from houyi.adapters.memory.answerer import AnswerResult
from houyi.adapters.memory.bench.judge import JudgeVerdict, MemoryJudge

logger = logging.getLogger(__name__)


# A case-runner takes one case and returns the system's answer for it.
# The bench harness assembles a closure over (ingestor, recall,
# answerer) and hands it to BenchRunner.run.
CaseRunner = Callable[[Any], Awaitable[AnswerResult]]


@dataclass(frozen=True)
class BenchOutcome:
    """One case's full record — query, answer, judge verdict, latency."""

    case_id: str
    answer: AnswerResult
    verdict: JudgeVerdict
    duration_s: float
    error: str | None = None
    """Set when the case_runner raised; answer is then a placeholder."""


@dataclass
class BenchReport:
    """Aggregate result of a BenchRunner.run call."""

    outcomes: list[BenchOutcome] = field(default_factory=list)
    total: int = 0
    correct: int = 0
    errors: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)
    """Counter of verdict.reason across outcomes — surfaces the
 distribution of failure modes without needing to grep the trace.
 """

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "errors": self.errors,
            "accuracy": self.accuracy,
            "by_reason": dict(self.by_reason),
        }


class BenchRunner:
    """Drive one case_runner over an iterable of cases."""

    def __init__(
        self,
        judge: MemoryJudge,
        *,
        case_id_attr: str = "id",
    ) -> None:
        if judge is None:
            raise ValueError("judge is required")
        self._judge = judge
        self._case_id_attr = case_id_attr

    async def run(
        self,
        cases: Iterable[Any],
        case_runner: CaseRunner,
        *,
        on_progress: Callable[[BenchOutcome], None] | None = None,
    ) -> BenchReport:
        """Execute case_runner for each case and collect verdicts.

        on_progress is invoked after each outcome — useful for
        streaming logs out of long bench runs without buffering the
        full report.
        """
        report = BenchReport()
        reasons: Counter[str] = Counter()

        for case in cases:
            cid = str(getattr(case, self._case_id_attr, "?"))
            t0 = time.perf_counter()
            err: str | None = None
            try:
                answer = await case_runner(case)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("case_runner failed for %s: %s", cid, exc)
                err = str(exc)[:200]
                # Synthesize an abstain so the judge still runs and
                # produces a stable report shape.
                answer = AnswerResult(
                    answer="",
                    abstained=True,
                    reason="case_runner_failed",
                )

            try:
                verdict = await self._judge.judge(case, answer)
            except Exception as exc:
                logger.warning("judge failed for %s: %s", cid, exc)
                verdict = JudgeVerdict(correct=False, reason="judge_raised", detail=str(exc)[:200])

            outcome = BenchOutcome(
                case_id=cid,
                answer=answer,
                verdict=verdict,
                duration_s=time.perf_counter() - t0,
                error=err,
            )
            report.outcomes.append(outcome)
            report.total += 1
            if verdict.correct:
                report.correct += 1
            if err is not None:
                report.errors += 1
            reasons[verdict.reason] += 1
            if on_progress is not None:
                try:
                    on_progress(outcome)
                except Exception:
                    logger.warning("on_progress hook failed", exc_info=True)

        report.by_reason = dict(reasons)
        return report


__all__ = ["BenchOutcome", "BenchReport", "BenchRunner", "CaseRunner"]
