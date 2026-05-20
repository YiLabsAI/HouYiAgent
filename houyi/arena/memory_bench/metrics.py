"""HaluMem evaluation metrics.

Three orthogonal scoring surfaces, mirroring the HaluMem paper:

- ExtractionMetrics — Memory Recall (anti-omission), Memory
 Accuracy (anti-fabrication), and the F1 / weighted variants.
- UpdateMetrics — UpdAcc / UpdHall / UpdOmit per the
 paper's update-evaluation rubric.
- QAMetrics — QA-Acc / QA-Hall / QA-Omit on the QA task.

The aggregate BenchMetrics is the value returned by
MemoryBenchRunner; it carries the per-task views plus the
cross-task derived error-propagation indicators discussed in §4 of the
paper (E_ex, E_upd).

All metrics are computed from already-judged outcomes — that is, the
judge has already mapped each prediction to correct / hallucinated /
omitted. The metric module deliberately does not call the judge
itself so it stays pure-Python and trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass


def _safe_div(num: float, denom: float) -> float:
    """Defensive division: zero denominator collapses to zero."""

    if denom <= 0:
        return 0.0
    return num / denom


@dataclass(frozen=True, slots=True)
class ExtractionMetrics:
    """Per-session extraction scores.

    Population definitions (all integers):

    - gold_total : reference memory points for the session
    - predicted_total : memories the SUT emitted
    - recalled : gold points the judge confirmed as covered
    - accurate : predicted points the judge confirmed as
    matching some gold (i.e. not fabricated)
    - weighted_recalled : sum of salience weights for recalled gold;
    divided by total salience to produce
    weighted_memory_recall. Falls back
    to plain recall when salience is uniform.
    - salience_total : sum of salience over all gold points
    """

    gold_total: int
    predicted_total: int
    recalled: int
    accurate: int
    weighted_recalled: float = 0.0
    salience_total: float = 0.0

    @property
    def memory_recall(self) -> float:
        """Anti-omission: recalled / gold_total."""

        return _safe_div(self.recalled, self.gold_total)

    @property
    def memory_accuracy(self) -> float:
        """Anti-fabrication: accurate / predicted_total."""

        return _safe_div(self.accurate, self.predicted_total)

    @property
    def weighted_memory_recall(self) -> float:
        """Weighted recall, useful when gold salience is non-uniform."""

        if self.salience_total <= 0:
            return self.memory_recall
        return _safe_div(self.weighted_recalled, self.salience_total)

    @property
    def f1(self) -> float:
        """Harmonic mean of recall and accuracy (precision)."""

        recall = self.memory_recall
        precision = self.memory_accuracy
        if recall <= 0 or precision <= 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    """Per-session update-task scores.

    HaluMem decomposes updates into three mutually exclusive outcomes:

    - correct : the SUT applied the right old → new change
    - wrong : the SUT applied an update but with the wrong value
    - missed : the SUT failed to apply the update at all

    The three counts always sum to target_total so the metrics
    behave like a mass-conserving partition.
    """

    target_total: int
    correct: int
    wrong: int
    missed: int

    @property
    def upd_acc(self) -> float:
        """correct / target_total — the rate the writer got right."""

        return _safe_div(self.correct, self.target_total)

    @property
    def upd_hall(self) -> float:
        """wrong / target_total — fabricated update value."""

        return _safe_div(self.wrong, self.target_total)

    @property
    def upd_omit(self) -> float:
        """missed / target_total — update silently dropped."""

        return _safe_div(self.missed, self.target_total)


@dataclass(frozen=True, slots=True)
class QAMetrics:
    """Per-session QA scores.

    Directly maps the judge's per-question verdict counts onto the
    accuracy / hallucination / omission triple defined in the paper.
    """

    total: int
    correct: int
    hallucinated: int
    omitted: int

    @property
    def qa_acc(self) -> float:
        return _safe_div(self.correct, self.total)

    @property
    def qa_hall(self) -> float:
        return _safe_div(self.hallucinated, self.total)

    @property
    def qa_omit(self) -> float:
        return _safe_div(self.omitted, self.total)


@dataclass(frozen=True, slots=True)
class BenchMetrics:
    """Aggregated per-session scores plus dataset-level rollups.

    Construction is via merge which folds together the per-task
    counts of multiple sessions; the rollup stays cheap because we only
    track integer counters here, not raw predictions.
    """

    extraction: ExtractionMetrics
    update: UpdateMetrics
    qa: QAMetrics

    @classmethod
    def empty(cls) -> BenchMetrics:
        return cls(
            extraction=ExtractionMetrics(0, 0, 0, 0, 0.0, 0.0),
            update=UpdateMetrics(0, 0, 0, 0),
            qa=QAMetrics(0, 0, 0, 0),
        )

    def merge(self, other: BenchMetrics) -> BenchMetrics:
        """Pure-functional accumulation; returns a new instance."""

        return BenchMetrics(
            extraction=ExtractionMetrics(
                gold_total=self.extraction.gold_total + other.extraction.gold_total,
                predicted_total=self.extraction.predicted_total + other.extraction.predicted_total,
                recalled=self.extraction.recalled + other.extraction.recalled,
                accurate=self.extraction.accurate + other.extraction.accurate,
                weighted_recalled=self.extraction.weighted_recalled
                + other.extraction.weighted_recalled,
                salience_total=self.extraction.salience_total + other.extraction.salience_total,
            ),
            update=UpdateMetrics(
                target_total=self.update.target_total + other.update.target_total,
                correct=self.update.correct + other.update.correct,
                wrong=self.update.wrong + other.update.wrong,
                missed=self.update.missed + other.update.missed,
            ),
            qa=QAMetrics(
                total=self.qa.total + other.qa.total,
                correct=self.qa.correct + other.qa.correct,
                hallucinated=self.qa.hallucinated + other.qa.hallucinated,
                omitted=self.qa.omitted + other.qa.omitted,
            ),
        )

    @property
    def extraction_error(self) -> float:
        """E_ex = 1 - MemAcc per the paper's propagation analysis."""

        return 1.0 - self.extraction.memory_accuracy

    @property
    def update_error(self) -> float:
        """E_upd = UpdHall."""

        return self.update.upd_hall
