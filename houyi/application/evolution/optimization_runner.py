"""Drive a single optimization run and emit a before/after report."""

from __future__ import annotations

from dataclasses import dataclass, field

from houyi.application.evolution.artifacts import EvolutionArtifact
from houyi.application.evolution.before_after import BeforeAfterReport, make_run_id
from houyi.application.evolution.dataset import (
    EvolutionDataset,
    EvolutionDatasetBuilder,
    SignalDatasetBuilder,
)
from houyi.application.evolution.evaluation import (
    EvolutionEvaluator,
    HeuristicEvolutionEvaluator,
)
from houyi.application.evolution.events import EvolutionSignal
from houyi.application.evolution.optimizers import (
    DeterministicEvolutionOptimizer,
    EvolutionOptimizer,
)
from houyi.application.evolution.shadow import (
    SHADOW_VERDICT_REJECT,
    DatasetShadowEvaluator,
    ShadowEvaluator,
)


@dataclass(slots=True)
class OptimizationRunner:
    optimizer: EvolutionOptimizer = field(default_factory=DeterministicEvolutionOptimizer)
    evaluator: EvolutionEvaluator = field(default_factory=HeuristicEvolutionEvaluator)
    dataset_builder: EvolutionDatasetBuilder = field(default_factory=SignalDatasetBuilder)
    shadow_evaluator: ShadowEvaluator | None = None
    optimizer_name: str = "deterministic"

    def __post_init__(self) -> None:
        if self.shadow_evaluator is None:
            self.shadow_evaluator = DatasetShadowEvaluator(self.evaluator)

    def run(
        self,
        baseline: EvolutionArtifact,
        signals: list[EvolutionSignal],
        *,
        run_id: str | None = None,
    ) -> BeforeAfterReport:
        run_id = run_id or make_run_id()
        if not signals:
            return _empty_report(baseline, run_id, self.optimizer_name)
        dataset = self.dataset_builder.build(signals)
        candidates = self.optimizer.propose(baseline, signals)
        if not candidates:
            return _no_candidate_report(baseline, run_id, self.optimizer_name, dataset, signals)
        best = max(candidates, key=lambda candidate: candidate.score)
        assert self.shadow_evaluator is not None
        report = self.shadow_evaluator.compare(baseline, best, dataset)
        return BeforeAfterReport(
            run_id=run_id,
            optimizer=self.optimizer_name,
            artifact_type=baseline.artifact_type.value,
            baseline_content=baseline.content,
            optimized_content=best.artifact.content,
            baseline_score=report.active_score,
            optimized_score=report.shadow_score,
            delta=report.delta,
            sample_size=report.sample_size,
            signal_count=len(signals),
            verdict=report.verdict,
            reason=report.reason,
            metrics=dict(report.metrics),
        )


def _empty_report(
    baseline: EvolutionArtifact,
    run_id: str,
    optimizer_name: str,
) -> BeforeAfterReport:
    return BeforeAfterReport(
        run_id=run_id,
        optimizer=optimizer_name,
        artifact_type=baseline.artifact_type.value,
        baseline_content=baseline.content,
        optimized_content=baseline.content,
        baseline_score=0.0,
        optimized_score=0.0,
        delta=0.0,
        sample_size=0,
        signal_count=0,
        verdict=SHADOW_VERDICT_REJECT,
        reason="no_signals",
    )


def _no_candidate_report(
    baseline: EvolutionArtifact,
    run_id: str,
    optimizer_name: str,
    dataset: EvolutionDataset,
    signals: list[EvolutionSignal],
) -> BeforeAfterReport:
    sample = len(dataset.train) + len(dataset.holdout)
    return BeforeAfterReport(
        run_id=run_id,
        optimizer=optimizer_name,
        artifact_type=baseline.artifact_type.value,
        baseline_content=baseline.content,
        optimized_content=baseline.content,
        baseline_score=0.0,
        optimized_score=0.0,
        delta=0.0,
        sample_size=sample,
        signal_count=len(signals),
        verdict=SHADOW_VERDICT_REJECT,
        reason="optimizer_emitted_no_candidates",
    )


__all__ = ["OptimizationRunner"]
