"""OptimizationRunner + before/after report tests."""

from __future__ import annotations

import json

from houyi.application.evolution import (
    SHADOW_VERDICT_HOLD,
    SHADOW_VERDICT_PROMOTE,
    SHADOW_VERDICT_REJECT,
    BeforeAfterReport,
    CandidateEvaluation,
    CandidateVariant,
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionDataset,
    EvolutionSignal,
    OptimizationRunner,
    write_report,
)


def _baseline() -> EvolutionArtifact:
    return EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline content",
    )


def _signals() -> list[EvolutionSignal]:
    return [
        EvolutionSignal(
            signal_type="recall_failure",
            target="recall_policy",
            severity=0.8,
            event_ids=("evt-1",),
        ),
        EvolutionSignal(
            signal_type="recall_failure",
            target="recall_policy",
            severity=0.6,
            event_ids=("evt-2",),
        ),
    ]


class _ScoreByContent:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def evaluate(
        self,
        candidates: list[CandidateVariant],
        dataset: EvolutionDataset,
    ) -> list[CandidateEvaluation]:
        return [
            CandidateEvaluation(
                candidate=candidate,
                score=self.scores.get(candidate.artifact.content, 0.0),
                passed=True,
                reason="scored",
            )
            for candidate in candidates
        ]


class _RewriteOptimizer:
    def __init__(self, new_content: str, score: float) -> None:
        self.new_content = new_content
        self.score = score

    def propose(self, artifact, signals):
        new_artifact = EvolutionArtifact(
            artifact_type=artifact.artifact_type,
            content=self.new_content,
            parent_id=artifact.artifact_id,
        )
        return [CandidateVariant(artifact=new_artifact, score=self.score)]


class _NoCandidateOptimizer:
    def propose(self, artifact, signals):
        return []


class TestOptimizationRunner:
    def test_promote_when_optimizer_wins(self) -> None:
        evaluator = _ScoreByContent({"baseline content": 0.4, "rewritten": 0.7})
        runner = OptimizationRunner(
            optimizer=_RewriteOptimizer("rewritten", score=0.5),
            evaluator=evaluator,
            optimizer_name="test_rewrite",
        )
        report = runner.run(_baseline(), _signals(), run_id="run_promote")
        assert isinstance(report, BeforeAfterReport)
        assert report.optimizer == "test_rewrite"
        assert report.verdict == SHADOW_VERDICT_PROMOTE
        assert report.optimized_content == "rewritten"
        assert report.delta > 0

    def test_hold_when_optimizer_neutral(self) -> None:
        evaluator = _ScoreByContent({"baseline content": 0.5, "neutral": 0.5})
        runner = OptimizationRunner(
            optimizer=_RewriteOptimizer("neutral", score=0.5),
            evaluator=evaluator,
        )
        report = runner.run(_baseline(), _signals(), run_id="run_hold")
        assert report.verdict == SHADOW_VERDICT_HOLD

    def test_reject_when_optimizer_regresses(self) -> None:
        evaluator = _ScoreByContent({"baseline content": 0.8, "worse": 0.3})
        runner = OptimizationRunner(
            optimizer=_RewriteOptimizer("worse", score=0.5),
            evaluator=evaluator,
        )
        report = runner.run(_baseline(), _signals(), run_id="run_reject")
        assert report.verdict == SHADOW_VERDICT_REJECT
        assert report.delta < 0

    def test_empty_signals(self) -> None:
        runner = OptimizationRunner(optimizer_name="empty")
        report = runner.run(_baseline(), [], run_id="run_empty")
        assert report.signal_count == 0
        assert report.reason == "no_signals"

    def test_no_candidate(self) -> None:
        runner = OptimizationRunner(
            optimizer=_NoCandidateOptimizer(),
            optimizer_name="empty_optimizer",
        )
        report = runner.run(_baseline(), _signals(), run_id="run_no_cand")
        assert report.reason == "optimizer_emitted_no_candidates"


class TestReportPersistence:
    def test_markdown_and_json_written(self, tmp_path) -> None:
        evaluator = _ScoreByContent({"baseline content": 0.4, "rewritten": 0.7})
        runner = OptimizationRunner(
            optimizer=_RewriteOptimizer("rewritten", score=0.5),
            evaluator=evaluator,
        )
        report = runner.run(_baseline(), _signals(), run_id="persist")
        out_dir = tmp_path / "evolution" / "persist"
        md_path = write_report(report, out_dir)
        assert md_path.exists()
        text = md_path.read_text(encoding="utf-8")
        assert "Evolution before/after" in text
        assert "rewritten" in text
        json_path = out_dir / "before_after.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["verdict"] == SHADOW_VERDICT_PROMOTE
        assert payload["optimized_content"] == "rewritten"
