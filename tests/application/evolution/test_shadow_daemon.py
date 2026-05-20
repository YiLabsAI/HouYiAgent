"""Daemon shadow gating tests."""

from __future__ import annotations

from houyi.application.evolution import (
    SHADOW_VERDICT_HOLD,
    SHADOW_VERDICT_PROMOTE,
    SHADOW_VERDICT_REJECT,
    CandidateVariant,
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionDaemon,
    EvolutionEvent,
    EvolutionEventType,
    InMemoryEvolutionPolicyStore,
    PromotionLevel,
    ShadowReport,
)


def _baseline() -> EvolutionArtifact:
    return EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline",
    )


def _failure_event() -> EvolutionEvent:
    return EvolutionEvent(
        EvolutionEventType.RECALL_FAILURE,
        target="recall_policy",
        metrics={"severity": 0.9},
    )


class _CandidateOptimizer:
    def __init__(self, content: str, score: float = 0.9) -> None:
        self.content = content
        self.score = score

    def propose(self, artifact: EvolutionArtifact, signals) -> list[CandidateVariant]:
        candidate_artifact = EvolutionArtifact(
            artifact_type=artifact.artifact_type,
            content=self.content,
            parent_id=artifact.artifact_id,
        )
        return [CandidateVariant(artifact=candidate_artifact, score=self.score)]


class _StubShadow:
    def __init__(self, verdict: str, *, active: float = 0.5, shadow: float = 0.7) -> None:
        self.verdict = verdict
        self.active = active
        self.shadow = shadow

    def compare(self, active, candidate, dataset):
        return ShadowReport(
            active_score=self.active,
            shadow_score=self.shadow,
            delta=self.shadow - self.active,
            sample_size=2,
            holdout_size=1,
            verdict=self.verdict,
            reason=f"stub_{self.verdict}",
            metrics={"active_score": self.active, "shadow_score": self.shadow},
        )


class TestDaemonShadowGate:
    def test_promote_flips_active(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        daemon = EvolutionDaemon(
            _baseline(),
            policy_store=store,
            optimizer=_CandidateOptimizer("better"),
            shadow_evaluator=_StubShadow(SHADOW_VERDICT_PROMOTE),
        )
        daemon.emit_event(_failure_event())
        daemon.start()
        report = daemon.tick(force=True)
        assert report.promotion is not None
        assert report.promotion.level == PromotionLevel.ACTIVE
        assert daemon.artifact.content == "better"
        active = store.get_active(EvolutionArtifactType.RECALL_POLICY)
        assert active is not None and active.content == "better"

    def test_hold_writes_shadow(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        daemon = EvolutionDaemon(
            _baseline(),
            policy_store=store,
            optimizer=_CandidateOptimizer("uncertain"),
            shadow_evaluator=_StubShadow(SHADOW_VERDICT_HOLD, shadow=0.52),
        )
        daemon.emit_event(_failure_event())
        daemon.start()
        report = daemon.tick(force=True)
        assert report.promotion is not None
        assert report.promotion.level == PromotionLevel.SHADOW
        assert daemon.artifact.content == "baseline"
        assert store.get_active(EvolutionArtifactType.RECALL_POLICY).content == "baseline"
        held = store.get_shadow(EvolutionArtifactType.RECALL_POLICY)
        assert held is not None and held.content == "uncertain"

    def test_reject_keeps_active(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        daemon = EvolutionDaemon(
            _baseline(),
            policy_store=store,
            optimizer=_CandidateOptimizer("regression"),
            shadow_evaluator=_StubShadow(SHADOW_VERDICT_REJECT, shadow=0.2),
        )
        daemon.emit_event(_failure_event())
        daemon.start()
        report = daemon.tick(force=True)
        assert report.promotion is not None
        assert report.promotion.level == PromotionLevel.REJECTED
        assert daemon.artifact.content == "baseline"
        assert store.get_shadow(EvolutionArtifactType.RECALL_POLICY) is None
