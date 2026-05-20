"""Shadow evaluator + stage_shadow + SQLite shadow slot tests."""

from __future__ import annotations

from houyi.application.evolution import (
    SHADOW_VERDICT_HOLD,
    SHADOW_VERDICT_PROMOTE,
    SHADOW_VERDICT_REJECT,
    CandidateEvaluation,
    CandidateVariant,
    DatasetShadowEvaluator,
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionDataset,
    EvolutionExample,
    InMemoryEvolutionPolicyStore,
    PromotionLevel,
    PromotionManager,
    ShadowReport,
    SQLiteEvolutionStore,
)


def _artifact(content: str) -> EvolutionArtifact:
    return EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content=content,
    )


def _dataset() -> EvolutionDataset:
    examples = (
        EvolutionExample(
            task_input="q1",
            expected_behavior="recall_failure",
            category="recall_failure",
            source="evolution_signal",
        ),
        EvolutionExample(
            task_input="q2",
            expected_behavior="recall_failure",
            category="recall_failure",
            source="evolution_signal",
        ),
    )
    return EvolutionDataset(
        train=examples,
        validation=examples[:1],
        holdout=examples[1:],
    )


class _ScoringEvaluator:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []

    def evaluate(
        self,
        candidates: list[CandidateVariant],
        dataset: EvolutionDataset,
    ) -> list[CandidateEvaluation]:
        contents = [candidate.artifact.content for candidate in candidates]
        self.calls.append(contents)
        return [
            CandidateEvaluation(
                candidate=candidate,
                score=self.scores.get(candidate.artifact.content, 0.0),
                passed=True,
                reason="scoring_evaluator",
            )
            for candidate in candidates
        ]


# ---------------------------------------------------------------------------
# DatasetShadowEvaluator verdicts
# ---------------------------------------------------------------------------


class TestShadowVerdict:
    def test_promote_clears_margin(self) -> None:
        evaluator = _ScoringEvaluator({"active": 0.5, "shadow": 0.7})
        shadow_eval = DatasetShadowEvaluator(evaluator, min_delta=0.1)
        report = shadow_eval.compare(
            _artifact("active"),
            CandidateVariant(artifact=_artifact("shadow"), score=0.0),
            _dataset(),
        )
        assert report.verdict == SHADOW_VERDICT_PROMOTE
        assert report.delta == report.shadow_score - report.active_score
        assert report.metrics["shadow_score"] == 0.7

    def test_hold_below_margin(self) -> None:
        evaluator = _ScoringEvaluator({"active": 0.5, "shadow": 0.55})
        shadow_eval = DatasetShadowEvaluator(evaluator, min_delta=0.1)
        report = shadow_eval.compare(
            _artifact("active"),
            CandidateVariant(artifact=_artifact("shadow"), score=0.0),
            _dataset(),
        )
        assert report.verdict == SHADOW_VERDICT_HOLD
        assert report.delta < shadow_eval.min_delta

    def test_reject_on_regression(self) -> None:
        evaluator = _ScoringEvaluator({"active": 0.7, "shadow": 0.5})
        shadow_eval = DatasetShadowEvaluator(
            evaluator,
            min_delta=0.1,
            regression_tolerance=0.05,
        )
        report = shadow_eval.compare(
            _artifact("active"),
            CandidateVariant(artifact=_artifact("shadow"), score=0.0),
            _dataset(),
        )
        assert report.verdict == SHADOW_VERDICT_REJECT
        assert report.delta < 0


# ---------------------------------------------------------------------------
# PromotionManager.stage_shadow integration with policy store
# ---------------------------------------------------------------------------


class _StubShadow:
    def __init__(self, report: ShadowReport) -> None:
        self.report = report
        self.calls: list[tuple[str, str]] = []

    def compare(self, active, candidate, dataset):
        self.calls.append((active.content, candidate.artifact.content))
        return self.report


def _report(verdict: str, *, active: float = 0.5, shadow: float = 0.6) -> ShadowReport:
    return ShadowReport(
        active_score=active,
        shadow_score=shadow,
        delta=shadow - active,
        sample_size=2,
        holdout_size=1,
        verdict=verdict,
        reason=f"stub_{verdict}",
        metrics={"active_score": active, "shadow_score": shadow},
    )


class TestStageShadow:
    def test_promote_writes_active(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        active = _artifact("active")
        store.set_active(active)
        candidate = CandidateVariant(artifact=_artifact("shadow"), score=0.6)
        manager = PromotionManager()
        decision = manager.stage_shadow(
            [candidate],
            active,
            _StubShadow(_report(SHADOW_VERDICT_PROMOTE)),
            _dataset(),
            policy_store=store,
        )
        assert decision.level == PromotionLevel.ACTIVE
        assert store.get_active(EvolutionArtifactType.RECALL_POLICY).content == "shadow"
        assert store.get_shadow(EvolutionArtifactType.RECALL_POLICY) is None
        assert decision.shadow_report is not None
        assert decision.shadow_report.verdict == SHADOW_VERDICT_PROMOTE

    def test_hold_writes_shadow(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        active = _artifact("active")
        store.set_active(active)
        candidate = CandidateVariant(artifact=_artifact("shadow"), score=0.6)
        manager = PromotionManager()
        decision = manager.stage_shadow(
            [candidate],
            active,
            _StubShadow(_report(SHADOW_VERDICT_HOLD, shadow=0.52)),
            _dataset(),
            policy_store=store,
        )
        assert decision.level == PromotionLevel.SHADOW
        assert store.get_active(EvolutionArtifactType.RECALL_POLICY).content == "active"
        held = store.get_shadow(EvolutionArtifactType.RECALL_POLICY)
        assert held is not None and held.content == "shadow"

    def test_reject_leaves_store_untouched(self) -> None:
        store = InMemoryEvolutionPolicyStore()
        active = _artifact("active")
        store.set_active(active)
        candidate = CandidateVariant(artifact=_artifact("shadow"), score=0.6)
        manager = PromotionManager()
        decision = manager.stage_shadow(
            [candidate],
            active,
            _StubShadow(_report(SHADOW_VERDICT_REJECT, shadow=0.3)),
            _dataset(),
            policy_store=store,
        )
        assert decision.level == PromotionLevel.REJECTED
        assert store.get_active(EvolutionArtifactType.RECALL_POLICY).content == "active"
        assert store.get_shadow(EvolutionArtifactType.RECALL_POLICY) is None

    def test_below_min_score(self) -> None:
        manager = PromotionManager(min_score=0.5)
        candidate = CandidateVariant(artifact=_artifact("shadow"), score=0.1)
        stub = _StubShadow(_report(SHADOW_VERDICT_PROMOTE))
        decision = manager.stage_shadow(
            [candidate],
            _artifact("active"),
            stub,
            _dataset(),
        )
        assert decision.level == PromotionLevel.REJECTED
        assert decision.reason == "score_below_threshold"
        assert stub.calls == []


# ---------------------------------------------------------------------------
# SQLite policy store: shadow slot persistence
# ---------------------------------------------------------------------------


class TestSqliteShadowSlot:
    def test_set_and_get_shadow(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        store.set_active(_artifact("active"))
        shadow = _artifact("shadow")
        store.set_shadow(shadow)
        loaded = store.get_shadow(EvolutionArtifactType.RECALL_POLICY)
        assert loaded is not None and loaded.artifact_id == shadow.artifact_id

    def test_active_clears_shadow(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        store.set_active(_artifact("active"))
        shadow = _artifact("shadow")
        store.set_shadow(shadow)
        store.set_active(shadow)
        assert store.get_shadow(EvolutionArtifactType.RECALL_POLICY) is None
        active = store.get_active(EvolutionArtifactType.RECALL_POLICY)
        assert active is not None and active.artifact_id == shadow.artifact_id

    def test_clear_demotes_staged(self, tmp_path) -> None:
        store = SQLiteEvolutionStore(tmp_path / "evolution.db")
        store.set_active(_artifact("active"))
        shadow = _artifact("shadow")
        store.set_shadow(shadow)
        store.clear_shadow(EvolutionArtifactType.RECALL_POLICY)
        assert store.get_shadow(EvolutionArtifactType.RECALL_POLICY) is None
        staged = store.list_staged(EvolutionArtifactType.RECALL_POLICY)
        assert any(item.artifact_id == shadow.artifact_id for item in staged)
