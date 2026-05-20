from __future__ import annotations

import pytest

from houyi.application.evolution import (
    CandidateVariant,
    DspyGepaOptimizer,
    DspyGepaUnavailableError,
    EvolutionArtifact,
    EvolutionArtifactType,
    EvolutionExample,
    EvolutionSignal,
    HeuristicEvolutionEvaluator,
    InMemoryEvolutionCursorStore,
    SignalDatasetBuilder,
    TextArtifactModuleFactory,
)


def test_signal_dataset_builder_splits() -> None:
    signals = [
        EvolutionSignal("recall_failure", "recall_policy", 0.8, ("a",)),
        EvolutionSignal("user_correction", "recall_policy", 0.7, ("b",)),
        EvolutionSignal("recall_failure", "recall_policy", 0.6, ("c",)),
    ]

    dataset = SignalDatasetBuilder().build(signals)

    assert len(dataset.train) == 1
    assert len(dataset.validation) == 1
    assert len(dataset.holdout) == 1


def test_heuristic_evaluator_scores() -> None:
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="candidate",
    )
    candidate = CandidateVariant(artifact, score=0.5)
    dataset = SignalDatasetBuilder().build(
        [EvolutionSignal("recall_failure", "recall_policy", 0.8, ("a",))]
    )

    evaluations = HeuristicEvolutionEvaluator().evaluate([candidate], dataset)

    assert evaluations[0].passed is True
    assert evaluations[0].score > candidate.score


def test_text_artifact_module_wraps() -> None:
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="use source-grounded recall",
    )
    module = TextArtifactModuleFactory().create(artifact)

    updated = module.apply_text("prefer source-grounded temporal recall")

    assert module.render() == "use source-grounded recall"
    assert updated.version == 2
    assert updated.parent_id == artifact.artifact_id
    assert "Task: temporal query" in module.predict(
        EvolutionExample("temporal query", "answer with evidence", "temporal", "test")
    )


def test_cursor_store_tracks_consumer() -> None:
    store = InMemoryEvolutionCursorStore()

    store.set_cursor("worker-a", 3)

    assert store.get_cursor("worker-a") == 3
    assert store.get_cursor("worker-b") == 0


def test_dspy_gepa_optional() -> None:
    pytest.importorskip("dspy", reason="dspy installed; unavailable path not applicable")
    optimizer = DspyGepaOptimizer()
    assert optimizer.optimizer_model


def test_dspy_gepa_guard() -> None:
    pytest.importorskip("dspy", reason="dspy is optional")
    optimizer = DspyGepaOptimizer()
    artifact = EvolutionArtifact(
        artifact_type=EvolutionArtifactType.RECALL_POLICY,
        content="baseline",
    )

    with pytest.raises(DspyGepaUnavailableError, match="compile is disabled"):
        optimizer.propose(
            artifact,
            [EvolutionSignal("recall_failure", "recall_policy", 0.8, ("a",))],
        )


def test_dspy_gepa_unavailable_message() -> None:
    try:
        import dspy  # noqa: F401
    except ImportError:
        with pytest.raises(DspyGepaUnavailableError, match="optional dependency 'dspy'"):
            DspyGepaOptimizer()
