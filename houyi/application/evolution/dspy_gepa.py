from __future__ import annotations

import os
from dataclasses import dataclass

from houyi.application.evolution.artifacts import CandidateVariant, EvolutionArtifact
from houyi.application.evolution.dataset import EvolutionDataset, SignalDatasetBuilder
from houyi.application.evolution.events import EvolutionSignal
from houyi.application.evolution.modules import EvolutionModuleFactory, TextArtifactModuleFactory


class DspyGepaUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DspyGepaConfig:
    optimizer_model: str = "openai/gpt-4.1"
    eval_model: str = "openai/gpt-4.1-mini"
    max_steps: int = 10
    enabled_env_var: str = "HOUYI_ENABLE_DSPY_GEPA"


class DspyGepaOptimizer:
    def __init__(
        self,
        *,
        config: DspyGepaConfig | None = None,
        module_factory: EvolutionModuleFactory | None = None,
    ) -> None:
        self.config = config or DspyGepaConfig()
        self.optimizer_model = self.config.optimizer_model
        self.eval_model = self.config.eval_model
        self.module_factory = module_factory or TextArtifactModuleFactory()
        try:
            import dspy
        except ImportError as exc:
            raise DspyGepaUnavailableError(
                "DSPy/GEPA optimization requires optional dependency 'dspy'. "
                "Install it in an evolution runtime environment; do not add it to the hot path."
            ) from exc

        self._dspy = dspy

    def propose(
        self,
        artifact: EvolutionArtifact,
        signals: list[EvolutionSignal],
    ) -> list[CandidateVariant]:
        if not signals:
            return []
        if os.getenv(self.config.enabled_env_var) != "1":
            raise DspyGepaUnavailableError(
                f"real DSPy/GEPA compile is disabled; set {self.config.enabled_env_var}=1 "
                "in a dedicated evolution runtime to enable it"
            )
        dataset = SignalDatasetBuilder().build(signals)
        optimized_text = self._compile_text_artifact(artifact, dataset)
        candidate = self.module_factory.create(artifact).apply_text(optimized_text)
        return [
            CandidateVariant(
                artifact=candidate,
                score=1.0,
                source_signal_ids=tuple(
                    event_id for signal in signals for event_id in signal.event_ids
                ),
                metadata={"optimizer": "dspy_gepa", "eval_model": self.eval_model},
            )
        ]

    def _compile_text_artifact(
        self,
        artifact: EvolutionArtifact,
        dataset: EvolutionDataset,
    ) -> str:
        dspy = self._dspy
        module = _DspyTextArtifactModule(dspy, artifact.content)
        trainset = [
            dspy.Example(
                task_input=example.task_input,
                expected_behavior=example.expected_behavior,
            ).with_inputs("task_input")
            for example in dataset.train
        ]
        valset = [
            dspy.Example(
                task_input=example.task_input,
                expected_behavior=example.expected_behavior,
            ).with_inputs("task_input")
            for example in dataset.validation
        ]
        optimizer = dspy.GEPA(
            metric=_dspy_text_metric,
            max_steps=self.config.max_steps,
        )
        optimized = optimizer.compile(module, trainset=trainset, valset=valset)
        return str(getattr(optimized, "artifact_text", artifact.content))


class _DspyTextArtifactModule:
    def __init__(self, dspy_module, artifact_text: str) -> None:
        self.artifact_text = artifact_text
        self._predictor = dspy_module.Predict("artifact_text, task_input -> output")

    def __call__(self, task_input: str):
        return self.forward(task_input)

    def forward(self, task_input: str):
        return self._predictor(
            artifact_text=self.artifact_text,
            task_input=task_input,
        )


def _dspy_text_metric(example, prediction, trace=None) -> float:
    output = str(getattr(prediction, "output", ""))
    expected = str(getattr(example, "expected_behavior", ""))
    if not output:
        return 0.0
    if not expected:
        return 0.5
    expected_terms = {term.lower() for term in expected.split() if len(term) > 2}
    output_terms = {term.lower() for term in output.split() if len(term) > 2}
    if not expected_terms:
        return 0.5
    return len(expected_terms & output_terms) / len(expected_terms)
