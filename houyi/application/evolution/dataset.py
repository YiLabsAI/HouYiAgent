from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from houyi.application.evolution.events import EvolutionSignal


@dataclass(frozen=True, slots=True)
class EvolutionExample:
    task_input: str
    expected_behavior: str
    category: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvolutionDataset:
    train: tuple[EvolutionExample, ...] = ()
    validation: tuple[EvolutionExample, ...] = ()
    holdout: tuple[EvolutionExample, ...] = ()


class EvolutionDatasetBuilder(Protocol):
    def build(self, signals: list[EvolutionSignal]) -> EvolutionDataset: ...


class SignalDatasetBuilder:
    def build(self, signals: list[EvolutionSignal]) -> EvolutionDataset:
        examples = tuple(
            EvolutionExample(
                task_input=signal.target,
                expected_behavior=signal.signal_type,
                category=signal.signal_type,
                source="evolution_signal",
                metadata={"severity": f"{signal.severity:.4f}"},
            )
            for signal in signals
        )
        if len(examples) < 3:
            return EvolutionDataset(train=examples, validation=examples, holdout=examples)
        return EvolutionDataset(
            train=examples[:-2],
            validation=examples[-2:-1],
            holdout=examples[-1:],
        )
