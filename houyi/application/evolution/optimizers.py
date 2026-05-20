from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from houyi.application.evolution.artifacts import CandidateVariant, EvolutionArtifact
from houyi.application.evolution.events import EvolutionSignal


class EvolutionOptimizer(Protocol):
    def propose(
        self,
        artifact: EvolutionArtifact,
        signals: list[EvolutionSignal],
    ) -> list[CandidateVariant]: ...


class DeterministicEvolutionOptimizer:
    def propose(
        self,
        artifact: EvolutionArtifact,
        signals: list[EvolutionSignal],
    ) -> list[CandidateVariant]:
        if not signals:
            return []
        severity = sum(signal.severity for signal in signals) / len(signals)
        source_ids = tuple(event_id for signal in signals for event_id in signal.event_ids)
        candidate = replace(
            artifact,
            version=artifact.version + 1,
            parent_id=artifact.artifact_id,
            metadata={**artifact.metadata, "optimizer": "deterministic"},
        )
        return [
            CandidateVariant(
                artifact=candidate,
                score=round(severity, 4),
                source_signal_ids=source_ids,
            )
        ]
