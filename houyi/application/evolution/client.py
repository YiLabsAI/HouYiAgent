from __future__ import annotations

from dataclasses import dataclass

from houyi.application.evolution.artifacts import EvolutionArtifact, EvolutionArtifactType
from houyi.application.evolution.event_log import EvolutionEventLog
from houyi.application.evolution.events import EvolutionEvent
from houyi.application.evolution.policy_store import EvolutionPolicyStore


@dataclass(slots=True)
class EvolutionClient:
    event_log: EvolutionEventLog
    policy_store: EvolutionPolicyStore
    artifact_type: EvolutionArtifactType

    def emit_event(self, event: EvolutionEvent) -> None:
        self.event_log.append(event)

    def get_active_artifact(self) -> EvolutionArtifact:
        active = self.policy_store.get_active(self.artifact_type)
        if active is None:
            raise LookupError("active evolution artifact is not configured")
        return active

    def update_active_artifact(self, artifact: EvolutionArtifact) -> None:
        self.policy_store.set_active(artifact)
