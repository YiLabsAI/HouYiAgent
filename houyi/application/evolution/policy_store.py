from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from houyi.application.evolution.artifacts import EvolutionArtifact, EvolutionArtifactType


class EvolutionPolicyStore(Protocol):
    def get_active(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact | None: ...

    def set_active(self, artifact: EvolutionArtifact) -> None: ...

    def stage(self, artifact: EvolutionArtifact) -> None: ...

    def list_staged(self, artifact_type: EvolutionArtifactType) -> list[EvolutionArtifact]: ...

    def activate(self, artifact_id: str) -> EvolutionArtifact: ...

    def rollback(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact: ...

    def list_history(self, artifact_type: EvolutionArtifactType) -> list[EvolutionArtifact]: ...

    def revert_to(self, artifact_id: str) -> EvolutionArtifact: ...

    def set_shadow(self, artifact: EvolutionArtifact) -> None: ...

    def get_shadow(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact | None: ...

    def clear_shadow(self, artifact_type: EvolutionArtifactType) -> None: ...


@dataclass(slots=True)
class InMemoryEvolutionPolicyStore:
    _active: dict[EvolutionArtifactType, EvolutionArtifact] = field(default_factory=dict)
    _staged: dict[EvolutionArtifactType, list[EvolutionArtifact]] = field(default_factory=dict)
    _history: dict[EvolutionArtifactType, list[EvolutionArtifact]] = field(default_factory=dict)
    _shadow: dict[EvolutionArtifactType, EvolutionArtifact] = field(default_factory=dict)

    def get_active(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact | None:
        return self._active.get(artifact_type)

    def set_active(self, artifact: EvolutionArtifact) -> None:
        current = self._active.get(artifact.artifact_type)
        if current is not None and current.artifact_id != artifact.artifact_id:
            self._history.setdefault(artifact.artifact_type, []).append(current)
        self._active[artifact.artifact_type] = artifact
        shadow = self._shadow.get(artifact.artifact_type)
        if shadow is not None and shadow.artifact_id == artifact.artifact_id:
            self._shadow.pop(artifact.artifact_type, None)

    def stage(self, artifact: EvolutionArtifact) -> None:
        self._staged.setdefault(artifact.artifact_type, []).append(artifact)

    def list_staged(self, artifact_type: EvolutionArtifactType) -> list[EvolutionArtifact]:
        return list(self._staged.get(artifact_type, []))

    def activate(self, artifact_id: str) -> EvolutionArtifact:
        for artifact_type, staged in self._staged.items():
            for artifact in staged:
                if artifact.artifact_id == artifact_id:
                    self.set_active(artifact)
                    self._staged[artifact_type] = [
                        item for item in staged if item.artifact_id != artifact_id
                    ]
                    return artifact
        raise LookupError(f"staged artifact not found: {artifact_id}")

    def rollback(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact:
        history = self._history.get(artifact_type, [])
        if not history:
            raise LookupError(f"no rollback artifact for {artifact_type.value}")
        previous = history.pop()
        self._active[artifact_type] = previous
        return previous

    def list_history(self, artifact_type: EvolutionArtifactType) -> list[EvolutionArtifact]:
        return list(self._history.get(artifact_type, []))

    def revert_to(self, artifact_id: str) -> EvolutionArtifact:
        for artifact_type, history in self._history.items():
            for index, candidate in enumerate(history):
                if candidate.artifact_id == artifact_id:
                    current = self._active.get(artifact_type)
                    new_history = history[:index] + history[index + 1 :]
                    if current is not None and current.artifact_id != artifact_id:
                        new_history.append(current)
                    self._history[artifact_type] = new_history
                    self._active[artifact_type] = candidate
                    return candidate
        raise LookupError(f"history artifact not found: {artifact_id}")

    def set_shadow(self, artifact: EvolutionArtifact) -> None:
        self._shadow[artifact.artifact_type] = artifact

    def get_shadow(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact | None:
        return self._shadow.get(artifact_type)

    def clear_shadow(self, artifact_type: EvolutionArtifactType) -> None:
        self._shadow.pop(artifact_type, None)
