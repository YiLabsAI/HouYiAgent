from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from houyi.application.evolution.artifacts import EvolutionArtifact
from houyi.application.evolution.dataset import EvolutionExample


class EvolutionModule(Protocol):
    @property
    def artifact(self) -> EvolutionArtifact: ...

    def render(self) -> str: ...

    def apply_text(self, text: str) -> EvolutionArtifact: ...

    def predict(self, example: EvolutionExample) -> str: ...


@dataclass(frozen=True, slots=True)
class TextArtifactModule:
    artifact: EvolutionArtifact

    def render(self) -> str:
        return self.artifact.content

    def apply_text(self, text: str) -> EvolutionArtifact:
        return replace(
            self.artifact,
            content=text,
            version=self.artifact.version + 1,
            parent_id=self.artifact.artifact_id,
        )

    def predict(self, example: EvolutionExample) -> str:
        return f"{self.artifact.content}\n\nTask: {example.task_input}"


class EvolutionModuleFactory(Protocol):
    def create(self, artifact: EvolutionArtifact) -> EvolutionModule: ...


class TextArtifactModuleFactory:
    def create(self, artifact: EvolutionArtifact) -> EvolutionModule:
        return TextArtifactModule(artifact)
