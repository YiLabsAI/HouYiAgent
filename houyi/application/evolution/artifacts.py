from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class EvolutionArtifactType(str, Enum):
    RECALL_POLICY = "recall_policy"
    RERANK_POLICY = "rerank_policy"
    IDK_POLICY = "idk_policy"
    WRITER_PROMPT = "writer_prompt"
    EXTRACTOR_PROMPT = "extractor_prompt"
    DREAMING_POLICY = "dreaming_policy"
    AGENT_STRATEGY = "agent_strategy"


@dataclass(frozen=True, slots=True)
class EvolutionArtifact:
    artifact_type: EvolutionArtifactType
    content: str
    version: int = 1
    metadata: dict[str, str] = field(default_factory=dict)
    parent_id: str | None = None
    artifact_id: str = field(default_factory=lambda: f"art_{uuid.uuid4().hex}")


@dataclass(frozen=True, slots=True)
class CandidateVariant:
    artifact: EvolutionArtifact
    score: float
    source_signal_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
