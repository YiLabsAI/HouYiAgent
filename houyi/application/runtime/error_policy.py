"""Error handling policies and conflict resolution for agent orchestration."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Error Policy ────────────────────────────────────────────


class FallbackStrategy(str, Enum):
    """What to do when an agent fails after all retries are exhausted."""

    SKIP = "skip"
    REASSIGN = "reassign"
    DEGRADE = "degrade"
    ABORT = "abort"


class ErrorPolicy(BaseModel):
    """Configurable error-handling policy for orchestrated agent execution.

    Applied per-agent or globally by the orchestrator to decide retry
    behaviour and fallback strategy on failure.
    """

    return_exceptions: bool = True
    max_retries: int = 1
    retry_delay_ms: int = 1000
    retry_on: list[str] = Field(default_factory=lambda: ["timeout", "transient"])
    fallback_strategy: FallbackStrategy = FallbackStrategy.SKIP

    def should_retry(self, error: BaseException, attempt: int) -> bool:
        """Return True if the error is retryable and we have attempts left."""
        if attempt >= self.max_retries:
            return False
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        return any(tag in error_type or tag in error_msg for tag in self.retry_on)


# ── Conflict Resolution ────────────────────────────────────


class ConflictResolution(BaseModel):
    """Outcome of resolving a single conflict."""

    method: str = ""
    winner: str | None = None
    reasoning: str = ""
    confidence: float = 0.0


class ConflictRecord(BaseModel):
    """A detected disagreement between two agent results."""

    question_id: str = ""
    agent_a_id: str = ""
    agent_a_conclusion: str = ""
    agent_b_id: str = ""
    agent_b_conclusion: str = ""
    conflict_type: str = "factual"
    resolution: ConflictResolution | None = None


class AgentTaskResult(BaseModel):
    """Lightweight result wrapper for conflict detection and error policy."""

    agent_id: str = ""
    task: str = ""
    output: Any = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictResolver:
    """Detects and resolves conflicting conclusions from parallel agents.

    Two resolution strategies:

    * **Source voting** — majority answer wins when sources overlap.
    * **LLM arbitration** — an LLM judge picks the more credible answer
      (placeholder; requires an LLM adapter in Phase 2).
    """

    async def detect(self, results: list[AgentTaskResult]) -> list[ConflictRecord]:
        """Compare pairwise results and return detected conflicts."""
        conflicts: list[ConflictRecord] = []
        for i, a in enumerate(results):
            for b in results[i + 1 :]:
                if not a.success or not b.success:
                    continue
                a_text = str(a.output).strip().lower() if a.output else ""
                b_text = str(b.output).strip().lower() if b.output else ""
                if a_text and b_text and a_text != b_text:
                    conflicts.append(
                        ConflictRecord(
                            agent_a_id=a.agent_id,
                            agent_a_conclusion=str(a.output),
                            agent_b_id=b.agent_id,
                            agent_b_conclusion=str(b.output),
                        )
                    )
        return conflicts

    async def resolve(self, conflict: ConflictRecord) -> ConflictResolution:
        """Resolve a conflict using source-voting heuristic.

        In Phase 2 this will support LLM arbitration with a judge prompt.
        """
        a_len = len(conflict.agent_a_conclusion)
        b_len = len(conflict.agent_b_conclusion)
        winner = conflict.agent_a_id if a_len >= b_len else conflict.agent_b_id
        return ConflictResolution(
            method="source_voting",
            winner=winner,
            reasoning="Longer answer assumed more detailed (heuristic placeholder)",
            confidence=0.6,
        )
