"""Unknown-answer guard for memory recall.

The guard is the final read-side safety check. It does not delete or
hide candidates; it assigns a reason and suggested action so the prompt
layer can avoid unsupported answers while traces still show the raw
evidence that was considered.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from houyi.adapters.memory.event_emitter import MemoryEventEmitter
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallReason,
    RecallResult,
)
from houyi.application.evolution.events import EvolutionEventType


@dataclass(frozen=True)
class IDKGuardConfig:
    """Thresholds controlling unknown-answer behavior."""

    evidence_threshold: float = 0.5
    coverage_threshold: float = 0.2
    contradiction_signal: str = "contradicts"
    recency_signal: str = "recency_winner"


class IDKGuard:
    """Map candidate evidence to a guarded recall result."""

    def __init__(
        self,
        config: IDKGuardConfig | None = None,
        *,
        emitter: MemoryEventEmitter | None = None,
    ) -> None:
        self._config = config or IDKGuardConfig()
        # Emitter is optional; when None the guard runs exactly as before.
        # When wired, every non-sufficient outcome publishes an
        # IDK_DECISION event so the evolution control plane can mine
        # under-served queries without us re-deriving them from logs.
        self._emitter = emitter or MemoryEventEmitter()

    def evaluate(
        self,
        *,
        query_type: QueryType,
        candidates: list[RecallCandidate],
        trace: dict[str, object] | None = None,
    ) -> RecallResult:
        """Return a recall result with guard reason and suggested action."""
        base_trace = dict(trace or {})
        guard_trace: dict[str, object] = {}
        base_trace["guard"] = guard_trace

        if not candidates:
            reason = (
                RecallReason.EXPLICIT_ABSENCE
                if query_type == QueryType.NEGATION_CHECK
                else RecallReason.NO_CANDIDATES
            )
            action = "admit_unknown" if reason != RecallReason.EXPLICIT_ABSENCE else "state_absence"
            guard_trace["signal"] = reason.value
            self._emit_idk(query_type, reason, top_score=0.0, candidate_count=0)
            return RecallResult(
                candidates=[],
                query_type=query_type,
                reason=reason,
                suggested_action=action,
                trace=base_trace,
            )

        top_score = _guard_score(candidates[0])
        if top_score < self._config.evidence_threshold:
            guard_trace["signal"] = RecallReason.LOW_EVIDENCE.value
            guard_trace["top_score"] = top_score
            self._emit_idk(
                query_type,
                RecallReason.LOW_EVIDENCE,
                top_score=top_score,
                candidate_count=len(candidates),
            )
            return RecallResult(
                candidates=candidates,
                query_type=query_type,
                reason=RecallReason.LOW_EVIDENCE,
                suggested_action="admit_unknown",
                trace=base_trace,
            )

        coverage = candidates[0].signals.get("evidence_coverage")
        if isinstance(coverage, int | float) and coverage < self._config.coverage_threshold:
            guard_trace["signal"] = RecallReason.LOW_EVIDENCE.value
            guard_trace["evidence_coverage"] = float(coverage)
            self._emit_idk(
                query_type,
                RecallReason.LOW_EVIDENCE,
                top_score=top_score,
                candidate_count=len(candidates),
                coverage=float(coverage),
            )
            return RecallResult(
                candidates=candidates,
                query_type=query_type,
                reason=RecallReason.LOW_EVIDENCE,
                suggested_action="admit_unknown",
                trace=base_trace,
            )

        if self._has_unresolved_contradiction(candidates):
            guard_trace["signal"] = RecallReason.CONTRADICTING_EVIDENCE.value
            self._emit_idk(
                query_type,
                RecallReason.CONTRADICTING_EVIDENCE,
                top_score=top_score,
                candidate_count=len(candidates),
            )
            return RecallResult(
                candidates=candidates,
                query_type=query_type,
                reason=RecallReason.CONTRADICTING_EVIDENCE,
                suggested_action="ask_user_clarify",
                trace=base_trace,
            )

        guard_trace["signal"] = RecallReason.SUFFICIENT.value
        return RecallResult(
            candidates=candidates,
            query_type=query_type,
            reason=RecallReason.SUFFICIENT,
            suggested_action="use_evidence",
            trace=base_trace,
        )

    def _emit_idk(
        self,
        query_type: QueryType,
        reason: RecallReason,
        *,
        top_score: float,
        candidate_count: int,
        coverage: float | None = None,
    ) -> None:
        metrics: dict[str, float] = {
            "top_score": float(top_score),
            "candidate_count": float(candidate_count),
        }
        if coverage is not None:
            metrics["coverage"] = float(coverage)
        self._emitter.emit(
            EvolutionEventType.IDK_DECISION,
            target="recall_idk_guard",
            payload={"query_type": query_type.value, "reason": reason.value},
            metrics=metrics,
        )

    def _has_unresolved_contradiction(self, candidates: Iterable[RecallCandidate]) -> bool:
        has_contradiction = False
        has_recency_winner = False
        for cand in candidates:
            has_contradiction = has_contradiction or bool(
                cand.signals.get(self._config.contradiction_signal)
            )
            has_recency_winner = has_recency_winner or bool(
                cand.signals.get(self._config.recency_signal)
            )
        return has_contradiction and not has_recency_winner


def _guard_score(candidate: RecallCandidate) -> float:
    return float(
        candidate.signals.get(
            "rerank_score",
            candidate.signals.get("fused_score", candidate.score),
        )
    )


__all__ = ["IDKGuard", "IDKGuardConfig"]
