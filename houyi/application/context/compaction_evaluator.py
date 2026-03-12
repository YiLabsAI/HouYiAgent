from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import CompactionMetrics, CompactionRecord


class CompactionEvaluator:
    def __init__(self, token_estimator: TokenEstimator):
        self.estimator = token_estimator

    def evaluate(
        self,
        *,
        before_messages: list[dict[str, Any]],
        summary: str,
        pinned_message_ids: Iterable[str] | None = None,
        retained_refs: Iterable[str] | None = None,
        source_message_ids: Iterable[str] | None = None,
        summary_coherence_score: float | None = None,
        metadata: dict[str, Any] | None = None,
        trigger: str = "threshold",
    ) -> CompactionRecord:
        pinned_ids = [str(item) for item in (pinned_message_ids or []) if str(item)]
        retained_ref_list = [str(item) for item in (retained_refs or []) if str(item)]
        source_ids = [str(item) for item in (source_message_ids or []) if str(item)]
        tokens_before = self.estimator.count_messages(before_messages)
        tokens_after = self.estimator.count_text(summary)
        entities_before = self._extract_entities_from_messages(before_messages)
        entities_after = self._extract_entities_from_text(summary)
        retained_count = len(entities_before & entities_after)
        retained_entity_coverage = retained_count / len(entities_before) if entities_before else 1.0
        pin_violation_count = self._count_pin_violations(
            before_messages=before_messages,
            summary=summary,
            pinned_message_ids=pinned_ids,
        )
        metrics = CompactionMetrics(
            compression_ratio=(tokens_after / tokens_before) if tokens_before > 0 else 1.0,
            retained_entity_coverage=retained_entity_coverage,
            summary_coherence_score=summary_coherence_score,
            pin_violation_count=pin_violation_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_compacted=len(before_messages),
            retained_refs_count=len(retained_ref_list),
            metadata=dict(metadata or {}),
        )
        return CompactionRecord(
            trigger=trigger,
            summary=summary,
            source_message_ids=source_ids,
            pinned_message_ids=pinned_ids,
            retained_refs=retained_ref_list,
            metrics=metrics,
            metadata=dict(metadata or {}),
        )

    def _extract_entities_from_messages(self, messages: list[dict[str, Any]]) -> set[str]:
        entities: set[str] = set()
        for message in messages:
            content = message.get("content")
            entities.update(self._extract_entities_from_text(str(content or "")))
        return entities

    def _extract_entities_from_text(self, text: str) -> set[str]:
        return {
            token for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", text) if len(token) >= 2
        }

    def _count_pin_violations(
        self,
        *,
        before_messages: list[dict[str, Any]],
        summary: str,
        pinned_message_ids: list[str],
    ) -> int:
        if not pinned_message_ids:
            return 0
        summary_lower = summary.lower()
        violations = 0
        for message in before_messages:
            message_id = str(message.get("message_id") or "")
            if message_id not in pinned_message_ids:
                continue
            content = str(message.get("content") or "").strip()
            if content and content.lower() not in summary_lower:
                violations += 1
        return violations
