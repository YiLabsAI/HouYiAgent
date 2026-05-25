"""EmphasisDetector — detect user emphasis patterns for priority context.

Analyzes conversation history for signals that the user is emphasizing
certain rules, preferences, or constraints. Emphasized context receive
higher priority during context compression and are injected as reminders.

Emphasis signals:
  1. **Repetition**: Same concept mentioned 2+ times across messages.
  2. **Linguistic cues**: "always", "never", "must", "important", "remember".
  3. **Correction patterns**: User corrects the agent on the same point again.
  4. **Formatting emphasis**: ALL CAPS, exclamation marks, bold markers.

This addresses the "agent forgetfulness" problem observed in long conversations
where context compression drops earlier instructions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field


class EmphasisSignal(BaseModel):
    """A detected emphasis signal from user messages."""

    text: str
    signal_type: str
    strength: float = 1.0
    occurrences: int = 1
    source_messages: list[int] = Field(default_factory=list)


class EmphasisReport(BaseModel):
    """Aggregated emphasis analysis across conversation history."""

    signals: list[EmphasisSignal] = Field(default_factory=list)
    top_emphasized: list[str] = Field(default_factory=list)
    total_emphasis_score: float = 0.0


_EMPHASIS_PATTERNS: list[tuple[str, str, float]] = [
    (r"\balways\b", "directive", 1.5),
    (r"\bnever\b", "directive", 1.5),
    (r"\bmust\b", "directive", 1.2),
    (r"\bimportant\b", "emphasis", 1.3),
    (r"\bremember\b", "emphasis", 1.4),
    (r"\bdon'?t forget\b", "emphasis", 1.5),
    (r"\bcritical\b", "emphasis", 1.3),
    (r"\brequired?\b", "directive", 1.1),
    (r"!{2,}", "formatting", 1.2),
    (r"\b[A-Z]{3,}\b", "formatting", 1.1),
]


class EmphasisDetector:
    """Detects emphasis patterns in user messages.

    Provides both a per-message scanner and a batch analyzer for
    conversation history. Outputs an EmphasisReport that the
    ReminderInjector can use to elevate priority memories.
    """

    def __init__(
        self,
        *,
        repetition_threshold: int = 2,
        min_strength: float = 1.0,
    ) -> None:
        self._rep_threshold = repetition_threshold
        self._min_strength = min_strength

    def analyze(self, messages: list[dict[str, Any]]) -> EmphasisReport:
        """Analyze conversation for emphasis patterns.

        Only examines user messages. Returns signals sorted by strength.
        """
        user_msgs = [
            (i, m["content"])
            for i, m in enumerate(messages)
            if m.get("role") == "user" and m.get("content")
        ]

        all_signals: list[EmphasisSignal] = []
        concept_counter: Counter[str] = Counter()

        for msg_idx, content in user_msgs:
            for pattern, signal_type, strength in _EMPHASIS_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    all_signals.append(
                        EmphasisSignal(
                            text=match if isinstance(match, str) else str(match),
                            signal_type=signal_type,
                            strength=strength,
                            source_messages=[msg_idx],
                        )
                    )

            phrases = _extract_key_phrases(content)
            for phrase in phrases:
                concept_counter[phrase] += 1

        repeated = [
            EmphasisSignal(
                text=phrase,
                signal_type="repetition",
                strength=1.0 + 0.5 * (count - 1),
                occurrences=count,
            )
            for phrase, count in concept_counter.items()
            if count >= self._rep_threshold
        ]
        all_signals.extend(repeated)

        filtered = [s for s in all_signals if s.strength >= self._min_strength]
        filtered.sort(key=lambda s: s.strength, reverse=True)

        top = []
        seen = set()
        for sig in filtered[:10]:
            normalized = sig.text.lower().strip()
            if normalized not in seen:
                top.append(sig.text)
                seen.add(normalized)

        total = sum(s.strength for s in filtered)

        return EmphasisReport(
            signals=filtered,
            top_emphasized=top[:5],
            total_emphasis_score=round(total, 2),
        )


def _extract_key_phrases(text: str) -> list[str]:
    """Extract short phrases (2-4 words) that might represent instructions."""
    words = re.findall(r"\b\w+\b", text.lower())
    phrases = []
    for size in (3, 2):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i : i + size])
            if len(phrase) > 6:
                phrases.append(phrase)
    return phrases
