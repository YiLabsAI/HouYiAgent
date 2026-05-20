"""Detect "please remember X" patterns and pin them at L0+ time.

 when a user says "remember X" (zh: ji-zhu / ji-xia / qing-ji-zhu)
or "pin this: X", we don't want to wait for the L1 extractor — the intent is
unambiguous and the cost of skipping the LLM is zero. The detector
runs synchronously inside TurnWriter.fast_path, parses the
trailing payload, and writes a high-confidence MemoryRecord whose
embedding is left None (the backfill worker will fill
it).

The detector is intentionally regex-driven and bilingual (zh + en).
Pattern coverage is conservative: false positives create durable rows
that pollute recall, so we only fire on prefixes that read as explicit
imperatives. Edge cases — multi-line content, embedded quotes — round
to the leftmost prefix match.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from houyi.adapters.memory.types import (
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RawTurn,
)

logger = logging.getLogger(__name__)


# Patterns are ordered from most specific to most generic. Each pattern
# captures a single "payload" group consisting of everything after the
# imperative cue. The regex is anchored to start-of-string after a
# leading whitespace strip; we don't want detector firings deep inside
# a message.
_PIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # English: "please remember that ...", "please remember: ..."
    re.compile(
        r"^(?:please\s+)?(?:remember|note|keep in mind)\s*(?:that|:)?\s*(?P<payload>.+)$",
        re.IGNORECASE | re.DOTALL,
    ),
    # English: "pin this:" / "pin: ..."
    re.compile(
        r"^pin(?:\s+this)?\s*:?\s*(?P<payload>.+)$",
        re.IGNORECASE | re.DOTALL,
    ),
    # Chinese imperatives (ji-zhu / ji-xia / bang-wo-ji / qing-ji-zhu)
    # followed by optional yi-xia / colon / fullwidth-comma / fullwidth-period.
    # We skip a single trailing punctuation char so payload="X" lands the
    # same way regardless of which separator the user typed.
    re.compile(
        "^(?:\u8bf7)?(?:\u5e2e\u6211)?"
        "(?:\u8bb0\u4f4f|\u8bb0\u4e0b|\u8bb0\u4e00\u4e0b)"
        "\\s*[:\uff1a,\uff0c.\u3002]?\\s*(?P<payload>.+)$",
        re.DOTALL,
    ),
)


@dataclass(frozen=True)
class ExplicitPinSignal:
    """Result of a successful pin-pattern match."""

    payload: str
    """The text the user asked us to remember, stripped of the cue."""

    pattern_index: int
    """Which entry in _PIN_PATTERNS fired. Surfaced for tests
    and analytics so we can tell zh / en cues apart in audit logs.
    """


class _RecordSink(Protocol):
    def put(self, record: MemoryRecord) -> None: ...


class ExplicitPinDetector:
    """TurnDetector that materializes pin-imperative payloads.

    Construction takes a backend exposing put. On every call
    the detector strips the input, scans _PIN_PATTERNS in
    order, and on the first hit writes a MemoryRecord:

    - key = f"pin:{turn.session_id}:{turn.turn_id}" so duplicate
    pins don't collide; the recall layer treats pins as standalone
    hits, not entity-state rows.
    - content = signal.payload
    - scope = USER, memory_type = FACT, confidence = 0.95
    - embedding = None (deferred — see EmbeddingBackfillWorker)
    """

    def __init__(
        self,
        backend: _RecordSink,
        *,
        scope: MemoryScope = MemoryScope.USER,
        provider_label: str = "explicit_pin_detector",
    ) -> None:
        if backend is None:
            raise ValueError("backend is required")
        self._backend = backend
        self._scope = scope
        self._provider_label = provider_label
        self._last_signal: ExplicitPinSignal | None = None

    @property
    def last_signal(self) -> ExplicitPinSignal | None:
        """Most recent successful match, or None if the last turn
        did not fire. Reset on every detect call.
        """
        return self._last_signal

    def detect(self, turn: RawTurn) -> None:
        signal = self._scan(turn.content)
        self._last_signal = signal
        if signal is None:
            return
        try:
            self._backend.put(self._make_record(turn, signal))
        except Exception:
            logger.warning("explicit-pin write failed for turn %s", turn.turn_id, exc_info=True)

    @staticmethod
    def _scan(text: str) -> ExplicitPinSignal | None:
        stripped = text.strip()
        if not stripped:
            return None
        for idx, pattern in enumerate(_PIN_PATTERNS):
            match = pattern.match(stripped)
            if match is None:
                continue
            payload = match.group("payload").strip()
            if not payload:
                # "remember" with no payload — refuse to write empty pins.
                return None
            return ExplicitPinSignal(payload=payload, pattern_index=idx)
        return None

    def _make_record(self, turn: RawTurn, signal: ExplicitPinSignal) -> MemoryRecord:
        return MemoryRecord(
            key=f"pin:{turn.session_id}:{turn.turn_id}",
            content=signal.payload,
            scope=self._scope,
            memory_type=MemoryType.FACT,
            confidence=0.95,
            provenance=MemoryProvenance(
                source_type="explicit_pin",
                source_ids=[turn.turn_id],
                extracted_by=self._provider_label,
            ),
            embedding=None,
            tags=["pinned"],
            metadata={
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "pattern_index": str(signal.pattern_index),
            },
        )


__all__ = ["ExplicitPinDetector", "ExplicitPinSignal"]
