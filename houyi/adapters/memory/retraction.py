"""Retraction signal detection.

Identifies natural-language cues that the speaker is taking back a prior
statement, so the writer pipeline can call invalidate_unit instead of
recording a contradictory new fact. Six bilingual patterns cover the
common surface forms.

The detector intentionally returns a coarse boolean plus the matched
pattern label rather than attempting to extract the retracted target;
target identification is handled downstream by the agent loop, which
already has the surrounding entity context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from houyi.adapters.memory.event_emitter import MemoryEventEmitter
from houyi.adapters.memory.resolver import MemoryWriterTools
from houyi.application.evolution.events import EvolutionEventType


@dataclass(frozen=True)
class RetractionSignal:
    """A single retraction match.

    label identifies which pattern fired so analytics and tests can
    report per-pattern recall without re-deriving it from the surface
    form. matched_text is the substring that triggered the match,
    handy for telemetry and prompt-side debugging.
    """

    label: str
    matched_text: str


# Pattern catalogue - keep this list explicit and ordered so the
# detector reports the strongest signal first when multiple patterns
# overlap. Each entry is (label, compiled regex).
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "explicit_wrong",
        re.compile(
            r"(?i)\b(?:i\s+(?:was|am)\s+wrong|i\s+made\s+a\s+mistake|"
            r"my\s+(?:bad|mistake)|i\s+misspoke)\b"
        ),
    ),
    (
        "actually_correction",
        re.compile(
            r"(?i)\b(?:actually|wait|hold on|on\s+second\s+thought|"
            r"come\s+to\s+think\s+of\s+it)\b[^.!?\n]*?\b"
            r"(?:not|isn'?t|wasn'?t|aren'?t|weren'?t)\b"
        ),
    ),
    (
        "scratch_that",
        re.compile(
            r"(?i)\b(?:scratch\s+that|forget\s+(?:what|that)\s+i\s+said|"
            r"ignore\s+(?:what|that)\s+i\s+(?:just\s+)?said|"
            r"never\s*mind(?:\s+what\s+i\s+said)?)\b"
        ),
    ),
    (
        "let_me_correct",
        re.compile(
            r"(?i)\b(?:let\s+me\s+(?:correct|fix|amend|revise)|"
            r"correction|to\s+correct\s+(?:myself|that)|"
            r"(?:i|let\s+me)\s+take\s+that\s+back)\b"
        ),
    ),
    (
        "earlier_statement_void",
        re.compile(
            r"(?i)\b(?:what\s+i\s+(?:said|told\s+you)\s+(?:earlier|before|"
            r"a\s+(?:moment|second|minute)\s+ago)\s+(?:was|is)\s+"
            r"(?:wrong|incorrect|not\s+right|inaccurate))\b"
        ),
    ),
    (
        "zh_retraction",
        re.compile(
            # \u521a\u624d\u8bf4\u9519\u4e86 / \u6211\u641e\u9519\u4e86 / \u4e0d\u5bf9\u4e0d\u5bf9 / \u7b97\u6211\u6ca1\u8bf4 /
            # \u53d6\u6d88\u4e4b\u524d / \u66f4\u6b63\u4e00\u4e0b (just-said-wrong / I-got-it-wrong /
            # no-no / strike-my-words / cancel-prior / let-me-correct).
            "(?:"
            "\u521a\u624d\u8bf4\u9519|\u6211\u641e\u9519|\u4e0d\u5bf9\u4e0d\u5bf9|\u7b97\u6211\u6ca1\u8bf4|"
            "\u53d6\u6d88\u4e4b\u524d|\u66f4\u6b63\u4e00\u4e0b|\u8bf4\u9519\u4e86|\u8bb0\u9519\u4e86"
            ")"
        ),
    ),
)


class RetractionDetector:
    """Detects retraction cues in free-form user text.

    The detector is stateless and thread-safe; instantiate once at
    application start and reuse across requests.
    """

    def detect(self, text: str) -> RetractionSignal | None:
        """Return the strongest matching signal, or None if no pattern fires.

        Patterns are tried in catalogue order; the first match wins so
        explicit "I was wrong" beats softer hedges like "actually..."
        when both appear in the same utterance.
        """
        if not text:
            return None
        for label, pattern in _PATTERNS:
            match = pattern.search(text)
            if match is not None:
                return RetractionSignal(label=label, matched_text=match.group(0))
        return None

    def is_retraction(self, text: str) -> bool:
        """Boolean shortcut for callers that do not need the label."""
        return self.detect(text) is not None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetractionTarget:
    """A specific (entity, attribute) the orchestrator should invalidate.

    The write path accumulates these as it admits facts so a later
    retraction signal has a precise list of candidates to close. Plain
    dataclass (not Pydantic) because this lives only inside the write
    path and never crosses a serialization boundary.
    """

    entity: str
    attribute: str


@dataclass(frozen=True)
class RetractionOutcome:
    """Result of one orchestrator pass.

    - signal is None when the input did not look like a
    retraction; downstream code uses this as the branch condition.
    - invalidated lists the targets whose active row was actually
    closed. A target the caller passed in but which had no active row
    at invalidation time is absent here.
    """

    signal: RetractionSignal | None
    invalidated: tuple[RetractionTarget, ...] = ()


class RetractionOrchestrator:
    """Bridges retraction detection to MemoryWriterTools.invalidate_unit.

    The detector alone only tells us *that* the speaker is taking
    something back; deciding *what* to take back is the orchestrator's
    job. Because the write path is the only component that knows which
    entity/attribute pairs have just been written, we let the caller
    pass that list in explicitly. This keeps the orchestrator stateless
    and trivially testable.
    """

    def __init__(
        self,
        detector: RetractionDetector,
        writer_tools: MemoryWriterTools,
        *,
        emitter: MemoryEventEmitter | None = None,
    ) -> None:
        self._detector = detector
        self._writer = writer_tools
        # Optional hot-path event emitter. Each detected retraction
        # publishes a RETRACTION_FIRED event so the evolution control
        # plane can correlate user corrections with subsequent recall
        # hits/misses; emission is best-effort and non-blocking.
        self._emitter = emitter or MemoryEventEmitter()

    def process(
        self,
        text: str,
        recent_targets: list[RetractionTarget] | tuple[RetractionTarget, ...],
        *,
        valid_to: float | None = None,
    ) -> RetractionOutcome:
        """Detect retraction in text and invalidate recent_targets.

        recent_targets is the caller's claim of "this is what the
        speaker most recently committed". If the detector does not fire,
        nothing is touched. Otherwise each target's active row is closed
        via the writer tools; targets without a current active row are
        silently skipped (they were already closed or never existed).
        """
        signal = self._detector.detect(text)
        if signal is None:
            return RetractionOutcome(signal=None)

        invalidated: list[RetractionTarget] = []
        for target in recent_targets:
            closed = self._writer.invalidate_unit(
                target.entity,
                target.attribute,
                valid_to=valid_to,
            )
            if closed:
                invalidated.append(target)

        self._emitter.emit(
            EvolutionEventType.RETRACTION_FIRED,
            target="retraction_orchestrator",
            payload={
                "label": signal.label,
                "matched_text": signal.matched_text,
                "invalidated": [
                    {"entity": t.entity, "attribute": t.attribute} for t in invalidated
                ],
            },
            metrics={
                "candidates": float(len(recent_targets)),
                "invalidated": float(len(invalidated)),
            },
            namespace=getattr(self._writer, "namespace", "default"),
        )
        return RetractionOutcome(signal=signal, invalidated=tuple(invalidated))
