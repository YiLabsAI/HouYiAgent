"""Detect emphasis cues and tag turns for prioritized recall.

 ALL-CAPS shouts, repeated punctuation ("!!!"), and explicit
emphasis words ("important", zh: zhong-yao / qian-wan) all signal that the
speaker thinks this turn matters more than average. The emphasis
detector does not write its own MemoryRecord — that's the
explicit-pin detector's job. Instead it stamps a hint tag on the
in-flight turn and exposes last_signal so downstream consumers
(the L1 extractor's confidence boost, retrieval rerank, audit log)
can pick it up.

The detector is intentionally side-effect-free on storage: it only
mutates turn.metadata (in place) so callers that don't care can
ignore the signal entirely. Mutation is safe because RawTurn
is a pydantic model and metadata is a plain dict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from houyi.adapters.memory.types import RawTurn


class EmphasisKind(str, Enum):
    """Why the detector fired. Stored as a string in turn.metadata."""

    ALL_CAPS = "all_caps"
    REPEATED_PUNCT = "repeated_punct"
    KEYWORD = "keyword"


@dataclass(frozen=True)
class EmphasisSignal:
    """A single detected emphasis cue."""

    kind: EmphasisKind
    score: float
    """Magnitude in [0, 1]. Reflects how strong the cue is — useful
 for the rerank weight matrix at .
 """
    fragment: str
    """The text fragment that triggered the match (truncated to 80
 chars). Surfaced for audit logging only.
 """


_KEYWORDS = (
    "important",
    "critical",
    "\u91cd\u8981",  # zhong-yao
    "\u5343\u4e07",  # qian-wan
    "\u52a1\u5fc5",  # wu-bi
    "\u91cd\u70b9",  # zhong-dian
    "must remember",
)
_KEYWORD_PATTERN = re.compile("|".join(re.escape(k) for k in _KEYWORDS), re.IGNORECASE)
_REPEATED_PUNCT = re.compile("([!?\uff01\uff1f])\\1{1,}")
# At least two consecutive A-Z words, joined by spaces. Length ≥ 6 chars
# total to avoid e.g. "OK GO" false positives.
_ALL_CAPS = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b")


class EmphasisDetector:
    """TurnDetector that tags emphasized turns in place."""

    METADATA_KEY = "emphasis"
    """Name of the turn.metadata field this detector writes."""

    def __init__(self, *, min_score: float = 0.3) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        self._min_score = min_score
        self._last_signal: EmphasisSignal | None = None

    @property
    def last_signal(self) -> EmphasisSignal | None:
        """Most recent emphasis signal, or None for non-emphasized turns."""
        return self._last_signal

    def detect(self, turn: RawTurn) -> None:
        signal = self._scan(turn.content)
        self._last_signal = signal
        if signal is None:
            return
        # Mutate in-place: callers that pass a fresh RawTurn each time
        # will never see cross-call contamination, and any downstream
        # consumer that already holds the same instance gets the tag.
        turn.metadata[self.METADATA_KEY] = signal.kind.value
        turn.metadata[f"{self.METADATA_KEY}_score"] = f"{signal.score:.2f}"

    def _scan(self, text: str) -> EmphasisSignal | None:
        stripped = text.strip()
        if not stripped:
            return None

        # Order matters: keyword > repeated punct > all-caps. A turn
        # like "IMPORTANT!!!" should attribute to the keyword cue, not
        # the punctuation, so the audit trail picks the most semantic
        # reason first.
        kw = _KEYWORD_PATTERN.search(stripped)
        if kw is not None:
            return _maybe(EmphasisSignal(EmphasisKind.KEYWORD, 0.7, kw.group(0)), self._min_score)

        rp = _REPEATED_PUNCT.search(stripped)
        if rp is not None:
            return _maybe(
                EmphasisSignal(EmphasisKind.REPEATED_PUNCT, 0.5, rp.group(0)),
                self._min_score,
            )

        cap = _ALL_CAPS.search(stripped)
        if cap is not None:
            # Score scales with the length of the capitalized run, capped at
            # 1.0. A 2-word shout earns 0.4; an 8+ word tirade saturates at 1.0.
            n_chars = len(cap.group(0))
            score = min(1.0, 0.3 + 0.05 * n_chars)
            return _maybe(
                EmphasisSignal(EmphasisKind.ALL_CAPS, score, cap.group(0)),
                self._min_score,
            )

        return None


def _maybe(signal: EmphasisSignal, threshold: float) -> EmphasisSignal | None:
    """Drop the signal if it doesn't clear the configured min_score."""
    return signal if signal.score >= threshold else None


__all__ = ["EmphasisDetector", "EmphasisKind", "EmphasisSignal"]
