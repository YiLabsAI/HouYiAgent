"""L1 extract triggers — decide when a turn deserves the LLM round-trip.

 the write path always logs every turn at L0 (raw_turn_log),
but L1 extraction is expensive (LLM call + projection) so we gate it
behind a small composable trigger system. Triggers run before
SQLiteMemoryBackend.enqueue_extract so a rejected turn never
hits the queue.

Three built-in triggers cover the common cases:

- MinLengthTrigger — drop tiny / whitespace-only turns
- RoleTrigger — only extract from a configured role set
- RegexBlocklistTrigger — skip turns matching a regex

Combinators:

- all_of — AND semantics; default policy used by the write path
- any_of — OR semantics; useful for mutual-exclusion

Triggers are intentionally pure functions wrapped in lightweight classes
so callers can carry policy as data (e.g. load it from YAML / env)
without a dependency on the write-path module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from houyi.adapters.memory.types import RawTurn


class ExtractTrigger(Protocol):
    """A decision predicate over a single turn.

    Returns True when the turn should be enqueued for L1 extraction,
    False to skip. Implementations must be cheap and side-effect-free
    (no logging beyond debug, no I/O); the trigger is on the synchronous
    write path.
    """

    def should_extract(self, turn: RawTurn) -> bool:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class MinLengthTrigger:
    """Reject turns whose stripped content is shorter than min_chars.

    The default of 4 filters out one-word "ok"/"yes" replies that
    almost never carry standalone facts and would otherwise waste an
    LLM call.
    """

    min_chars: int = 4

    def should_extract(self, turn: RawTurn) -> bool:
        return len(turn.content.strip()) >= self.min_chars


@dataclass(frozen=True)
class RoleTrigger:
    """Restrict extraction to a configured set of roles.

    The frozen frozenset shape lets the dataclass remain hashable
    (callers may want to use triggers as dict keys for trace metadata).
    """

    allowed_roles: frozenset[str] = frozenset({"user", "assistant"})

    def should_extract(self, turn: RawTurn) -> bool:
        return turn.role in self.allowed_roles


class RegexBlocklistTrigger:
    """Reject turns whose content matches *any* of the given patterns.

    Patterns are compiled at construction so the hot path stays free of
    re-compilation cost.
    """

    def __init__(self, patterns: Iterable[str]) -> None:
        self._patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(p, re.IGNORECASE) for p in patterns
        )

    def should_extract(self, turn: RawTurn) -> bool:
        return not any(p.search(turn.content) for p in self._patterns)


class _CompositeTrigger:
    """Shared base for AND / OR combinators (kept private for now)."""

    def __init__(self, triggers: Iterable[ExtractTrigger]) -> None:
        self._triggers = tuple(triggers)


class _AllTrigger(_CompositeTrigger):
    def should_extract(self, turn: RawTurn) -> bool:
        return all(t.should_extract(turn) for t in self._triggers)


class _AnyTrigger(_CompositeTrigger):
    def should_extract(self, turn: RawTurn) -> bool:
        # Empty composite votes False to avoid surprising "no triggers
        # → enqueue everything" semantics.
        if not self._triggers:
            return False
        return any(t.should_extract(turn) for t in self._triggers)


def all_of(*triggers: ExtractTrigger) -> ExtractTrigger:
    """AND-combine triggers. Skips when any member votes False."""
    return _AllTrigger(triggers)


def any_of(*triggers: ExtractTrigger) -> ExtractTrigger:
    """OR-combine triggers. Skips only when every member votes False."""
    return _AnyTrigger(triggers)


def default_extract_policy() -> ExtractTrigger:
    """The default policy used by TurnWriter when none is given.

    Currently composes MinLengthTrigger(4) AND
    RoleTrigger({user, assistant}). This rejects bare
    acknowledgements and system-bus messages without making any
    semantic judgments — the LLM still does the heavy lifting.
    """
    return all_of(MinLengthTrigger(min_chars=4), RoleTrigger())


__all__ = [
    "ExtractTrigger",
    "MinLengthTrigger",
    "RegexBlocklistTrigger",
    "RoleTrigger",
    "all_of",
    "any_of",
    "default_extract_policy",
]
