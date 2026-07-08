"""Turn-writer facade — synchronous fast path + async extract scheduling.

Gives ingestion a single entry point that respects the layered write contract:

- L0 (sync) — every turn lands in raw_turn_log verbatim, no matter what
  comes after. This is the durable spine of the system.
- L0+ (sync, opt-in) — when an explicit-pin / emphasis / structured command
  detector fires, the same call may also write a fast entity-state row
  directly. The hooks live here but are no-ops until wired.
- L1 (async) — every L0 write enqueues an extract job in extract_queue so
  the L1 extractor can later pull a batch, run its LLM pipeline, and produce
  AtomicFact rows.

The facade does not spawn workers or schedule asyncio tasks; it only persists
 the turn and parks a queue row. Worker policy is the worker's job. This
 keeps the write path trivially testable and re-entrant.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from houyi.adapters.memory.triggers import ExtractTrigger
from houyi.adapters.memory.types import RawTurn


class TurnDetector(Protocol):
    """Optional sync detector hook called inside TurnWriter.fast_path.

    A detector inspects a single turn and may emit zero or more side-effects
    (e.g. writing an entity-state row, tagging the turn). Detectors must be
    cheap and side-effect-only; they may not block on network calls. The
    detector contract is intentionally open so ExplicitPinDetector /
    EmphasisDetector can plug in without churning the facade.
    """

    def detect(self, turn: RawTurn) -> None:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a single fast-path write.

    Returned to callers (the ingestor or the chat adapter) so they can
    surface the assigned turn_index for ordering and the queue id for
    tracing.
    """

    turn: RawTurn
    queue_id: str | None
    """None when schedule_extract=False or when an L1 trigger voted to skip
    this turn.
    """

    detectors_fired: tuple[str, ...] = ()
    """Names of detectors that observed this turn. Empty when no detectors are
    registered or none matched.
    """

    extract_skipped: bool = False
    """True when the L1 trigger policy rejected the turn. queue_id is None in
    that case; the L0 row is still durable.
    """


class _RawTurnSink(Protocol):
    """Minimal storage protocol the writer relies on.

    Defining a Protocol lets tests inject an in-memory fake without spinning
    up SQLite. The real implementation is SQLiteMemoryBackend.
    """

    def append_raw_turn(self, turn: RawTurn) -> RawTurn: ...
    def enqueue_extract(self, turn: RawTurn) -> str: ...
    def append_raw_turn_and_enqueue(self, turn: RawTurn) -> tuple[RawTurn, str]: ...


class TurnWriter:
    """Single entry point for memory writes.

    Construction takes a backend that satisfies _RawTurnSink and an optional
    iterable of detectors. Each call to fast_path runs L0 first, then
    detectors (L0+), then queues L1. The order matters: the durable L0 row
    is never lost even if a detector raises, because we materialize the row
    before the detector loop.
    """

    def __init__(
        self,
        backend: _RawTurnSink,
        *,
        detectors: Iterable[TurnDetector] | None = None,
        extract_trigger: ExtractTrigger | None = None,
    ) -> None:
        """Construct a turn-writer facade.

        Args:
            backend: storage handle satisfying _RawTurnSink.
            detectors: optional sync L0+ hooks that observe each turn.
            extract_trigger: optional L1 gate. When None we install
                triggers.default_extract_policy, which rejects very short
                and non-conversational turns; pass a custom composite (or
                triggers.all_of() for "always extract") to override.
        """
        if backend is None:
            raise ValueError("backend is required")
        self._backend = backend
        self._detectors: tuple[TurnDetector, ...] = tuple(detectors or ())
        if extract_trigger is None:
            from houyi.adapters.memory.triggers import default_extract_policy

            extract_trigger = default_extract_policy()
        self._extract_trigger: ExtractTrigger = extract_trigger

    def fast_path(
        self,
        turn: RawTurn,
        *,
        schedule_extract: bool = True,
    ) -> WriteResult:
        """Persist a turn through the layered write tiers.

        Args:
            turn: the conversation turn to log. turn_index may be left unset;
                the storage layer assigns it.
            schedule_extract: if False, skip the L1 enqueue. Useful for replays
                / backfills where the caller plans to drive the extractor
                directly.

        Returns:
            A WriteResult carrying the persisted turn (with assigned
            turn_index), the queue id, and the names of detectors that fired.
        """
        # Detectors run BEFORE the L0 write so any metadata they stamp
        # (e.g. emphasis tags) is persisted with the row. Detectors are
        # contract-bound to be best-effort: a raise here only affects the
        # offending detector, never L0 durability.
        fired: list[str] = []
        for detector in self._detectors:
            try:
                detector.detect(turn)
            except Exception:
                # Detectors are best-effort. A buggy detector must not corrupt
                # the L0 path or the L1 enqueue. We log via the detector's own
                # logger; the facade stays silent so callers see a deterministic
                # side-effect contract.
                continue
            fired.append(type(detector).__name__)

        queue_id: str | None = None
        skipped = False
        if schedule_extract and self._extract_trigger.should_extract(turn):
            # Fold the L0 insert and the L1 enqueue into a single
            # transaction/commit instead of two independent round trips.
            persisted, queue_id = self._backend.append_raw_turn_and_enqueue(turn)
        else:
            persisted = self._backend.append_raw_turn(turn)
            if schedule_extract:
                skipped = True

        return WriteResult(
            turn=persisted,
            queue_id=queue_id,
            detectors_fired=tuple(fired),
            extract_skipped=skipped,
        )

    def schedule_extract(self, turn: RawTurn) -> str:
        """Enqueue L1 extraction for an already-persisted turn.

        Useful when an upstream system wrote to raw_turn_log directly (e.g.
        the dreamer replaying historical traces) and only needs the L1
        hand-off. Idempotent on turn.turn_id.
        """
        return self._backend.enqueue_extract(turn)


__all__ = ["TurnDetector", "TurnWriter", "WriteResult"]
