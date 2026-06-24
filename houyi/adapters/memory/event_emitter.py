"""Hot-path event emitter for the memory pipeline.

Memory hot-path components (recall orchestrator, IDK guard, retraction
orchestrator, ingestor, engine) need to publish trace signals to the
evolution control plane without taking on its full dependency tree or
blocking on its I/O. This module provides a thin, never-raises wrapper
around EvolutionClient that:

- Accepts None as a no-op so production wiring remains optional and
  every existing call site keeps working when the control plane is not
  configured.
- Swallows every exception out of EvolutionClient.emit_event. Hot-path
  callers must never have a memory write or recall fail because the
  evolution side-channel had a transient backend issue.
- Keeps the construction signature identical to a regular emit so tests
  can substitute a recording double via duck typing.

The emitter is deliberately not async: EvolutionClient.emit_event is
already non-blocking (writes to an event log abstraction, typically
SQLite or in-memory), and forcing every memory call site through an
await would invade their existing sync code paths (retraction, IDK
guard) for no benefit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from houyi.application.evolution.client import EvolutionClient
from houyi.application.evolution.event_log import EvolutionEventLog
from houyi.application.evolution.events import EvolutionEvent, EvolutionEventType

logger = logging.getLogger(__name__)


class MemoryEventEmitter:
    """Non-blocking emitter that forwards memory hot-path events.

    Construct with client=None to disable emission entirely; all
    emit calls then return immediately without touching anything.
    Production wiring passes a real EvolutionClient; tests pass a
    recording fake.
    """

    __slots__ = ("_client",)

    def __init__(self, client: EvolutionClient | None = None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def event_log(self) -> EvolutionEventLog | None:
        """Expose the backing event log for replay-style signal mining.

        Returns None when emission is disabled (no client), so callers can
        fall back to a signal-free path instead of inventing queries.
        """
        return self._client.event_log if self._client is not None else None

    def emit(
        self,
        event_type: EvolutionEventType,
        *,
        target: str,
        payload: Mapping[str, object] | None = None,
        metrics: Mapping[str, float] | None = None,
        namespace: str = "default",
    ) -> None:
        """Forward one event; never raises.

        Any exception from the downstream client is logged at debug level
        and swallowed. The hot path must remain unaffected by control
        plane backpressure or transient storage failures.
        """
        if self._client is None:
            return
        try:
            event = EvolutionEvent(
                event_type=event_type,
                target=target,
                payload=dict(payload or {}),
                metrics=dict(metrics or {}),
                namespace=namespace,
            )
            self._client.emit_event(event)
        except Exception:
            logger.debug(
                "evolution emit_event swallowed for %s/%s",
                event_type.value,
                target,
                exc_info=True,
            )


def disabled_emitter() -> MemoryEventEmitter:
    """Convenience factory for explicit no-op emitters in tests / defaults."""
    return MemoryEventEmitter(client=None)


__all__ = ["MemoryEventEmitter", "disabled_emitter"]
