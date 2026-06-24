"""Trigger policies for background memory evolution.

The DreamerWorker can run continuously, but firing an evolution pass on a
fixed timer is wasteful (it competes with the hot path) and naive (it ignores
how much new signal has accumulated). A TriggerPolicy decouples *when* to
evolve from *how* to evolve: the worker asks the policy ``should_run`` each
cycle and only spends a pass when the policy says so.

Three signals drive the default HybridTriggerPolicy:

- schedule: never fire more often than ``min_interval_s`` (a rate floor).
- idle: only fire when the hot path has been quiet for ``idle_threshold_s``
  so evolution does not steal cycles from live writes and recalls.
- failure-pressure: fire regardless of idleness once enough recall failures
  have piled up, because a backlog of misses is exactly when fresh higher-
  level memories are most valuable.

Evolution is OFF by default: the ManualTriggerPolicy never fires, so a worker
wired with it only ever evolves when a caller invokes ``process_once``
explicitly. Production opts in by passing a HybridTriggerPolicy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TriggerPolicy(Protocol):
    """Decide whether a background evolution pass should run now."""

    def should_run(self, *, now: float | None = None) -> bool:
        """Return True when an evolution pass should fire this cycle."""
        ...

    def record_run(self, *, now: float | None = None) -> None:
        """Note that a pass just ran so schedule/pressure state can reset."""
        ...


class ActivityMonitor:
    """Track hot-path activity and recall-failure pressure.

    The monitor is the shared state a TriggerPolicy reads. The hot path feeds
    it: ``record_activity`` on each write/recall and ``record_failure`` on each
    recall miss. It is intentionally tiny and lock-free; updates are plain
    attribute writes cheap enough to sit on the hot path.
    """

    __slots__ = ("_clock", "_failures", "_last_activity")

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_activity = clock()
        self._failures = 0

    def record_activity(self) -> None:
        """Mark the hot path as active as of now."""
        self._last_activity = self._clock()

    def record_failure(self, count: int = 1) -> None:
        """Add to the accumulated recall-failure pressure."""
        if count > 0:
            self._failures += count

    @property
    def failure_count(self) -> int:
        return self._failures

    def reset_failures(self) -> None:
        self._failures = 0

    def idle_seconds(self, *, now: float | None = None) -> float:
        """Seconds elapsed since the last recorded activity."""
        return (now if now is not None else self._clock()) - self._last_activity


class ManualTriggerPolicy:
    """Default policy: never auto-fires.

    A worker wired with this policy evolves only when a caller invokes
    ``process_once`` directly, keeping background evolution off by default.
    """

    def should_run(self, *, now: float | None = None) -> bool:
        return False

    def record_run(self, *, now: float | None = None) -> None:
        return None


@dataclass(frozen=True, slots=True)
class HybridTriggerConfig:
    """Thresholds for the HybridTriggerPolicy."""

    min_interval_s: float = 3600.0
    """Minimum seconds between passes; a rate floor against thrashing."""

    idle_threshold_s: float = 300.0
    """Required hot-path idleness before a scheduled pass may fire."""

    failure_pressure: int = 20
    """Recall-failure count that fires a pass regardless of idleness."""


class HybridTriggerPolicy:
    """Fire on idle-after-interval, or immediately under failure pressure.

    A pass fires when:

    - failure pressure has reached ``failure_pressure`` (overrides the rate
      floor and the idle requirement), or
    - at least ``min_interval_s`` has elapsed since the last pass AND the hot
      path has been idle for ``idle_threshold_s``.
    """

    __slots__ = ("_clock", "_config", "_last_run", "_monitor")

    def __init__(
        self,
        monitor: ActivityMonitor,
        *,
        config: HybridTriggerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if monitor is None:
            raise ValueError("monitor is required")
        self._monitor = monitor
        self._config = config or HybridTriggerConfig()
        self._clock = clock
        self._last_run = clock()

    def should_run(self, *, now: float | None = None) -> bool:
        t = now if now is not None else self._clock()
        cfg = self._config
        under_pressure = self._monitor.failure_count >= cfg.failure_pressure
        if under_pressure:
            return True
        if (t - self._last_run) < cfg.min_interval_s:
            return False
        return self._monitor.idle_seconds(now=t) >= cfg.idle_threshold_s

    def record_run(self, *, now: float | None = None) -> None:
        self._last_run = now if now is not None else self._clock()
        self._monitor.reset_failures()


__all__ = [
    "ActivityMonitor",
    "HybridTriggerConfig",
    "HybridTriggerPolicy",
    "ManualTriggerPolicy",
    "TriggerPolicy",
]
