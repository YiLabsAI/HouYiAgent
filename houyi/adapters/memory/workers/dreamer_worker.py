"""Dreamer worker — drive memory evolution off the hot path.

The recall and write paths only ever emit trace events; the actual memory
evolution runs here, in a background worker, so the hot path is never blocked
by sampling, reflection, or recall replay. The worker is off by default: a
caller must construct it and drive process_once / run_forever explicitly,
mirroring ExtractorWorker and EmbeddingBackfillWorker.

Worker shape:

- process_once runs one bounded evolution pass and returns the number of
  records promoted. It is the explicit, always-on manual entry point.
- run_forever drives a loop that consults the TriggerPolicy each cycle and
  only spends a pass when the policy fires, sleeping between checks.

Failures are logged and swallowed: a transient evolution error must never
take down the surrounding process.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Protocol

from houyi.adapters.memory.dreamer import EvolutionBudget, EvolutionRunReport
from houyi.adapters.memory.workers.trigger import ManualTriggerPolicy, TriggerPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DreamerWorkerConfig:
    """Tunables for the dreamer worker."""

    idle_sleep_s: float = 30.0
    """Seconds to sleep between evolution passes in run_forever."""

    fail_sleep_s: float = 60.0
    """Seconds to sleep after a pass raised before re-running."""


class _EvolvableEngine(Protocol):
    """The slice of MemoryEngine this worker depends on."""

    def evolve(self, *, budget: EvolutionBudget | None = ...) -> EvolutionRunReport: ...


class DreamerWorker:
    """Run bounded memory evolution passes in the background."""

    def __init__(
        self,
        *,
        engine: _EvolvableEngine,
        budget: EvolutionBudget | None = None,
        config: DreamerWorkerConfig | None = None,
        trigger: TriggerPolicy | None = None,
    ) -> None:
        if engine is None:
            raise ValueError("engine is required")
        self._engine = engine
        self._budget = budget
        self._config = config or DreamerWorkerConfig()
        # Default trigger never fires, so run_forever stays idle until a
        # caller wires a real policy (e.g. HybridTriggerPolicy). process_once
        # ignores the trigger entirely: it is the explicit manual path.
        self._trigger = trigger or ManualTriggerPolicy()

    async def process_once(self) -> int:
        """Run one evolution pass; return the count of promoted records."""
        report = await asyncio.to_thread(self._run_pass)
        return len(report.created_records)

    def _run_pass(self) -> EvolutionRunReport:
        return self._engine.evolve(budget=self._budget)

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        """Loop until stop is set, firing a pass only when the trigger says so.

        Each cycle consults the TriggerPolicy. When it fires, one pass runs and
        the trigger's schedule/pressure state is reset via record_run. The loop
        then sleeps idle_sleep_s before re-checking (fail_sleep_s after an
        error) so the trigger is polled cheaply without busy-waiting.
        """
        signal = stop or asyncio.Event()
        while not signal.is_set():
            sleep_s = self._config.idle_sleep_s
            try:
                if self._trigger.should_run():
                    await self.process_once()
                    self._trigger.record_run()
            except Exception:
                logger.exception("dreamer worker pass failed; backing off")
                sleep_s = self._config.fail_sleep_s
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(signal.wait(), timeout=sleep_s)


__all__ = ["DreamerWorker", "DreamerWorkerConfig"]
