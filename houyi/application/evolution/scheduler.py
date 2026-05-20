from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class EvolutionScheduler:
    interval_seconds: float = 3600.0
    last_run_at: float | None = None

    def should_run(self, *, now: float | None = None, force: bool = False) -> bool:
        if force:
            return True
        current = time.time() if now is None else now
        if self.last_run_at is None:
            return True
        return current - self.last_run_at >= self.interval_seconds

    def mark_run(self, *, now: float | None = None) -> None:
        self.last_run_at = time.time() if now is None else now
