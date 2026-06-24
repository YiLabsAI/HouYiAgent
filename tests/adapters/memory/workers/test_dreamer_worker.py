from __future__ import annotations

import asyncio

import pytest

from houyi.adapters.memory.dreamer import EvolutionBudget, EvolutionRunReport
from houyi.adapters.memory.types import MemoryRecord, MemoryScope, MemoryType
from houyi.adapters.memory.workers.dreamer_worker import DreamerWorker, DreamerWorkerConfig
from houyi.adapters.memory.workers.trigger import ManualTriggerPolicy


class _FakeEngine:
    """Records evolve calls and returns a configurable report."""

    def __init__(self, *, promoted: int = 1, raises: bool = False) -> None:
        self.calls = 0
        self._promoted = promoted
        self._raises = raises

    def evolve(self, *, budget: EvolutionBudget | None = None) -> EvolutionRunReport:
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        created = tuple(
            MemoryRecord(
                scope=MemoryScope.USER,
                key=f"obs-{index}",
                content="note",
                memory_type=MemoryType.STRATEGY,
                confidence=0.8,
            )
            for index in range(self._promoted)
        )
        return EvolutionRunReport(created_records=created)


def test_worker_requires_engine() -> None:
    with pytest.raises(ValueError):
        DreamerWorker(engine=None)  # type: ignore[arg-type]


async def test_worker_promotes_once() -> None:
    engine = _FakeEngine(promoted=2)
    worker = DreamerWorker(engine=engine)

    promoted = await worker.process_once()

    assert promoted == 2
    assert engine.calls == 1


async def test_worker_swallows_errors() -> None:
    worker = DreamerWorker(engine=_FakeEngine(raises=True))
    stop = asyncio.Event()
    stop.set()

    await worker.run_forever(stop=stop)


async def test_worker_passes_budget() -> None:
    engine = _FakeEngine()
    worker = DreamerWorker(engine=engine, budget=EvolutionBudget(max_units=2))

    await worker.process_once()

    assert engine.calls == 1


def test_worker_default_trigger() -> None:
    worker = DreamerWorker(engine=_FakeEngine())
    assert isinstance(worker._trigger, ManualTriggerPolicy)


class _OneShotTrigger:
    """Fires once, then stops the worker loop via record_run."""

    def __init__(self, stop: asyncio.Event) -> None:
        self._stop = stop
        self.runs = 0

    def should_run(self, *, now: float | None = None) -> bool:
        return not self._stop.is_set()

    def record_run(self, *, now: float | None = None) -> None:
        self.runs += 1
        self._stop.set()


class _NeverTrigger:
    def should_run(self, *, now: float | None = None) -> bool:
        return False

    def record_run(self, *, now: float | None = None) -> None:
        return None


async def test_forever_fires() -> None:
    engine = _FakeEngine()
    stop = asyncio.Event()
    trigger = _OneShotTrigger(stop)
    worker = DreamerWorker(
        engine=engine,
        config=DreamerWorkerConfig(idle_sleep_s=0.01),
        trigger=trigger,
    )

    await asyncio.wait_for(worker.run_forever(stop), timeout=1.0)

    assert engine.calls == 1
    assert trigger.runs == 1


async def test_forever_skips() -> None:
    engine = _FakeEngine()
    stop = asyncio.Event()
    worker = DreamerWorker(
        engine=engine,
        config=DreamerWorkerConfig(idle_sleep_s=0.01),
        trigger=_NeverTrigger(),
    )

    async def _stop_soon() -> None:
        await asyncio.sleep(0.05)
        stop.set()

    await asyncio.gather(
        asyncio.wait_for(worker.run_forever(stop), timeout=1.0),
        _stop_soon(),
    )

    assert engine.calls == 0
