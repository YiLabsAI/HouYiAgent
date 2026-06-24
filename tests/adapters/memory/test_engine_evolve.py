from __future__ import annotations

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.workers.trigger import ManualTriggerPolicy


class _FakeEmitter:
    """Minimal emitter exposing only the event_log seam evolve() reads."""

    def __init__(self, event_log: object | None) -> None:
        self.event_log = event_log


async def test_start_launches_dreamer() -> None:
    engine = MemoryEngine(MemoryStore(), evolution_trigger=ManualTriggerPolicy())

    await engine.start()
    try:
        task_names = {t.get_name() for t in engine._worker_tasks}
        assert "memory-dreamer-worker" in task_names
    finally:
        await engine.stop()
    assert engine._worker_tasks == []


async def test_start_no_dreamer() -> None:
    engine = MemoryEngine(MemoryStore())

    await engine.start()
    try:
        assert engine._worker_tasks == []
    finally:
        await engine.stop()


def test_evolve_runs_consolidation_first(tmp_path) -> None:
    """engine.evolve runs the deterministic consolidator before reflection.

    The consolidator closes the superseded active row of a single-valued
    attribute; reflection (when wired) then runs over the repaired state. With
    no recall/llm wired, reflection is skipped cleanly and only consolidation
    runs. consolidate=False skips it.
    """
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    view = SQLiteEntityStateView(backend)
    engine = MemoryEngine(
        MemoryStore(backend=backend),
        entity_state=view,
        emitter=_FakeEmitter(None),
    )
    view.upsert("default", "Andrew", "job", "banker", valid_from=100.0)
    view.upsert("default", "Andrew", "job", "designer", valid_from=200.0)

    report = engine.evolve()

    assert report.consolidation is not None
    assert report.consolidation.rows_closed == 1
    assert len(view.get_active("default", "Andrew", "job")) == 1
    # No recall/llm wired -> reflection skipped cleanly.
    assert report.reflection is None

    # consolidate=False skips it: a fresh conflict stays unresolved.
    view.upsert("default", "Andrew", "city", "NYC", valid_from=100.0)
    view.upsert("default", "Andrew", "city", "LA", valid_from=200.0)
    skipped = engine.evolve(consolidate=False)
    assert skipped.consolidation is None
    assert len(view.get_active("default", "Andrew", "city")) == 2
    backend.close()


def test_evolve_reflect_can_disable(tmp_path) -> None:
    """reflect=False skips reflection even when failing queries are supplied."""
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    view = SQLiteEntityStateView(backend)
    engine = MemoryEngine(
        MemoryStore(backend=backend),
        entity_state=view,
        emitter=_FakeEmitter(None),
    )
    report = engine.evolve(reflect=False, failing_queries=["what does Andrew do for fun?"])
    assert report.reflection is None
    backend.close()
