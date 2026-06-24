from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.dreamer_consolidate import (
    ConsolidationReport,
    EntityStateConsolidator,
)


@pytest.fixture
def view(tmp_path) -> Iterator[SQLiteEntityStateView]:
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    try:
        yield SQLiteEntityStateView(backend)
    finally:
        backend.close()


class TestEntityStateConsolidator:
    """Consolidator: scan conflicts, keep newest for single-value, skip accumulate."""

    def test_resolves_single_value(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=200.0)

        report = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert isinstance(report, ConsolidationReport)
        assert report.triples_scanned == 1
        assert report.triples_resolved == 1
        assert report.rows_closed == 1
        assert report.skipped_accumulate == 0
        active = view.get_active("ws", "Andrew", "job")
        assert len(active) == 1
        assert active[0].value == "designer"

    def test_skips_accumulate(self, view: SQLiteEntityStateView) -> None:
        """Accumulate-tagged attributes are open sets: multiple active values
        are expected, so the consolidator must not close any."""
        view.upsert("ws", "Andrew", "hobby", "boardgames", valid_from=100.0, accumulate=True)
        view.upsert(
            "ws",
            "Andrew",
            "hobby",
            "volunteering",
            valid_from=200.0,
            accumulate=True,
        )

        report = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert report.triples_scanned == 1
        assert report.rows_closed == 0
        assert report.skipped_accumulate == 1
        active = view.get_active("ws", "Andrew", "hobby")
        assert {r.value for r in active} == {"boardgames", "volunteering"}

    def test_idempotent(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=200.0)

        EntityStateConsolidator(view).consolidate(namespace="ws")
        second = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert second.triples_scanned == 0
        assert second.rows_closed == 0

    def test_scans_all_namespaces(self, view: SQLiteEntityStateView) -> None:
        """namespace=None sweeps every namespace in one run."""
        view.upsert("ws-a", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws-a", "Andrew", "job", "designer", valid_from=200.0)
        view.upsert("ws-b", "Bob", "city", "NYC", valid_from=100.0)
        view.upsert("ws-b", "Bob", "city", "LA", valid_from=200.0)

        report = EntityStateConsolidator(view).consolidate()

        assert report.triples_scanned == 2
        assert report.rows_closed == 2
        assert view.get_active("ws-a", "Andrew", "job")[0].value == "designer"
        assert view.get_active("ws-b", "Bob", "city")[0].value == "LA"

    def test_mixed_accumulate_conservative(self, view: SQLiteEntityStateView) -> None:
        """If any row of a triple is tagged accumulate, treat it as an open set
        rather than risk closing a legitimate concurrent value."""
        view.upsert("ws", "Andrew", "hobby", "boardgames", valid_from=100.0)
        view.upsert("ws", "Andrew", "hobby", "volunteering", valid_from=200.0, accumulate=True)

        report = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert report.skipped_accumulate == 1
        assert report.rows_closed == 0
        assert len(view.get_active("ws", "Andrew", "hobby")) == 2

    def test_keeps_newest(self, view: SQLiteEntityStateView) -> None:
        """Three versions: the middle one must not survive; only the newest stays."""
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "engineer", valid_from=200.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=300.0)

        report = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert report.rows_closed == 2
        active = view.get_active("ws", "Andrew", "job")
        assert len(active) == 1
        assert active[0].value == "designer"
