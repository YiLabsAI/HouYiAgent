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

    def test_keeps_distinct_values(self, view: SQLiteEntityStateView) -> None:
        """Distinct values are kept (multi-valued set), not superseded.

        Without schema-level cardinality we cannot tell a genuine open set
        from a single-valued attribute that changed, so we keep both and let
        recall MMR de-duplicate. Closing distinct values loses gold facts
        (conv-48/30 regression).
        """
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=200.0)

        report = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert isinstance(report, ConsolidationReport)
        assert report.triples_scanned == 1
        assert report.triples_resolved == 0
        assert report.rows_closed == 0
        assert report.skipped_multi_value == 1
        active = view.get_active("ws", "Andrew", "job")
        assert len(active) == 2

    def test_dedupes_same_value(self, view: SQLiteEntityStateView) -> None:
        """Exact duplicate values are de-duplicated (keep newest)."""
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "banker", valid_from=200.0)

        report = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert report.triples_resolved == 1
        assert report.rows_closed == 1
        active = view.get_active("ws", "Andrew", "job")
        assert len(active) == 1

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
        view.upsert("ws", "Andrew", "job", "banker", valid_from=200.0)

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
        assert report.rows_closed == 0
        assert report.skipped_multi_value == 2
        assert len(view.get_active("ws-a", "Andrew", "job")) == 2
        assert len(view.get_active("ws-b", "Bob", "city")) == 2

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
        """Three duplicate versions: only the newest stays (dedupe)."""
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "banker", valid_from=200.0)
        view.upsert("ws", "Andrew", "job", "banker", valid_from=300.0)

        report = EntityStateConsolidator(view).consolidate(namespace="ws")

        assert report.rows_closed == 2
        active = view.get_active("ws", "Andrew", "job")
        assert len(active) == 1
        assert active[0].value == "banker"
