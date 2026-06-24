from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.types import Certainty


@pytest.fixture
def view(tmp_path) -> Iterator[SQLiteEntityStateView]:
    """Fresh on-disk SQLite-backed view for each test."""
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    try:
        yield SQLiteEntityStateView(backend)
    finally:
        backend.close()


class TestSQLiteEntityStateViewUpsert:
    """Closed-open interval semantics on upsert."""

    def test_first_upsert_active(self, view: SQLiteEntityStateView) -> None:
        record = view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        assert record.is_active is True
        assert record.valid_from == 100.0

        active = view.get_active("ws", "user", "city")
        assert len(active) == 1
        assert active[0].value == "Beijing"

    def test_second_upsert_appends(self, view: SQLiteEntityStateView) -> None:
        """Write path is append-only: a second upsert leaves the prior active
        row open. Closing stale single-valued rows is the consolidator's job
        (the supersede pass), not upsert's."""
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)

        active = view.get_active("ws", "user", "city")
        assert len(active) == 2
        assert active[0].value == "Shanghai"
        assert active[1].value == "Beijing"

        history = view.get_history("ws", "user", "city")
        assert [r.value for r in history] == ["Shanghai", "Beijing"]
        assert history[0].valid_to is None
        assert history[1].valid_to is None

    def test_namespace_isolation(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws-a", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws-b", "user", "city", "Tokyo", valid_from=100.0)

        assert view.get_active("ws-a", "user", "city")[0].value == "Beijing"
        assert view.get_active("ws-b", "user", "city")[0].value == "Tokyo"

    @pytest.mark.parametrize(
        "namespace,entity,attribute",
        [("", "u", "a"), ("ws", "", "a"), ("ws", "u", "")],
    )
    def test_empty_key_rejected(
        self,
        view: SQLiteEntityStateView,
        namespace: str,
        entity: str,
        attribute: str,
    ) -> None:
        with pytest.raises(ValueError):
            view.upsert(namespace, entity, attribute, "v", valid_from=1.0)

    def test_qualifiers_round_trip(self, view: SQLiteEntityStateView) -> None:
        view.upsert(
            "ws",
            "user",
            "lives_in",
            "Beijing",
            valid_from=100.0,
            qualifiers={"since": "2022", "with": "family"},
        )
        active = view.get_active("ws", "user", "lives_in")
        assert active[0].qualifiers == {"since": "2022", "with": "family"}

    def test_certainty_persisted(self, view: SQLiteEntityStateView) -> None:
        view.upsert(
            "ws",
            "user",
            "city",
            "Beijing",
            valid_from=100.0,
            certainty=Certainty.PROBABLE,
        )
        assert view.get_active("ws", "user", "city")[0].certainty is Certainty.PROBABLE


class TestSQLiteEntityStateViewInvalidate:
    """Explicit retraction without successor."""

    def test_invalidate_closes_active_row(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        assert view.invalidate("ws", "user", "city", valid_to=200.0) is True

        assert view.get_active("ws", "user", "city") == []
        history = view.get_history("ws", "user", "city")
        assert len(history) == 1
        assert history[0].valid_to == 200.0

    def test_invalidate_no_active(self, view: SQLiteEntityStateView) -> None:
        assert view.invalidate("ws", "user", "city", valid_to=200.0) is False


class TestSQLiteEntityStateViewQuery:
    """Read-side fast paths."""

    def test_active_all_attrs(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "job", "engineer", valid_from=110.0)

        active = view.get_active("ws", "user")
        attrs = {r.attribute: r.value for r in active}
        assert attrs == {"city": "Beijing", "job": "engineer"}

    def test_active_excludes_closed(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)
        view.upsert("ws", "user", "city", "Hangzhou", valid_from=300.0)

        active = view.get_active("ws", "user", "city")
        assert len(active) == 3
        assert active[0].value == "Hangzhou"
        assert active[-1].value == "Beijing"

    def test_as_of_at_instant(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)
        view.upsert("ws", "user", "city", "Hangzhou", valid_from=300.0)

        assert view.get_as_of("ws", "user", 50.0, "city") == []
        assert view.get_as_of("ws", "user", 100.0, "city")[0].value == "Beijing"
        assert view.get_as_of("ws", "user", 199.0, "city")[0].value == "Beijing"
        assert view.get_as_of("ws", "user", 250.0, "city")[0].value == "Shanghai"
        assert view.get_as_of("ws", "user", 9_999.0, "city")[0].value == "Hangzhou"

    def test_as_of_boundary(self, view: SQLiteEntityStateView) -> None:
        # With append-only, both rows remain active.
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)

        rows = view.get_as_of("ws", "user", 200.0, "city")
        assert [r.value for r in rows] == ["Shanghai", "Beijing"]

    def test_history_newest_first(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)
        view.upsert("ws", "user", "city", "Hangzhou", valid_from=300.0)

        history = view.get_history("ws", "user", "city")
        assert [r.valid_from for r in history] == [300.0, 200.0, 100.0]

    def test_active_unknown_empty(self, view: SQLiteEntityStateView) -> None:
        assert view.get_active("ws", "ghost", "city") == []
        assert view.get_active("ws", "ghost") == []


class TestSQLiteEntityStateViewMonotonic:
    """Explicit timestamps still raise on backdating; wall-clock ones auto-bump."""

    def test_explicit_backdated_reconciles(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=200.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=100.0)

        history = view.get_history("ws", "user", "city")
        assert len(history) == 2
        assert history[0].value == "Beijing"
        assert history[0].valid_from == 200.0
        assert history[0].valid_to is None
        assert history[1].value == "Shanghai"
        assert history[1].valid_from == 100.0
        assert history[1].valid_to is None

    def test_explicit_equal_auto_bumps(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        rec = view.upsert("ws", "user", "city", "Shanghai", valid_from=100.0)
        # Same explicit ts is treated as a granularity collision and bumped.
        assert rec.valid_from > 100.0
        assert view.get_active("ws", "user", "city")[0].value == "Shanghai"

    def test_wallclock_collision_resolves(self, view: SQLiteEntityStateView) -> None:
        # Three rapid wall-clock writes (valid_from=None) must all succeed.
        for value in ("a", "b", "c"):
            view.upsert("ws", "user", "tag", value)
        active = view.get_active("ws", "user", "tag")
        assert len(active) == 3

    def test_multi_ts_collision(self, view: SQLiteEntityStateView) -> None:
        # 10 writes at the same explicit ts must all succeed via iterative nudge.
        ts = 100.0
        for value in (str(i) for i in range(10)):
            view.upsert("ws", "user", "tag", value, valid_from=ts)
        history = view.get_history("ws", "user", "tag")
        assert len(history) == 10
        # All valid_from values must be unique and >= ts.
        timestamps = [r.valid_from for r in history]
        assert len(set(timestamps)) == 10
        assert all(t >= ts for t in timestamps)


class TestSQLiteEntityStateViewSourceLink:
    """Source unit linkage for provenance lookups."""

    def test_source_round_trip(self, view: SQLiteEntityStateView) -> None:
        view.upsert(
            "ws",
            "user",
            "city",
            "Beijing",
            valid_from=100.0,
            source_unit_id="unit-42",
        )
        assert view.get_active("ws", "user", "city")[0].source_unit_id == "unit-42"


class TestSQLiteEntityStateViewCascade:
    """Cascade valid_to propagation from entity_state to memories."""

    def test_upsert_propagates_valid_to(self, view: SQLiteEntityStateView) -> None:
        from houyi.adapters.memory.types import MemoryProvenance, MemoryRecord, MemoryScope

        # Pre-seed active memory record in the memories table
        # Key must follow entity.attribute.digest pattern
        rec = MemoryRecord(
            record_id="rec-1",
            key="user.city.somehash",
            content="user lives in Beijing",
            scope=MemoryScope.WORKSPACE,
            confidence=1.0,
            valid_from=100.0,
            valid_to=None,
            provenance=MemoryProvenance(source_type="test"),
        )
        view._backend.put(rec)

        # Pre-seed entity state row
        view.upsert("workspace", "user", "city", "Beijing", valid_from=100.0)

        # Act: upsert a new value at valid_from=200.0
        view.upsert("workspace", "user", "city", "Shanghai", valid_from=200.0)

        # Assert: under append-only, the old memory's valid_to remains None
        stored = view._backend.get("user.city.somehash", MemoryScope.WORKSPACE)
        assert stored is not None
        assert stored.valid_to is None

    def test_invalidate_propagates_valid_to(self, view: SQLiteEntityStateView) -> None:
        from houyi.adapters.memory.types import MemoryProvenance, MemoryRecord, MemoryScope

        rec = MemoryRecord(
            record_id="rec-2",
            key="user.city.somehash",
            content="user lives in Beijing",
            scope=MemoryScope.WORKSPACE,
            confidence=1.0,
            valid_from=100.0,
            valid_to=None,
            provenance=MemoryProvenance(source_type="test"),
        )
        view._backend.put(rec)

        # Pre-seed entity state row
        view.upsert("workspace", "user", "city", "Beijing", valid_from=100.0)

        # Act: invalidate at valid_to=300.0
        view.invalidate("workspace", "user", "city", valid_to=300.0)

        # Assert: the memories record should now have valid_to = 300.0!
        stored = view._backend.get("user.city.somehash", MemoryScope.WORKSPACE)
        assert stored is not None
        assert stored.valid_to == 300.0


class TestSQLiteEntityStateSupersede:
    """Supersede: close stale active rows of single-valued attributes."""

    def test_list_finds_duplicates(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=200.0)
        view.upsert("ws", "Andrew", "city", "NYC", valid_from=100.0)

        conflicts = view.list_conflicted_triples("ws")
        assert conflicts == [("ws", "Andrew", "job")]
        # namespace=None scans every namespace.
        assert view.list_conflicted_triples(None) == [("ws", "Andrew", "job")]

    def test_closes_old_keeps_new(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=200.0)

        active = view.get_active("ws", "Andrew", "job")
        keeper = max(active, key=lambda r: r.valid_from)
        closed, _ = view.supersede(
            "ws", "Andrew", "job", keep_state_id=keeper.state_id, valid_to=200.0
        )

        assert closed == 1
        active_after = view.get_active("ws", "Andrew", "job")
        assert len(active_after) == 1
        assert active_after[0].value == "designer"
        banker = view.get_history("ws", "Andrew", "job")[-1]
        assert banker.value == "banker"
        assert banker.valid_to == 200.0

    def test_supersede_idempotent(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=200.0)
        active = view.get_active("ws", "Andrew", "job")
        keeper = max(active, key=lambda r: r.valid_from)
        view.supersede("ws", "Andrew", "job", keep_state_id=keeper.state_id, valid_to=200.0)

        # Second call closes nothing: the valid_to IS NULL guard skips resolved rows.
        closed, _ = view.supersede(
            "ws", "Andrew", "job", keep_state_id=keeper.state_id, valid_to=200.0
        )
        assert closed == 0

    def test_preserves_asof_history(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "Andrew", "job", "banker", valid_from=100.0)
        view.upsert("ws", "Andrew", "job", "designer", valid_from=200.0)
        active = view.get_active("ws", "Andrew", "job")
        keeper = max(active, key=lambda r: r.valid_from)
        view.supersede("ws", "Andrew", "job", keep_state_id=keeper.state_id, valid_to=200.0)

        # At t=150 the banker still held the job; as-of must still resolve to it.
        as_of = view.get_as_of("ws", "Andrew", 150.0, "job")
        assert len(as_of) == 1
        assert as_of[0].value == "banker"
        # At t=250 the designer holds it.
        as_of_late = view.get_as_of("ws", "Andrew", 250.0, "job")
        assert len(as_of_late) == 1
        assert as_of_late[0].value == "designer"

    def test_propagates_to_memories(self, view: SQLiteEntityStateView) -> None:
        """Closing the old entity_state row must close its backing memories row
        (by re-derived record_id) while leaving the successor's row active."""
        from houyi.adapters.memory.fact_identity import fact_record_id
        from houyi.adapters.memory.types import (
            MemoryProvenance,
            MemoryRecord,
            MemoryScope,
        )

        anchor = "turn:1"
        banker_id = fact_record_id("Andrew", "job", "banker", anchor)
        designer_id = fact_record_id("Andrew", "job", "designer", anchor)

        def _seed(record_id: str, value: str) -> None:
            view._backend.put(
                MemoryRecord(
                    record_id=record_id,
                    key=f"Andrew.job.{record_id.split(':')[-1]}",
                    content=f"Andrew job {value}",
                    scope=MemoryScope.WORKSPACE,
                    confidence=1.0,
                    valid_from=100.0,
                    valid_to=None,
                    provenance=MemoryProvenance(source_type="test"),
                )
            )

        _seed(banker_id, "banker")
        _seed(designer_id, "designer")
        view.upsert("workspace", "Andrew", "job", "banker", valid_from=100.0, source_unit_id=anchor)
        view.upsert(
            "workspace", "Andrew", "job", "designer", valid_from=200.0, source_unit_id=anchor
        )

        active = view.get_active("workspace", "Andrew", "job")
        keeper = max(active, key=lambda r: r.valid_from)
        closed, propagated = view.supersede(
            "workspace",
            "Andrew",
            "job",
            keep_state_id=keeper.state_id,
            valid_to=200.0,
        )

        assert closed == 1
        assert propagated == 1
        banker_row = view._backend.get_by_id(banker_id)
        designer_row = view._backend.get_by_id(designer_id)
        assert banker_row is not None and banker_row.valid_to == 200.0
        assert designer_row is not None and designer_row.valid_to is None
