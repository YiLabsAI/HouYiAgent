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

    def test_second_upsert_closes_prior(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)

        active = view.get_active("ws", "user", "city")
        assert len(active) == 1
        assert active[0].value == "Shanghai"
        assert active[0].valid_from == 200.0

        history = view.get_history("ws", "user", "city")
        assert [r.value for r in history] == ["Shanghai", "Beijing"]
        assert history[1].valid_to == 200.0
        assert history[0].valid_to is None

    def test_namespace_isolation(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws-a", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws-b", "user", "city", "Tokyo", valid_from=100.0)

        assert view.get_active("ws-a", "user", "city")[0].value == "Beijing"
        assert view.get_active("ws-b", "user", "city")[0].value == "Tokyo"

    def test_backdated_upsert_rejected(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=200.0)
        with pytest.raises(ValueError, match="valid_from"):
            view.upsert("ws", "user", "city", "Shanghai", valid_from=100.0)

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
        assert len(active) == 1
        assert active[0].value == "Hangzhou"

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
        # Closed-open contract: at valid_to itself the row is no longer active.
        view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)

        rows = view.get_as_of("ws", "user", 200.0, "city")
        assert [r.value for r in rows] == ["Shanghai"]

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

    def test_explicit_backdated_raises(self, view: SQLiteEntityStateView) -> None:
        view.upsert("ws", "user", "city", "Beijing", valid_from=200.0)
        with pytest.raises(ValueError, match="valid_from must be"):
            view.upsert("ws", "user", "city", "Shanghai", valid_from=100.0)

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
        assert len(active) == 1
        assert active[0].value == "c"


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
