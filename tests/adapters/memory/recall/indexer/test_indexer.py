"""Multi-axis indexer tests: registry, axes, prefilter."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.backends.base import EntityStateView
from houyi.adapters.memory.recall.indexer import (
    AxisQuery,
    AxisRegistry,
    AxisResult,
    MultiAxisPrefilter,
)
from houyi.adapters.memory.recall.indexer.axes import EntityAxis, TimeAxis
from houyi.adapters.memory.types import EntityStateRecord


class _StubView(EntityStateView):
    def __init__(self, rows: list[EntityStateRecord]) -> None:
        self._rows = rows

    def get_active(
        self, namespace: str, entity: str | None = None, attribute: str | None = None
    ) -> list[EntityStateRecord]:
        result = [r for r in self._rows if r.namespace == namespace]
        if entity is not None:
            result = [r for r in result if r.entity == entity]
        if attribute is not None:
            result = [r for r in result if r.attribute == attribute]
        return result

    def get_as_of(
        self, namespace: str, as_of: float, entity: str | None = None
    ) -> list[EntityStateRecord]:
        return []

    def list_entities(self, namespace: str) -> list[str]:
        return []

    def get_history(
        self, namespace: str, entity: str, attribute: str | None = None
    ) -> list[EntityStateRecord]:
        return []

    def invalidate(self, namespace: str, entity: str, attribute: str) -> None:
        pass

    def upsert(self, record: EntityStateRecord) -> None:
        pass

    def list_conflicted_triples(self, namespace: str | None = None) -> list[tuple[str, str, str]]:
        return []

    def supersede(
        self, namespace, entity, attribute, *, keep_state_id, valid_to
    ) -> tuple[int, int]:
        return (0, 0)


def _record(entity: str, attr: str, value: str, updated_at: float = 0.0) -> EntityStateRecord:
    r = EntityStateRecord(namespace="default", entity=entity, attribute=attr, value=value)
    object.__setattr__(r, "updated_at", updated_at)
    return r


class TestAxisRegistry:
    def test_register_and_get(self) -> None:
        reg = AxisRegistry()
        view = _StubView([])
        axis = EntityAxis(view)
        reg.register(axis)
        assert reg.get("entity") is axis
        assert "entity" in reg
        assert len(reg) == 1

    def test_register_empty_name_raises(self) -> None:
        reg = AxisRegistry()

        class _BadAxis:
            name = ""

            async def query(self, q, *, deadline_ms=None):
                return AxisResult(axis="", matched_ids=frozenset())

        with pytest.raises(ValueError):
            reg.register(_BadAxis())

    def test_names_sorted(self) -> None:
        reg = AxisRegistry()
        view = _StubView([])
        reg.register(EntityAxis(view))
        reg.register(TimeAxis(view))
        assert reg.names() == ["entity", "time"]


class TestEntityAxis:
    async def test_returns_matching_ids(self) -> None:
        view = _StubView(
            [
                _record("alice", "phone", "1234"),
                _record("alice", "email", "a@b.com"),
                _record("bob", "phone", "5678"),
            ]
        )
        axis = EntityAxis(view)
        result = await axis.query(AxisQuery(axis="entity", key="alice"))
        assert result.axis == "entity"
        assert "alice:phone:1234" in result.matched_ids
        assert "alice:email:a@b.com" in result.matched_ids
        assert "bob:phone:5678" not in result.matched_ids

    async def test_empty_when_no_match(self) -> None:
        view = _StubView([_record("alice", "phone", "1234")])
        axis = EntityAxis(view)
        result = await axis.query(AxisQuery(axis="entity", key="nobody"))
        assert len(result.matched_ids) == 0


class TestTimeAxis:
    async def test_filters_by_since(self) -> None:
        view = _StubView(
            [
                _record("alice", "phone", "old", updated_at=100.0),
                _record("alice", "phone", "new", updated_at=200.0),
            ]
        )
        axis = TimeAxis(view)
        result = await axis.query(AxisQuery(axis="time", key="", params={"since": 150.0}))
        ids = result.matched_ids
        assert "alice:phone:new" in ids
        assert "alice:phone:old" not in ids

    async def test_filters_by_until(self) -> None:
        view = _StubView(
            [
                _record("alice", "phone", "old", updated_at=100.0),
                _record("alice", "phone", "new", updated_at=200.0),
            ]
        )
        axis = TimeAxis(view)
        result = await axis.query(AxisQuery(axis="time", key="", params={"until": 150.0}))
        ids = result.matched_ids
        assert "alice:phone:old" in ids
        assert "alice:phone:new" not in ids


class TestMultiAxisPrefilter:
    async def test_parallel_query_and_intersect(self) -> None:
        view = _StubView(
            [
                _record("alice", "phone", "1234", updated_at=100.0),
                _record("alice", "email", "a@b.com", updated_at=200.0),
                _record("bob", "phone", "5678", updated_at=100.0),
            ]
        )
        reg = AxisRegistry()
        reg.register(EntityAxis(view))
        reg.register(TimeAxis(view))
        prefilter = MultiAxisPrefilter(reg)

        results = await prefilter.prefilter(
            [
                AxisQuery(axis="entity", key="alice"),
                AxisQuery(axis="time", key="", params={"since": 150.0}),
            ]
        )
        assert len(results) == 2
        intersection = prefilter.intersect(results)
        assert intersection == {"alice:email:a@b.com"}

    async def test_empty_queries(self) -> None:
        reg = AxisRegistry()
        prefilter = MultiAxisPrefilter(reg)
        results = await prefilter.prefilter([])
        assert results == []

    async def test_unknown_axis_skipped(self) -> None:
        reg = AxisRegistry()
        prefilter = MultiAxisPrefilter(reg)
        results = await prefilter.prefilter([AxisQuery(axis="nonexistent", key="x")])
        assert results == []
