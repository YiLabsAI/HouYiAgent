from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.resolver import (
    ConflictError,
    MemoryWriterTools,
    MissingActiveError,
)
from houyi.adapters.memory.types import AtomicFact, Certainty


@pytest.fixture
def tools(tmp_path) -> Iterator[MemoryWriterTools]:
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    try:
        view = SQLiteEntityStateView(backend)
        inbox = SQLiteCandidateInbox(backend)
        yield MemoryWriterTools(view, inbox, namespace="ws")
    finally:
        backend.close()


@pytest.fixture
def backend(tmp_path) -> Iterator[SQLiteMemoryBackend]:
    backend = SQLiteMemoryBackend(db_path=tmp_path / "m.db")
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def inbox_view(backend: SQLiteMemoryBackend) -> tuple[SQLiteCandidateInbox, SQLiteEntityStateView]:
    inbox = SQLiteCandidateInbox(backend)
    view = SQLiteEntityStateView(backend)
    return inbox, view


def _fact(
    *,
    subject: str = "user",
    predicate: str = "city",
    obj: str = "Beijing",
    certainty: Certainty = Certainty.CERTAIN,
    valid_from: float | None = 100.0,
    accumulate: bool = False,
) -> AtomicFact:
    return AtomicFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        certainty=certainty,
        source_anchor="chunk-1",
        valid_from=valid_from,
        accumulate=accumulate,
    )


class TestWriteUnit:
    """write_unit: insert when no active row, reject otherwise."""

    def test_inserts_when_empty(self, tools: MemoryWriterTools) -> None:
        decision = tools.write_unit(_fact())
        assert decision.decision == "admitted"
        assert decision.state is not None
        assert decision.state.value == "Beijing"

    def test_conflict_when_active(self, tools: MemoryWriterTools) -> None:
        tools.write_unit(_fact())
        with pytest.raises(ConflictError, match="active row"):
            tools.write_unit(_fact(obj="Shanghai", valid_from=200.0))


class TestUpdateUnit:
    """update_unit: supersede an existing active row."""

    def test_supersedes_active(self, tools: MemoryWriterTools) -> None:
        tools.write_unit(_fact())
        decision = tools.update_unit(_fact(obj="Shanghai", valid_from=200.0))
        assert decision.decision == "admitted"
        active = tools.read_entity_state("user", "city")
        assert len(active) == 2
        assert {a.value for a in active} == {"Shanghai", "Beijing"}

    def test_missing_when_empty(self, tools: MemoryWriterTools) -> None:
        with pytest.raises(MissingActiveError, match="no active row"):
            tools.update_unit(_fact())


class TestInvalidateUnit:
    """invalidate_unit: close the active row or no-op."""

    def test_closes_active(self, tools: MemoryWriterTools) -> None:
        tools.write_unit(_fact())
        assert tools.invalidate_unit("user", "city", valid_to=200.0) is True
        assert tools.read_entity_state("user", "city") == []

    def test_returns_false_when_empty(self, tools: MemoryWriterTools) -> None:
        assert tools.invalidate_unit("user", "city") is False


class TestReadEntityState:
    """read_entity_state: active vs as-of."""

    def test_active_default(self, tools: MemoryWriterTools) -> None:
        tools.write_unit(_fact())
        rows = tools.read_entity_state("user")
        assert len(rows) == 1
        assert rows[0].attribute == "city"

    def test_as_of_historical(self, tools: MemoryWriterTools) -> None:
        tools.write_unit(_fact())
        tools.update_unit(_fact(obj="Shanghai", valid_from=200.0))
        rows = tools.read_entity_state("user", "city", as_of=150.0)
        assert len(rows) == 1
        assert rows[0].value == "Beijing"


class TestVagueRouting:
    """T6: vague facts must never enter the main store."""

    def test_vague_deferred(self, tools: MemoryWriterTools) -> None:
        decision = tools.write_unit(_fact(certainty=Certainty.VAGUE))
        assert decision.decision == "deferred_vague"
        assert decision.candidate_id is not None
        assert tools.read_entity_state("user", "city") == []

    def test_update_vague_deferred(self, tools: MemoryWriterTools) -> None:
        tools.write_unit(_fact())
        decision = tools.update_unit(
            _fact(obj="Shanghai", certainty=Certainty.VAGUE, valid_from=200.0)
        )
        # Active row from prior certain write must not be overwritten.
        assert decision.decision == "deferred_vague"
        active = tools.read_entity_state("user", "city")
        assert active[0].value == "Beijing"

    def test_inbox_lists_candidate(self, inbox_view) -> None:
        inbox, view = inbox_view
        tools = MemoryWriterTools(view, inbox, namespace="ws")
        tools.write_unit(_fact(certainty=Certainty.VAGUE))
        parked = inbox.list_for("ws", "user", "city")
        assert len(parked) == 1
        assert parked[0].object == "Beijing"
        assert parked[0].certainty is Certainty.VAGUE

    def test_inbox_rejects_non_vague(self, inbox_view) -> None:
        inbox, _ = inbox_view
        with pytest.raises(ValueError, match="vague"):
            inbox.add("ws", _fact(certainty=Certainty.CERTAIN))


class TestIngestFact:
    """End-to-end routing convenience entrypoint."""

    def test_first_call_admits(self, tools: MemoryWriterTools) -> None:
        decision = tools.ingest_fact(_fact())
        assert decision.decision == "admitted"

    def test_second_call_supersedes(self, tools: MemoryWriterTools) -> None:
        tools.ingest_fact(_fact())
        decision = tools.ingest_fact(_fact(obj="Shanghai", valid_from=200.0))
        assert decision.decision == "admitted"
        assert decision.state is not None
        assert decision.state.value == "Shanghai"

    def test_vague_call_defers(self, tools: MemoryWriterTools) -> None:
        decision = tools.ingest_fact(_fact(certainty=Certainty.VAGUE))
        assert decision.decision == "deferred_vague"


class TestAccumulate:
    """ingest_fact with accumulate=True appends instead of superseding."""

    def test_first_item_admitted(self, tools: MemoryWriterTools) -> None:
        d = tools.ingest_fact(_fact(predicate="visited", obj="cafe", accumulate=True))
        assert d.decision == "admitted"
        assert d.state is not None
        assert d.state.value == "cafe"

    def test_second_item_appended(self, tools: MemoryWriterTools) -> None:
        tools.ingest_fact(_fact(predicate="visited", obj="cafe", accumulate=True))
        d = tools.ingest_fact(_fact(predicate="visited", obj="park", accumulate=True))
        assert d.decision == "admitted"
        assert d.state is not None
        assert d.state.value == "cafe, park"

    def test_duplicate_skipped(self, tools: MemoryWriterTools) -> None:
        tools.ingest_fact(_fact(predicate="visited", obj="cafe", accumulate=True))
        d = tools.ingest_fact(_fact(predicate="visited", obj="cafe", accumulate=True))
        assert d.decision == "duplicate"
        active = tools.read_entity_state("user", "visited")
        assert active[0].value == "cafe"

    def test_three_items_merged(self, tools: MemoryWriterTools) -> None:
        for place in ["cafe", "park", "shelter"]:
            tools.ingest_fact(_fact(predicate="visited", obj=place, accumulate=True))
        active = tools.read_entity_state("user", "visited")
        assert len(active) == 3
        # The latest one has the full accumulated list
        assert active[0].value == "cafe, park, shelter"

    def test_false_flag_supersedes(self, tools: MemoryWriterTools) -> None:
        tools.ingest_fact(_fact(obj="Beijing"))
        tools.ingest_fact(_fact(obj="Shanghai", valid_from=200.0))
        active = tools.read_entity_state("user", "city")
        assert active[0].value == "Shanghai"


class TestNamespaceGuard:
    """Constructor must reject empty namespace to prevent cross-tenant writes."""

    def test_empty_namespace_rejected(self, inbox_view) -> None:
        inbox, view = inbox_view
        with pytest.raises(ValueError, match="namespace"):
            MemoryWriterTools(view, inbox, namespace="")
