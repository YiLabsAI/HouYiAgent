"""Tests for SQLiteEventStore and MemoryEvent model."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.types import Certainty, MemoryEvent


@pytest.fixture
def backend(tmp_path) -> Iterator[SQLiteMemoryBackend]:
    b = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    try:
        yield b
    finally:
        b.close()


class TestMemoryEventModel:
    def test_deterministic_id(self) -> None:
        e1 = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor="conv-42:D1:9",
        )
        e2 = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor="conv-42:D1:9",
        )
        assert e1.event_id == e2.event_id
        assert e1.event_id.startswith("evt_")

    def test_requires_nonempty_fields(self) -> None:
        with pytest.raises(ValueError):
            MemoryEvent(
                namespace="default",
                subject="",
                action="watched",
                object="movie",
                timestamp="2019",
                certainty=Certainty.CERTAIN,
                source_anchor="a1",
            )
        with pytest.raises(ValueError):
            MemoryEvent(
                namespace="default",
                subject="Joanna",
                action="",
                object="movie",
                timestamp="2019",
                certainty=Certainty.CERTAIN,
                source_anchor="a1",
            )
        with pytest.raises(ValueError):
            MemoryEvent(
                namespace="default",
                subject="Joanna",
                action="watched",
                object="movie",
                timestamp="",
                certainty=Certainty.CERTAIN,
                source_anchor="a1",
            )

    def test_is_active_default(self) -> None:
        e = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor="a1",
        )
        assert e.is_active is True


class TestSQLiteEventStore:
    def test_add_and_get_event(self, backend: SQLiteMemoryBackend) -> None:
        event = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor="a1",
        )
        stored = backend.add_event(event)
        assert stored.event_id == event.event_id

        retrieved = backend.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.subject == "Joanna"
        assert retrieved.action == "watched"
        assert retrieved.object == "Eternal Sunshine"
        assert retrieved.timestamp == "2019"

    def test_get_events_by_subject(self, backend: SQLiteMemoryBackend) -> None:
        e1 = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor="a1",
        )
        e2 = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="moved_to",
            object="Shanghai",
            timestamp="last year",
            certainty=Certainty.CERTAIN,
            source_anchor="a2",
        )
        backend.add_event(e1)
        backend.add_event(e2)

        events = backend.get_events_by_subject("default", "Joanna")
        assert len(events) == 2
        actions = {e.action for e in events}
        assert "watched" in actions
        assert "moved_to" in actions

    def test_query_by_subject_action(self, backend: SQLiteMemoryBackend) -> None:
        event = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor="a1",
        )
        backend.add_event(event)

        events = backend.get_events_by_subject_and_action("default", "Joanna", "watched")
        assert len(events) == 1
        assert events[0].object == "Eternal Sunshine"

        # Different action returns nothing
        events = backend.get_events_by_subject_and_action("default", "Joanna", "moved_to")
        assert len(events) == 0

    def test_missing_event_returns_none(self, backend: SQLiteMemoryBackend) -> None:
        result = backend.get_event("evt_nonexistent")
        assert result is None

    def test_backend_add_event(self, backend: SQLiteMemoryBackend) -> None:
        event = MemoryEvent(
            namespace="default",
            subject="Deborah",
            action="lost_family_member",
            object="mother",
            timestamp="a few years ago",
            certainty=Certainty.CERTAIN,
            source_anchor="a1",
        )
        stored = backend.add_event(event)
        assert stored.event_id == event.event_id

        retrieved = backend.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.action == "lost_family_member"

    def test_qualifiers_round_trip(self, backend: SQLiteMemoryBackend) -> None:
        event = MemoryEvent(
            namespace="default",
            subject="Joanna",
            action="watched",
            object="Eternal Sunshine",
            timestamp="2019",
            certainty=Certainty.CERTAIN,
            source_anchor="a1",
            qualifiers={"location": "home", "emotion": "moved"},
        )
        backend.add_event(event)

        retrieved = backend.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.qualifiers == {"location": "home", "emotion": "moved"}
