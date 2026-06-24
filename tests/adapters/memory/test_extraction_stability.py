from __future__ import annotations

from typing import Any

from houyi.adapters.memory.backends.base import EntityStateRecord, EntityStateView
from houyi.adapters.memory.entity_resolver import EntityStateAwareResolver, TurnContext
from houyi.adapters.memory.extractor import AtomicFactExtractor
from houyi.adapters.memory.types import AtomicFact, Certainty


class DummyEntityStateView(EntityStateView):
    """Simple in-memory state view for testing."""

    def __init__(self, records: list[EntityStateRecord]) -> None:
        self._records = records

    def get_active(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        out = []
        for r in self._records:
            if (
                r.namespace == namespace
                and r.entity == entity
                and (attribute is None or r.attribute == attribute)
            ):
                out.append(r)
        return out

    def get_history(
        self,
        namespace: str,
        entity: str,
        attribute: str,
    ) -> list[EntityStateRecord]:
        return self.get_active(namespace, entity, attribute)

    def get_as_of(
        self, namespace: str, entity: str, attribute: str, as_of: float
    ) -> list[EntityStateRecord]:
        return []

    def invalidate(
        self, namespace: str, entity: str, attribute: str, valid_to: float | None = None
    ) -> bool:
        return False

    def list_entities(self, namespace: str) -> list[str]:
        return []

    def list_conflicted_triples(self, namespace: str | None = None) -> list[tuple[str, str, str]]:
        return []

    def supersede(
        self, namespace, entity, attribute, *, keep_state_id, valid_to
    ) -> tuple[int, int]:
        return (0, 0)

    def upsert(
        self,
        namespace: str,
        entity: str,
        attribute: str,
        value: str,
        certainty: Any = None,
        valid_from: float | None = None,
        source_unit_id: str | None = None,
        qualifiers: dict[str, str] | None = None,
    ) -> EntityStateRecord:
        raise NotImplementedError()


def test_restore_generic_objects() -> None:
    # 1. Test restoration of same-type generics (location)
    f1 = AtomicFact(
        subject="user",
        predicate="visited_place",
        object="Tokyo",
        certainty=Certainty.CERTAIN,
        source_anchor="turn-1",
        qualifiers={"object_type": "location"},
    )
    f2 = AtomicFact(
        subject="user",
        predicate="likes",
        object="the city",
        certainty=Certainty.CERTAIN,
        source_anchor="turn-1",
        qualifiers={"object_type": "location"},
    )

    restored = AtomicFactExtractor._restore_generic_objects([f1, f2])
    assert len(restored) == 2
    assert restored[0].object == "Tokyo"
    assert restored[1].object == "Tokyo"  # Restored to Tokyo!
    assert restored[1].certainty == Certainty.CERTAIN

    # 2. Test restoration of same-type generics (vehicle)
    f3 = AtomicFact(
        subject="user",
        predicate="bought",
        object="Ferrari 488 GTB",
        certainty=Certainty.CERTAIN,
        source_anchor="turn-1",
        qualifiers={"object_type": "vehicle"},
    )
    f4 = AtomicFact(
        subject="user",
        predicate="drove",
        object="new ride",
        certainty=Certainty.CERTAIN,
        source_anchor="turn-1",
        qualifiers={"object_type": "vehicle"},
    )

    restored2 = AtomicFactExtractor._restore_generic_objects([f3, f4])
    assert len(restored2) == 2
    assert restored2[0].object == "Ferrari 488 GTB"
    assert restored2[1].object == "Ferrari 488 GTB"  # Restored!

    # 3. Test unrestorable generics -> downgrade certainty to PROBABLE
    f5 = AtomicFact(
        subject="user",
        predicate="has_pet",
        object="some animal",
        certainty=Certainty.CERTAIN,
        source_anchor="turn-1",
        qualifiers={"object_type": "animal"},
    )

    restored3 = AtomicFactExtractor._restore_generic_objects([f5])
    assert len(restored3) == 1
    assert restored3[0].object == "some animal"
    assert restored3[0].certainty == Certainty.PROBABLE  # Downgraded!


class TestEntityStateResolver:
    def test_resolver_multiword(self) -> None:
        # Setup dummy database records
        record = EntityStateRecord(
            namespace="ws_1",
            entity="Caroline",
            attribute="has_vehicle",
            value="Ferrari 488 GTB",
            certainty=Certainty.CERTAIN,
            valid_from=1000.0,
            source_unit_id="turn-1",
        )
        view = DummyEntityStateView([record])

        resolver = EntityStateAwareResolver(view, namespace="ws_1")

        # 1. Test direct generic word
        turn_1 = TurnContext(
            text="I drove my new ride",
            speaker_id="Caroline",
            session_id="session_1",
            turn_id="turn_2",
        )

        # Let's mock a simple inner resolver that returns the base entity placeholder
        class InnerMock:
            def resolve(self, turn: TurnContext) -> str:
                return "vehicle"

        resolver_1 = EntityStateAwareResolver(view, namespace="ws_1", inner=InnerMock())
        assert resolver_1.resolve(turn_1) == "Ferrari 488 GTB"

        # 2. Test multiword generic phrase ("new ride")
        class InnerMock2:
            def resolve(self, turn: TurnContext) -> str:
                return "new ride"

        resolver_2 = EntityStateAwareResolver(view, namespace="ws_1", inner=InnerMock2())
        assert resolver_2.resolve(turn_1) == "Ferrari 488 GTB"
