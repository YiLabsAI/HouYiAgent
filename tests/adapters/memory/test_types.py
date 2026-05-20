from __future__ import annotations

import pytest
from pydantic import ValidationError

from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    ForgettingPolicy,
    MemoryType,
)


def _fact(**overrides: object) -> AtomicFact:
    """Build a minimal valid AtomicFact for tests to override selectively."""
    base: dict[str, object] = {
        "subject": "user",
        "predicate": "lives_in",
        "object": "California",
        "certainty": Certainty.CERTAIN,
        "source_anchor": "chunk-1",
    }
    base.update(overrides)
    return AtomicFact(**base)


class TestAtomicFactHappyPath:
    """Happy-path behaviour for the minimal valid construction."""

    def test_build_minimal(self) -> None:
        fact = _fact()
        assert fact.subject == "user"
        assert fact.predicate == "lives_in"
        assert fact.object == "California"
        assert fact.certainty is Certainty.CERTAIN
        assert fact.source_anchor == "chunk-1"

    def test_defaults(self) -> None:
        fact = _fact()
        assert fact.event_time is None
        assert fact.valid_from is None
        assert fact.valid_to is None
        assert fact.source_offset is None
        assert fact.keywords == []
        assert fact.qualifiers is None

    def test_accepts_full_payload(self) -> None:
        fact = _fact(
            event_time="2023-05",
            valid_from=1_700_000_000.0,
            valid_to=1_700_010_000.0,
            source_offset=(10, 42),
            keywords=["California", "residence"],
            qualifiers={"since": "2022"},
        )
        assert fact.event_time == "2023-05"
        assert fact.valid_from == 1_700_000_000.0
        assert fact.valid_to == 1_700_010_000.0
        assert fact.source_offset == (10, 42)
        assert fact.keywords == ["California", "residence"]
        assert fact.qualifiers == {"since": "2022"}


class TestAtomicFactValidation:
    """Boundary and error-path validation."""

    @pytest.mark.parametrize(
        "field",
        ["subject", "predicate", "object", "source_anchor"],
    )
    def test_rejects_empty_required_field(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _fact(**{field: ""})

    @pytest.mark.parametrize(
        "field",
        ["subject", "predicate", "object", "source_anchor"],
    )
    def test_rejects_whitespace_only_field(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _fact(**{field: " "})

    def test_trims_whitespace(self) -> None:
        fact = _fact(
            subject=" user ",
            predicate=" lives_in ",
            object=" California ",
            source_anchor=" chunk-1 ",
        )
        assert fact.subject == "user"
        assert fact.predicate == "lives_in"
        assert fact.object == "California"
        assert fact.source_anchor == "chunk-1"

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValidationError):
            _fact(subject=123)

    def test_requires_certainty(self) -> None:
        # Explicit absence of certainty must fail; there is intentionally no default.
        with pytest.raises(ValidationError):
            AtomicFact(
                subject="user",
                predicate="lives_in",
                object="California",
                source_anchor="chunk-1",
            )  # type: ignore[call-arg]

    def test_rejects_invalid_certainty_value(self) -> None:
        with pytest.raises(ValidationError):
            _fact(certainty="maybe")

    def test_validity_equal_ok(self) -> None:
        # Equality is allowed so a one-moment fact can be represented.
        fact = _fact(valid_from=1_700_000_000.0, valid_to=1_700_000_000.0)
        assert fact.valid_from == fact.valid_to

    def test_validity_inverted_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _fact(valid_from=1_700_010_000.0, valid_to=1_700_000_000.0)

    def test_offset_inverted_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _fact(source_offset=(10, 5))

    def test_offset_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _fact(source_offset=(-1, 5))

    def test_keywords_trim_blanks(self) -> None:
        fact = _fact(keywords=[" California ", "", " ", "residence"])
        assert fact.keywords == ["California", "residence"]

    def test_keywords_none_to_empty(self) -> None:
        fact = _fact(keywords=None)
        assert fact.keywords == []

    def test_keywords_non_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _fact(keywords=["ok", 123])


class TestAtomicFactDerivedState:
    """Derived properties driving writer pipeline decisions."""

    def test_active_open_ended(self) -> None:
        assert _fact().is_active is True

    def test_inactive_when_closed(self) -> None:
        fact = _fact(valid_from=1_700_000_000.0, valid_to=1_700_010_000.0)
        assert fact.is_active is False

    @pytest.mark.parametrize(
        "certainty,expected",
        [
            (Certainty.CERTAIN, True),
            (Certainty.PROBABLE, True),
            (Certainty.VAGUE, False),
        ],
    )
    def test_admits_to_main_store(self, certainty: Certainty, expected: bool) -> None:
        assert _fact(certainty=certainty).admits_to_main_store is expected


class TestAtomicFactSerialization:
    """Round-trip behaviour for persistence and transport."""

    def test_model_dump_round_trip(self) -> None:
        original = _fact(
            valid_from=1_700_000_000.0,
            source_offset=(0, 10),
            keywords=["a"],
            qualifiers={"k": "v"},
        )
        rebuilt = AtomicFact.model_validate(original.model_dump())
        assert rebuilt == original

    def test_json_round_trip(self) -> None:
        original = _fact(event_time="2023-05")
        rebuilt = AtomicFact.model_validate_json(original.model_dump_json())
        assert rebuilt == original


class TestMemoryTypeEnum:
    """Coverage of the two-tier MemoryType contract."""

    REQUIRED = {
        MemoryType.FACT,
        MemoryType.EVENT,
        MemoryType.PREFERENCE,
        MemoryType.PROFILE,
        MemoryType.PROCEDURE,
        MemoryType.CONSTRAINT,
        MemoryType.STRATEGY,
    }
    OPTIONAL = {MemoryType.PROJECT, MemoryType.CODE_STRUCTURE}

    def test_required_has_seven(self) -> None:
        assert len(self.REQUIRED) == 7

    def test_tiers_cover_all(self) -> None:
        assert set(MemoryType) == self.REQUIRED | self.OPTIONAL

    def test_tiers_are_disjoint(self) -> None:
        assert self.REQUIRED.isdisjoint(self.OPTIONAL)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("FACT", "fact"),
            ("EVENT", "event"),
            ("STRATEGY", "strategy"),
            ("CODE_STRUCTURE", "code_structure"),
        ],
    )
    def test_string_values_are_stable(self, name: str, value: str) -> None:
        # String values are part of the wire/persistence contract; pinning
        # them here keeps future renames from silently breaking storage.
        assert MemoryType[name].value == value

    def test_unit_kinds_excluded(self) -> None:
        # OBSERVATION/REFLECTION/TRAJECTORY live on the unit_kind axis,
        # not here. If they reappear on MemoryType the orthogonality
        # contract between the two axes is broken.
        forbidden = {"observation", "reflection", "trajectory"}
        assert forbidden.isdisjoint({m.value for m in MemoryType})


class TestForgettingPolicyCoversAllTypes:
    """ForgettingPolicy must supply a decay rate for every MemoryType."""

    def test_all_types_have_decay(self) -> None:
        policy = ForgettingPolicy()
        missing = [m.value for m in MemoryType if m.value not in policy.natural_decay_rates]
        assert missing == []

    def test_constraint_has_zero_decay(self) -> None:
        # Hard constraints must not silently decay; if this ever changes it
        # should be an explicit policy decision with its own test.
        policy = ForgettingPolicy()
        assert policy.natural_decay_rates["constraint"] == 0.0
