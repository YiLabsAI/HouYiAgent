"""apply_forgetting unit tests.

Covers natural decay, capacity eviction, policy toggling, and edge cases.
"""

from __future__ import annotations

import time

from houyi.adapters.memory.forgetting import apply_forgetting
from houyi.adapters.memory.types import (
    ForgettingPolicy,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)


def _record(
    key: str = "k",
    decay: float = 1.0,
    updated_at: float | None = None,
    memory_type: MemoryType = MemoryType.FACT,
    scope: MemoryScope = MemoryScope.SESSION,
    confidence: float = 1.0,
) -> MemoryRecord:
    return MemoryRecord(
        key=key,
        content=f"content for {key}",
        decay=decay,
        updated_at=updated_at or time.time(),
        memory_type=memory_type,
        scope=scope,
        confidence=confidence,
    )


class TestNaturalDecay:
    def test_fresh_record_survives(self):
        records = [_record("fresh")]
        policy = ForgettingPolicy()
        survivors = apply_forgetting(records, policy)
        assert len(survivors) == 1

    def test_stale_record_evicted(self):
        records = [_record("stale", decay=0.05, updated_at=1.0)]
        policy = ForgettingPolicy(natural_decay_threshold=0.1)
        survivors = apply_forgetting(records, policy)
        assert len(survivors) == 0

    def test_decay_disabled(self):
        records = [_record("stale", decay=0.05, updated_at=1.0)]
        policy = ForgettingPolicy(natural_decay_enabled=False)
        survivors = apply_forgetting(records, policy)
        assert len(survivors) == 1

    def test_constraint_zero_rate(self):
        records = [
            _record("c", decay=1.0, updated_at=1.0, memory_type=MemoryType.CONSTRAINT),
        ]
        policy = ForgettingPolicy(
            natural_decay_rates={"constraint": 0.0, "fact": 0.01},
        )
        survivors = apply_forgetting(records, policy)
        assert len(survivors) == 1

    def test_decay_reduces_value(self):
        rec = _record("old", decay=1.0, updated_at=time.time() - 86400 * 30)
        policy = ForgettingPolicy(natural_decay_threshold=0.01)
        survivors = apply_forgetting([rec], policy)
        if survivors:
            assert survivors[0].decay < 1.0


class TestCapacityEviction:
    def test_under_capacity_all_survive(self):
        records = [_record(f"r{i}") for i in range(5)]
        policy = ForgettingPolicy(
            natural_decay_enabled=False,
            max_memories_per_scope=10,
        )
        survivors = apply_forgetting(records, policy)
        assert len(survivors) == 5

    def test_over_capacity_evicts(self):
        records = [_record(f"r{i}", confidence=0.5 + i * 0.01) for i in range(5)]
        policy = ForgettingPolicy(
            natural_decay_enabled=False,
            max_memories_per_scope=3,
        )
        survivors = apply_forgetting(records, policy)
        assert len(survivors) == 3

    def test_eviction_keeps_highest(self):
        low = _record("low", confidence=0.1, decay=0.1)
        high = _record("high", confidence=1.0, decay=1.0)
        policy = ForgettingPolicy(
            natural_decay_enabled=False,
            max_memories_per_scope=1,
        )
        survivors = apply_forgetting([low, high], policy)
        assert len(survivors) == 1
        assert survivors[0].key == "high"

    def test_eviction_per_scope(self):
        session = [_record(f"s{i}", scope=MemoryScope.SESSION) for i in range(3)]
        user = [_record(f"u{i}", scope=MemoryScope.USER) for i in range(3)]
        policy = ForgettingPolicy(
            natural_decay_enabled=False,
            max_memories_per_scope=2,
        )
        survivors = apply_forgetting(session + user, policy)
        assert len(survivors) == 4

    def test_eviction_disabled(self):
        records = [_record(f"r{i}") for i in range(10)]
        policy = ForgettingPolicy(
            natural_decay_enabled=False,
            capacity_eviction_enabled=False,
        )
        survivors = apply_forgetting(records, policy)
        assert len(survivors) == 10


class TestEdgeCases:
    def test_empty_records(self):
        survivors = apply_forgetting([], ForgettingPolicy())
        assert survivors == []

    def test_single_record_fresh(self):
        survivors = apply_forgetting([_record("only")], ForgettingPolicy())
        assert len(survivors) == 1

    def test_both_policies_combined(self):
        fresh = _record("fresh", decay=1.0)
        stale = _record("stale", decay=0.05, updated_at=1.0)
        policy = ForgettingPolicy(
            natural_decay_threshold=0.1,
            max_memories_per_scope=10,
        )
        survivors = apply_forgetting([fresh, stale], policy)
        assert len(survivors) == 1
        assert survivors[0].key == "fresh"
