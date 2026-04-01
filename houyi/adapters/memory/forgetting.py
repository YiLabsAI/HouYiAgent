"""Active forgetting policy for memory lifecycle management.

Implements four forgetting mechanisms:
1. Natural decay: time-based confidence erosion
2. Conflict supersede: new memory replaces conflicting old
3. Explicit forget: user-initiated deletion (handled externally)
4. Capacity eviction: LRU-like eviction when over scope capacity
"""

from __future__ import annotations

import math
import time

from houyi.adapters.memory.types import ForgettingPolicy, MemoryRecord


def apply_forgetting(
    records: list[MemoryRecord],
    policy: ForgettingPolicy,
) -> list[MemoryRecord]:
    """Apply forgetting policy to a list of records.

    Returns the surviving records (stale records are excluded).
    Modifies decay values in-place on surviving records.
    """
    survivors: list[MemoryRecord] = []

    for record in records:
        if policy.natural_decay_enabled:
            _apply_natural_decay(record, policy)
            if record.decay < policy.natural_decay_threshold:
                continue

        survivors.append(record)

    if policy.capacity_eviction_enabled:
        survivors = _apply_capacity_eviction(survivors, policy)

    return survivors


def _apply_natural_decay(
    record: MemoryRecord,
    policy: ForgettingPolicy,
) -> None:
    """Update record.decay based on time since last access.

    decay(t) = current_decay * exp(-rate * days_since_update)
    """
    rate = policy.natural_decay_rates.get(record.memory_type.value, 0.01)
    if rate == 0.0:
        return
    days = (time.time() - record.updated_at) / 86400.0
    record.decay = record.decay * math.exp(-rate * max(days, 0))


def _apply_capacity_eviction(
    records: list[MemoryRecord],
    policy: ForgettingPolicy,
) -> list[MemoryRecord]:
    """Evict oldest low-confidence records when over capacity per scope."""
    from collections import defaultdict

    by_scope: dict[str, list[MemoryRecord]] = defaultdict(list)
    for r in records:
        by_scope[r.scope.value].append(r)

    survivors: list[MemoryRecord] = []
    for scope_records in by_scope.values():
        if len(scope_records) <= policy.max_memories_per_scope:
            survivors.extend(scope_records)
        else:
            scope_records.sort(
                key=lambda r: (r.decay * r.confidence, r.updated_at),
                reverse=True,
            )
            survivors.extend(scope_records[: policy.max_memories_per_scope])

    return survivors
