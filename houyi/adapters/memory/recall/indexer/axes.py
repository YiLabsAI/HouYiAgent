"""Concrete IndexAxis implementations backed by existing recall infrastructure."""

from __future__ import annotations

import asyncio
import time

from houyi.adapters.memory.backends.base import EntityStateView
from houyi.adapters.memory.recall.indexer.axis import (
    AxisQuery,
    AxisResult,
)


class EntityAxis:
    def __init__(self, view: EntityStateView, namespace: str = "default") -> None:
        self._view = view
        self._namespace = namespace

    @property
    def name(self) -> str:
        return "entity"

    async def query(self, q: AxisQuery, *, deadline_ms: int | None = None) -> AxisResult:
        started = time.monotonic()
        entity = str(q.params.get("entity", q.key))
        rows = await asyncio.to_thread(self._view.get_active, self._namespace, entity=entity)
        ids: set[str] = set()
        for row in rows:
            ids.add(f"{row.entity}:{row.attribute}:{row.value}")
        elapsed = (time.monotonic() - started) * 1000
        return AxisResult(
            axis=self.name,
            matched_ids=frozenset(ids),
            total_scanned=len(rows),
            cost_ms=round(elapsed, 3),
        )


class TimeAxis:
    def __init__(self, view: EntityStateView, namespace: str = "default") -> None:
        self._view = view
        self._namespace = namespace

    @property
    def name(self) -> str:
        return "time"

    async def query(self, q: AxisQuery, *, deadline_ms: int | None = None) -> AxisResult:
        started = time.monotonic()
        since_raw = q.params.get("since")
        until_raw = q.params.get("until")
        since = float(since_raw) if isinstance(since_raw, (int, float)) else None
        until = float(until_raw) if isinstance(until_raw, (int, float)) else None
        rows = await asyncio.to_thread(self._view.get_active, self._namespace)
        ids: set[str] = set()
        for row in rows:
            ts_raw = getattr(row, "updated_at", None)
            if ts_raw is None or not isinstance(ts_raw, (int, float)):
                continue
            ts = float(ts_raw)
            if since is not None and ts < since:
                continue
            if until is not None and ts > until:
                continue
            ids.add(f"{row.entity}:{row.attribute}:{row.value}")
        elapsed = (time.monotonic() - started) * 1000
        return AxisResult(
            axis=self.name,
            matched_ids=frozenset(ids),
            total_scanned=len(rows),
            cost_ms=round(elapsed, 3),
        )


__all__ = ["EntityAxis", "TimeAxis"]
