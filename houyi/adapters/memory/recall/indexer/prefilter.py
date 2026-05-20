"""MultiAxisPrefilter — query multiple axes in parallel, intersect results."""

from __future__ import annotations

import asyncio
import time

from houyi.adapters.memory.recall.indexer.axis import (
    AxisQuery,
    AxisResult,
    IndexAxis,
)
from houyi.adapters.memory.recall.indexer.registry import AxisRegistry


class MultiAxisPrefilter:
    def __init__(self, registry: AxisRegistry) -> None:
        self._registry = registry

    async def prefilter(
        self,
        queries: list[AxisQuery],
        *,
        deadline_ms: int | None = None,
    ) -> list[AxisResult]:
        if not queries:
            return []
        started = time.monotonic()
        tasks: list[asyncio.Task[AxisResult | None]] = []
        for q in queries:
            axis = self._registry.get(q.axis)
            if axis is None:
                continue
            tasks.append(asyncio.create_task(_run_axis(axis, q, deadline_ms=deadline_ms)))
        if not tasks:
            return []
        results: list[AxisResult] = []
        for coro in asyncio.as_completed(tasks):
            elapsed_ms = (time.monotonic() - started) * 1000
            if deadline_ms is not None and elapsed_ms > deadline_ms:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break
            result = await coro
            if result is not None:
                results.append(result)
        return results

    def intersect(self, results: list[AxisResult]) -> frozenset[str]:
        if not results:
            return frozenset()
        ids = results[0].matched_ids
        for r in results[1:]:
            ids = ids & r.matched_ids
        return ids


async def _run_axis(
    axis: IndexAxis,
    q: AxisQuery,
    *,
    deadline_ms: int | None,
) -> AxisResult | None:
    try:
        return await axis.query(q, deadline_ms=deadline_ms)
    except Exception:
        return None


__all__ = ["MultiAxisPrefilter"]
