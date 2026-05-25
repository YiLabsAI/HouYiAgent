"""Bench timing primitives shared by the runner and CLI.

Captures per-call latency for the ingest stage and (when applicable) the
async drain stage so callers can answer questions like "how much wall-clock
does the new tiered write path save versus the legacy inline-extract path?".

The dataclass intentionally records both raw call counts and percentile
summaries; raw samples are kept on the per-session object so reports can be
re-aggregated, while the aggregate keeps only the summary so a 200-session
run does not blow up memory.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

# Path-kind labels are part of the report contract and surface verbatim in
# Markdown / JSON outputs. Keep them stable so downstream tooling can pivot.
PATH_KIND_SYNC_INLINE = "sync_inline"
"""Legacy MemoryIngestor path: extraction LLM runs inside ingest_turn."""

PATH_KIND_TIERED_ASYNC = "tiered_async"
"""TurnWriter + ExtractorWorker path: ingest_turn returns after L0 enqueue;
extraction is drained out-of-band before the bench reads active memories."""


def _percentile(samples: list[float], pct: float) -> float:
    """Return the linear-interpolation percentile of samples (0-100)."""
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


@dataclass(frozen=True, slots=True)
class BenchTimings:
    """Aggregate latency / cost-proxy summary for one or more sessions.

    All time fields are in milliseconds. extractor_calls is the number of
    times the LLM-backed extractor ran during the measurement window — used
    as a cost proxy because each call dominates token spend in the bench.
    """

    path_kind: str = PATH_KIND_SYNC_INLINE
    ingest_calls: int = 0
    ingest_total_ms: float = 0.0
    ingest_p50_ms: float = 0.0
    ingest_p95_ms: float = 0.0
    ingest_max_ms: float = 0.0
    drain_total_ms: float = 0.0
    extractor_calls: int = 0

    @classmethod
    def empty(cls, path_kind: str = PATH_KIND_SYNC_INLINE) -> BenchTimings:
        return cls(path_kind=path_kind)

    @classmethod
    def from_samples(
        cls,
        path_kind: str,
        ingest_samples_ms: Iterable[float],
        *,
        drain_total_ms: float = 0.0,
        extractor_calls: int = 0,
    ) -> BenchTimings:
        samples = [float(x) for x in ingest_samples_ms]
        if not samples:
            return cls(
                path_kind=path_kind,
                drain_total_ms=drain_total_ms,
                extractor_calls=extractor_calls,
            )
        return cls(
            path_kind=path_kind,
            ingest_calls=len(samples),
            ingest_total_ms=sum(samples),
            ingest_p50_ms=_percentile(samples, 50.0),
            ingest_p95_ms=_percentile(samples, 95.0),
            ingest_max_ms=max(samples),
            drain_total_ms=drain_total_ms,
            extractor_calls=extractor_calls,
        )

    def merge(self, other: BenchTimings) -> BenchTimings:
        """Combine with another BenchTimings (used by the runner aggregate).

        Percentiles cannot be merged from summaries alone; the aggregate
        keeps p50/p95 of the larger sample as a best-effort indicator and
        takes the max-of-max. Callers needing exact aggregate percentiles
        should re-aggregate from raw samples held on each SessionReport.
        """
        if self.ingest_calls == 0 and other.ingest_calls == 0:
            return BenchTimings(
                path_kind=other.path_kind or self.path_kind,
                drain_total_ms=self.drain_total_ms + other.drain_total_ms,
                extractor_calls=self.extractor_calls + other.extractor_calls,
            )
        bigger = self if self.ingest_calls >= other.ingest_calls else other
        return BenchTimings(
            path_kind=other.path_kind or self.path_kind,
            ingest_calls=self.ingest_calls + other.ingest_calls,
            ingest_total_ms=self.ingest_total_ms + other.ingest_total_ms,
            ingest_p50_ms=bigger.ingest_p50_ms,
            ingest_p95_ms=bigger.ingest_p95_ms,
            ingest_max_ms=max(self.ingest_max_ms, other.ingest_max_ms),
            drain_total_ms=self.drain_total_ms + other.drain_total_ms,
            extractor_calls=self.extractor_calls + other.extractor_calls,
        )


@dataclass(frozen=True, slots=True)
class SessionTimingSamples:
    """Raw per-call latencies retained on a SessionReport.

    Kept separate from BenchTimings so the aggregate can stay summary-only
    without losing the ability to re-derive exact percentiles from a run.
    """

    path_kind: str
    ingest_ms: tuple[float, ...] = ()
    drain_total_ms: float = 0.0
    extractor_calls: int = 0


__all__ = [
    "PATH_KIND_SYNC_INLINE",
    "PATH_KIND_TIERED_ASYNC",
    "BenchTimings",
    "SessionTimingSamples",
]
