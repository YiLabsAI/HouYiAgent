"""Tiered (L0+L1) bench ingestor adapter for memory_bench.

Wraps the production TurnWriter + ExtractorWorker pair behind the
runner's IngestorLike protocol so the bench harness can drive the
asynchronous write path under the same interface as the legacy
MemoryIngestor. The adapter intentionally does *not* re-implement
extraction or fact projection — that work stays inside the worker so the
benchmark exercises the real production code path.

Two surfaces beyond ingest_turn are exposed for the runner:

- drain() — runs ExtractorWorker.process_once in a loop until the
  queue is empty so all enqueued L1 work is settled before the runner reads
  active memories.
- extractor_calls — cumulative count of extract invocations (a
  cheap cost proxy aligned with the bench's BenchTimings.extractor_calls
  field).

Retraction is intentionally retained on the synchronous fast path: it is
sub-millisecond regex work and skipping it for tiered would not reflect the
production contract (retraction signals fire from L0+).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from houyi.adapters.memory.retraction import RetractionOrchestrator, RetractionTarget
from houyi.adapters.memory.turn_writer import TurnWriter, WriteResult
from houyi.adapters.memory.types import RawTurn
from houyi.adapters.memory.workers.extractor_worker import ExtractorWorker


class _ExtractorLike(Protocol):
    """Subset of AtomicFactExtractor we wrap for the call counter."""

    async def extract(self, text: str, source_anchor: str | None) -> Any: ...


class _CountingExtractor:
    """Decorate an extractor so each call increments a counter.

    The bench exposes the resulting count via TieredBenchIngestor.extractor_calls
    so the runner's cost_probe can attribute LLM activity to the tiered path.
    """

    def __init__(self, inner: _ExtractorLike) -> None:
        self._inner = inner
        self.calls = 0

    async def extract(self, text: str, source_anchor: str | None) -> Any:
        self.calls += 1
        return await self._inner.extract(text, source_anchor)


class TieredBenchIngestor:
    """Bench-side facade for the L0+L1 tiered write path.

    Construct with the same backend / extractor / entity-state / inbox
    objects the production wiring uses. The adapter:

    1. Runs synchronous retraction on every turn (matches MemoryIngestor).
    2. Calls TurnWriter.fast_path to land the L0 row + enqueue L1 work.
    3. Exposes drain() so the bench runner can settle async work between
       the ingest stage and the active-memory readout.

    The constructor wraps the supplied extractor with a counter; user code
    should not pass the counted extractor in elsewhere or its calls would
    double-count.
    """

    def __init__(
        self,
        *,
        turn_writer: TurnWriter,
        extractor: _ExtractorLike,
        worker_factory,
        retraction: RetractionOrchestrator | None = None,
        namespace: str = "bench",
        session_id: str = "bench-session",
    ) -> None:
        """Construct the adapter.

        Args:
            turn_writer: production TurnWriter bound to the same SQLite
                backend as the worker.
            extractor: AtomicFactExtractor-like object. Wrapped with a
                counting decorator and handed to the worker via
                worker_factory.
            worker_factory: callable (counting_extractor) -> ExtractorWorker.
                Receives the wrapped extractor so the counter sees every
                drain call. The bench fixture is responsible for binding
                backend / entity_state / candidate_inbox.
            retraction: optional RetractionOrchestrator. When None, the
                fast path skips retraction (suitable for fixtures with no
                retraction turns).
            namespace / session_id: anchor metadata stamped on each
                synthesized RawTurn.
        """
        self._turn_writer = turn_writer
        self._counting_extractor = _CountingExtractor(extractor)
        self._worker: ExtractorWorker = worker_factory(self._counting_extractor)
        self._retraction = retraction
        self._namespace = namespace
        self._session_id = session_id
        self._turn_seq = 0

    # ------------------------------------------------------------------
    # IngestorLike surface
    # ------------------------------------------------------------------

    async def ingest_turn(
        self,
        text: str,
        *,
        source_anchor: str | None,
        recent_targets: Iterable[RetractionTarget] = (),
    ) -> WriteResult | None:
        """Run sync retraction + L0 write + L1 enqueue for one user turn.

        Returns the WriteResult on the normal path, or None when retraction
        absorbed the turn (matches MemoryIngestor's "retraction-only turn"
        contract).
        """
        if self._retraction is not None:
            outcome = self._retraction.process(text, list(recent_targets))
            if outcome.signal is not None:
                return None

        turn = RawTurn(
            namespace=self._namespace,
            session_id=self._session_id,
            role="user",
            content=text,
            metadata={"source_anchor": source_anchor or ""},
        )
        return self._turn_writer.fast_path(turn)

    # ------------------------------------------------------------------
    # Bench-only surface
    # ------------------------------------------------------------------

    async def drain(self, *, max_iterations: int = 1024) -> int:
        """Drain queued L1 jobs until the queue is empty.

        Returns the total number of jobs processed. max_iterations is a
        defensive cap so a buggy worker never spins forever inside a test.
        """
        total = 0
        for _ in range(max_iterations):
            processed = await self._worker.process_once()
            if processed == 0:
                break
            total += processed
        return total

    @property
    def extractor_calls(self) -> int:
        """Cumulative number of extractor.extract() invocations."""
        return self._counting_extractor.calls


__all__ = ["TieredBenchIngestor"]
