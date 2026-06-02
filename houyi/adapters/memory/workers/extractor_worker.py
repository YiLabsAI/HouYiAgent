"""L1 extractor worker: pull jobs from the extract queue and run the LLM.

The worker drains SQLiteMemoryBackend.claim_extract_jobs in batches,
hands each turn to an AtomicFactExtractor, and projects the resulting
facts onto the entity-state view (or the candidate inbox for vague /
sourceless rows). Job lifecycle:

- success -> mark_extract_done
- failure -> mark_extract_failed (retries up to max_attempts)

Two entry points are exposed:

- process_once runs a single batch and returns. Used by tests and by
  the dreamer's batched flush path so the caller controls the loop.
- run_forever drives an idle-sleep loop suitable for a long-running
  daemon. Cancellation is cooperative via an asyncio.Event.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from houyi.adapters.memory.fact_promoter import FactPromoter, MemoryRecordPromoter
from houyi.adapters.memory.types import AtomicFact, Certainty, MemoryRecord, RawTurn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractorWorkerConfig:
    """Tunables for the L1 worker. Defaults track budgets."""

    batch_size: int = 8
    """Max jobs claimed per process_once call."""

    lease_seconds: float = 60.0
    """How long an in-flight claim is honored before another worker may
    re-claim it. Must exceed the worst-case extraction latency for the
    chosen LLM provider; the dreamer's offline batch may need a larger
    value.
    """

    max_attempts: int = 5
    """After this many failures a job is parked in state='failed'
    instead of being re-queued. Matches the queue's max_attempts
    default so configuration drift cannot desynchronize the policy.
    """

    idle_sleep_s: float = 1.0
    """Seconds to sleep between empty polls in run_forever."""

    namespace: str | None = None
    """Optional namespace pin. Useful for sharded deployments where
    multiple workers each own a subset of namespaces.
    """


class _BackendProtocol(Protocol):
    """The slice of the SQLite memory backend this worker depends on.

    Defined as a Protocol so tests can inject lightweight fakes without
    standing up SQLite plus an embedding provider.
    """

    def claim_extract_jobs(
        self,
        *,
        limit: int,
        namespace: str | None,
        lease_seconds: float,
    ) -> list[tuple[str, RawTurn]]: ...
    def mark_extract_done(self, queue_id: str) -> None: ...
    def mark_extract_failed(
        self, queue_id: str, error: str, *, retry: bool, max_attempts: int
    ) -> None: ...
    def put(self, record: MemoryRecord) -> None: ...
    def transaction(self) -> Any: ...


class _EntityStateProtocol(Protocol):
    def upsert(
        self,
        namespace: str,
        entity: str,
        attribute: str,
        value: str,
        *,
        certainty: Certainty,
        valid_from: float | None,
        source_unit_id: str | None,
        qualifiers: dict[str, str] | None,
    ) -> Any: ...


class _CandidateInboxProtocol(Protocol):
    def add(self, namespace: str, fact: AtomicFact) -> str: ...
    def add_sourceless(self, namespace: str, raw_payload: dict[str, Any]) -> str: ...


class _ExtractorProtocol(Protocol):
    async def extract(self, text: str, source_anchor: str | None) -> Any: ...  # ExtractionResult


class _BatchExtractorProtocol(_ExtractorProtocol, Protocol):
    async def extract_batch(self, turns: list[tuple[str, str | None]]) -> list[Any]: ...


def _source_anchor_for(turn: RawTurn) -> str:
    raw = turn.metadata.get("source_anchor", "") if isinstance(turn.metadata, dict) else ""
    anchor = str(raw).strip()
    if anchor:
        return anchor
    return turn.turn_id


def _extract_text_for(turn: RawTurn) -> str:
    if isinstance(turn.metadata, dict):
        raw = turn.metadata.get("extract_text", "")
        if isinstance(raw, str) and raw.strip():
            return raw
    return turn.content


class ExtractorWorker:
    """Drain the extract queue and project facts onto the state view."""

    def __init__(
        self,
        *,
        backend: _BackendProtocol,
        extractor: _ExtractorProtocol,
        entity_state: _EntityStateProtocol,
        candidate_inbox: _CandidateInboxProtocol,
        promoter: FactPromoter | None = None,
        config: ExtractorWorkerConfig | None = None,
    ) -> None:
        """Construct the L1 worker.

        Args:
        backend: provides the queue claim/finalize API.
        extractor: an AtomicFactExtractor-compatible object.
        entity_state: where certain facts are projected.
        candidate_inbox: where vague / sourceless facts park.
        promoter: L1->L2 projection. Defaults to a MemoryRecordPromoter
        bound to the same backend so that every accepted certain fact
        materializes a MemoryRecord for the vector path out of the box.
        Pass a custom FactPromoter to override the policy, or a noop
        implementation to disable the projection entirely. Promoter
        failures are absorbed (logged + ignored); entity-state remains
        the source of truth.
        config: tunables; defaults from ExtractorWorkerConfig.
        """
        if backend is None or extractor is None:
            raise ValueError("backend and extractor are required")
        if entity_state is None or candidate_inbox is None:
            raise ValueError("entity_state and candidate_inbox are required")
        self._backend = backend
        self._extractor = extractor
        self._entity_state = entity_state
        self._candidate_inbox = candidate_inbox
        # Default to MemoryRecordPromoter so the L1 worker materializes
        # MemoryRecord rows that the vector path can index. Callers that
        # need to disable promotion pass a noop FactPromoter explicitly.
        self._promoter: FactPromoter = promoter or MemoryRecordPromoter(backend)
        self._config = config or ExtractorWorkerConfig()

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

    async def process_once(self) -> int:
        """Claim one batch and process every job in it.

        Returns the number of jobs that were processed (regardless of
        success/failure outcome). A return value of 0 means the queue
        was empty at claim time and the caller should sleep.
        """
        cfg = self._config
        claimed = await asyncio.to_thread(
            self._backend.claim_extract_jobs,
            limit=cfg.batch_size,
            namespace=cfg.namespace,
            lease_seconds=cfg.lease_seconds,
        )
        if not claimed:
            return 0

        if hasattr(self._extractor, "extract_batch"):
            await self._process_claimed_batch(claimed)
            return len(claimed)

        for queue_id, turn in claimed:
            await self._process_job(queue_id, turn)
        return len(claimed)

    async def _process_claimed_batch(self, claimed: list[tuple[str, RawTurn]]) -> None:
        extractor = self._extractor
        if not hasattr(extractor, "extract_batch"):
            for queue_id, turn in claimed:
                await self._process_job(queue_id, turn)
            return

        batch_extractor = extractor
        payload = [(_extract_text_for(turn), _source_anchor_for(turn)) for _, turn in claimed]
        try:
            results = await batch_extractor.extract_batch(payload)
        except Exception as exc:
            logger.warning("batch extractor failed for %d jobs: %s", len(claimed), exc)
            for queue_id, _turn in claimed:
                await asyncio.to_thread(
                    self._backend.mark_extract_failed,
                    queue_id,
                    f"batch_extract: {exc}"[:1000],
                    retry=True,
                    max_attempts=self._config.max_attempts,
                )
            return

        if len(results) != len(claimed):
            logger.warning(
                "batch extractor returned mismatched results: expected=%d got=%d",
                len(claimed),
                len(results),
            )
            for queue_id, _turn in claimed:
                await asyncio.to_thread(
                    self._backend.mark_extract_failed,
                    queue_id,
                    "batch_extract: mismatched result count",
                    retry=True,
                    max_attempts=self._config.max_attempts,
                )
            return

        for (queue_id, turn), result in zip(claimed, results, strict=False):
            try:
                await self._project_result(turn, result)
            except Exception as exc:
                logger.warning("projection failed for queue_id=%s: %s", queue_id, exc)
                await asyncio.to_thread(
                    self._backend.mark_extract_failed,
                    queue_id,
                    f"projection: {exc}"[:1000],
                    retry=True,
                    max_attempts=self._config.max_attempts,
                )
                continue
            await asyncio.to_thread(self._backend.mark_extract_done, queue_id)

    async def run_forever(
        self,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Loop until stop is set, processing batches with idle backoff.

        stop defaults to a freshly-created event so callers that do not
        care about cancellation can simply await worker.run_forever()
        and rely on the surrounding task cancellation. The loop never
        raises out of a single failed batch; failures are absorbed
        inside _process_job.
        """
        signal = stop or asyncio.Event()
        while not signal.is_set():
            try:
                processed = await self.process_once()
            except Exception:
                logger.exception("extractor worker batch failed; backing off")
                processed = 0
            if processed == 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(signal.wait(), timeout=self._config.idle_sleep_s)

    # ------------------------------------------------------------------
    # Per-job pipeline
    # ------------------------------------------------------------------

    async def _process_job(self, queue_id: str, turn: RawTurn) -> None:
        """Run one extraction job. Marks done/failed via the backend.

        Failures fall back to mark_extract_failed with retry enabled
        so transient LLM hiccups recover automatically. Persistent
        failures terminate after max_attempts.
        """
        try:
            result = await self._extractor.extract(
                text=_extract_text_for(turn),
                source_anchor=_source_anchor_for(turn),
            )
        except Exception as exc:
            logger.warning(
                "extractor failed for queue_id=%s turn=%s: %s",
                queue_id,
                turn.turn_id,
                exc,
            )
            await asyncio.to_thread(
                self._backend.mark_extract_failed,
                queue_id,
                str(exc)[:1000],
                retry=True,
                max_attempts=self._config.max_attempts,
            )
            return

        try:
            await self._project_result(turn, result)
        except Exception as exc:
            logger.warning("projection failed for queue_id=%s: %s", queue_id, exc)
            await asyncio.to_thread(
                self._backend.mark_extract_failed,
                queue_id,
                f"projection: {exc}"[:1000],
                retry=True,
                max_attempts=self._config.max_attempts,
            )
            return

        await asyncio.to_thread(self._backend.mark_extract_done, queue_id)

    async def _project_result(self, turn: RawTurn, result: Any) -> None:
        """Project an ExtractionResult onto storage.

        Three buckets:

        - facts with certainty != VAGUE -> entity-state upsert
        - facts with certainty == VAGUE -> candidate inbox (vague)
        - raw_sourceless -> candidate inbox (sourceless)
        """
        await asyncio.to_thread(self._sync_project_result, turn, result)

    def _sync_project_result(self, turn: RawTurn, result: Any) -> None:
        """Synchronous projection run on a single thread to guarantee atomic transaction."""
        ns = turn.namespace
        with self._backend.transaction():
            for fact in getattr(result, "facts", []) or []:
                if fact.certainty is Certainty.VAGUE:
                    self._candidate_inbox.add(ns, fact)
                    continue
                self._entity_state.upsert(
                    ns,
                    fact.subject,
                    fact.predicate,
                    str(fact.object),
                    certainty=fact.certainty,
                    valid_from=fact.valid_from,
                    source_unit_id=fact.source_anchor,
                    qualifiers=fact.qualifiers,
                    accumulate=getattr(fact, "accumulate", False),
                )
                self._promoter.promote(turn, fact)

            for raw in getattr(result, "raw_sourceless", []) or []:
                payload = raw if isinstance(raw, dict) else {"item": str(raw)}
                self._candidate_inbox.add_sourceless(ns, payload)


__all__ = ["ExtractorWorker", "ExtractorWorkerConfig"]
