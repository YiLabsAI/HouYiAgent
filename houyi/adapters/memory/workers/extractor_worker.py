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
import re
from dataclasses import dataclass
from typing import Any, Protocol

from houyi.adapters.memory.fact_identity import fact_record_id
from houyi.adapters.memory.fact_promoter import FactPromoter, MemoryRecordPromoter
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    MemoryEdge,
    MemoryEvent,
    MemoryRecord,
    MemoryRelation,
    RawTurn,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timestamp normalization for narrative chain ordering
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(\d{4})\b")


def _normalize_timestamp_for_sort(ts: str) -> tuple:
    """Derive a sortable key from a free-form event timestamp.

    Strategy:
    1. If the timestamp contains a recognizable 4-digit year, extract it as
       the primary sort key. Month and day are secondary keys if present.
    2. Otherwise, return a high sentinel value so unparseable timestamps
       sort after well-structured ones, preserving extraction order for
       same-batch events with no parseable year.
    """
    if not ts:
        return (999999, 99, 99)
    year_match = _YEAR_RE.search(ts)
    if year_match:
        year = int(year_match.group(1))
        # Try to extract month (1-12) and day from common patterns
        month_day_re = re.compile(r"\b(\d{1,2})[/-](\d{1,2})\b")
        md = month_day_re.search(ts)
        month = int(md.group(1)) if md and 1 <= int(md.group(1)) <= 12 else 99
        day = int(md.group(2)) if md and 1 <= int(md.group(2)) <= 31 else 99
        return (year, month, day)
    return (999999, 99, 99)


_PREDICATE_TO_RELATION: dict[str, MemoryRelation] = {
    # Fact predicates only -- state/attribute verbs extracted as AtomicFacts.
    # Event action verbs (watched, adopted, moved_to, purchased, started, lost,
    # passed_away, married, divorced, enrolled, quit, traveled_to, donated, sold,
    # bought, acquired, switched_to, started_job, lost_job, lost_family_member,
    # lost_friend) are NOT here by design: events use structural edges
    # (PARTICIPATES_IN, INVOLVES, NARRATIVE_NEXT) instead of predicate-driven
    # edges. See _wire_event_edges and _wire_narrative_next_chains.
    #
    # Relational & Social
    "knows": MemoryRelation.RELATED_TO,
    "likes": MemoryRelation.RELATED_TO,
    "friends with": MemoryRelation.RELATED_TO,
    "married to": MemoryRelation.RELATED_TO,
    "works with": MemoryRelation.RELATED_TO,
    "related to": MemoryRelation.RELATED_TO,
    "child of": MemoryRelation.RELATED_TO,
    "parent of": MemoryRelation.RELATED_TO,
    "spouse of": MemoryRelation.RELATED_TO,
    "friend": MemoryRelation.RELATED_TO,
    "husband": MemoryRelation.RELATED_TO,
    "wife": MemoryRelation.RELATED_TO,
    "son": MemoryRelation.RELATED_TO,
    "daughter": MemoryRelation.RELATED_TO,
    "brother": MemoryRelation.RELATED_TO,
    "sister": MemoryRelation.RELATED_TO,
    "employed by": MemoryRelation.RELATED_TO,
    "works at": MemoryRelation.RELATED_TO,
    "partner of": MemoryRelation.RELATED_TO,
    # Contradiction / Invalidation
    "contradicts": MemoryRelation.CONTRADICTS,
    "invalidates": MemoryRelation.INVALIDATES,
    "updates": MemoryRelation.UPDATES,
    "replaces": MemoryRelation.REPLACES,
    # Supporting
    "supports": MemoryRelation.SUPPORTS,
    "backs up": MemoryRelation.SUPPORTS,
    "helps": MemoryRelation.SUPPORTS,
    "assists": MemoryRelation.SUPPORTS,
    "encourages": MemoryRelation.SUPPORTS,
    "promotes": MemoryRelation.SUPPORTS,
    # Identity / Same
    "same as": MemoryRelation.SAME_AS,
    "alias of": MemoryRelation.SAME_AS,
    "identity of": MemoryRelation.SAME_AS,
    "identical to": MemoryRelation.SAME_AS,
    "equivalent to": MemoryRelation.SAME_AS,
}


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
    def add_edge(self, edge: MemoryEdge) -> None: ...
    def get_by_id(self, record_id: str) -> MemoryRecord | None: ...


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
        accumulate: bool = False,
    ) -> Any: ...
    def get_active(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[Any]: ...


class _CandidateInboxProtocol(Protocol):
    def add(self, namespace: str, fact: AtomicFact) -> str: ...
    def add_sourceless(self, namespace: str, raw_payload: dict[str, Any]) -> str: ...


class _EventViewProtocol(Protocol):
    def add_event(self, event: MemoryEvent) -> MemoryEvent: ...


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


def _is_valid_identity_candidate(s: str, require_proper: bool = False) -> bool:
    """Filter out empty, pure numbers, and extremely short/noisy candidates.

    If require_proper is True, also enforces that the candidate starts with
    an uppercase letter or contains CJK characters to avoid common noun pollution.
    """
    s_clean = s.strip().strip("'\"()")
    if not s_clean:
        return False
    # Avoid pure numbers (like years, days, counts) as entity anchors
    if s_clean.replace(".", "").isdigit():
        return False
    # Avoid extremely short/noisy words
    if s_clean.lower() in (
        "a",
        "an",
        "the",
        "it",
        "i",
        "he",
        "she",
        "we",
        "they",
        "this",
        "that",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "of",
        "to",
    ):
        return False

    if require_proper:
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", s_clean))
        if not (s_clean[0].isupper() or has_cjk):
            return False

    return True


def _is_verb_echo_fact(fact: AtomicFact) -> bool:
    """True when the object merely echoes a word already in the predicate.

    The LLM occasionally emits degenerate triples like
    (Evan, started_painting, painting) where the object repeats the
    predicate's verb/gerund and adds no information. Such facts pollute
    recall by reinforcing the generic verb over concrete content while
    contributing nothing new, so the worker drops them on projection.

    Only single-token objects are considered; multi-word objects always
    carry additional content (e.g. 'painting of a forest scene') and are
    never treated as echoes.
    """
    obj = str(fact.object).strip().lower()
    if not obj or " " in obj:
        return False
    pred_words = {w for w in re.split(r"[\s_]+", fact.predicate.strip().lower()) if w}
    return obj in pred_words


class ExtractorWorker:
    """Drain the extract queue and project facts onto the state view."""

    def __init__(
        self,
        *,
        backend: _BackendProtocol,
        extractor: _ExtractorProtocol,
        entity_state: _EntityStateProtocol,
        candidate_inbox: _CandidateInboxProtocol,
        event_view: _EventViewProtocol | None = None,
        promoter: FactPromoter | None = None,
        config: ExtractorWorkerConfig | None = None,
        write_lock: asyncio.Lock | None = None,
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
        write_lock: optional asyncio.Lock shared by all drain workers
        that target the same backend. SQLite is single-writer, so
        running multiple projection transactions concurrently on
        separate thread-local connections raises 'database is locked'.
        When supplied, the lock serializes only the write phases
        (claim + projection); the LLM extraction stays outside it so
        several drain workers can saturate the extractor in parallel.
        Leaving it None preserves the original unsynchronized behavior.
        """
        if backend is None or extractor is None:
            raise ValueError("backend and extractor are required")
        if entity_state is None or candidate_inbox is None:
            raise ValueError("entity_state and candidate_inbox are required")
        self._backend = backend
        self._extractor = extractor
        self._entity_state = entity_state
        self._candidate_inbox = candidate_inbox
        self._event_view = event_view
        # Default to MemoryRecordPromoter so the L1 worker materializes
        # MemoryRecord rows that the vector path can index. Callers that
        # need to disable promotion pass a noop FactPromoter explicitly.
        self._promoter: FactPromoter = promoter or MemoryRecordPromoter(backend)
        self._config = config or ExtractorWorkerConfig()
        self._write_lock = write_lock

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

    def _write_guard(self) -> Any:
        """Serialize SQLite write phases across concurrent drain workers.

        Returns the shared write lock when one was supplied, else a no-op
        async context manager so single-worker callers are unaffected.
        """
        if self._write_lock is not None:
            return self._write_lock
        return contextlib.nullcontext()

    async def process_once(self) -> int:
        """Claim one batch and process every job in it.

        Returns the number of jobs that were processed (regardless of
        success/failure outcome). A return value of 0 means the queue
        was empty at claim time and the caller should sleep.
        """
        cfg = self._config
        async with self._write_guard():
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
        # Turns within a claimed batch share the same namespace (a worker
        # claims jobs for one session/namespace). Take the namespace from the
        # first turn so extracted events are written under the query namespace
        # and become visible to the EventRetriever (which queries by the same
        # namespace). Falls back to "default" only when a turn lacks one.
        batch_namespace = claimed[0][1].namespace if claimed else "default"
        if not batch_namespace:
            batch_namespace = "default"
        payload = [(_extract_text_for(turn), _source_anchor_for(turn)) for _, turn in claimed]
        try:
            results = await batch_extractor.extract_batch(payload, namespace=batch_namespace)
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

        async with self._write_guard():
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

        async with self._write_guard():
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
            batch_map: dict[str, str] = {}

            for fact_idx, fact in enumerate(getattr(result, "facts", []) or []):
                if fact.certainty is Certainty.VAGUE:
                    self._candidate_inbox.add(ns, fact)
                    continue
                if _is_verb_echo_fact(fact):
                    logger.info(
                        "Skipping verb-echo fact: %s | %s | %s",
                        fact.subject,
                        fact.predicate,
                        fact.object,
                    )
                    continue
                state_record = self._entity_state.upsert(
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

                # Compute record_id for memories
                anchor = fact.source_anchor or ""
                record_id = fact_record_id(fact.subject, fact.predicate, fact.object, anchor)

                # Register mappings for edge resolution
                batch_map[f"{fact.subject}.{fact.predicate}"] = state_record.state_id
                batch_map[fact.subject] = state_record.state_id
                batch_map[f"fact-{fact_idx}"] = record_id
                batch_map[record_id] = record_id

                # Auto-derive MemoryEdge from high-certainty atomic fact (subject, predicate, object) triples
                self._auto_derive_edges(ns, fact, state_record, batch_map)

            # Process events
            batch_events: list[MemoryEvent] = []
            for event in getattr(result, "events", []) or []:
                if not isinstance(event, MemoryEvent):
                    continue
                if event.certainty is Certainty.VAGUE:
                    self._candidate_inbox.add(
                        ns,
                        AtomicFact(
                            subject=event.subject,
                            predicate=event.action,
                            object=f"{event.object} ({event.timestamp})",
                            certainty=Certainty.VAGUE,
                            source_anchor=event.source_anchor,
                        ),
                    )
                    continue
                if self._event_view is not None:
                    stored_event = self._event_view.add_event(event)
                else:
                    stored_event = event
                batch_map[event.event_id] = stored_event.event_id
                batch_map[f"event:{event.subject}.{event.action}.{event.object}"] = (
                    stored_event.event_id
                )
                batch_events.append(stored_event)
                self._wire_event_edges(ns, stored_event, batch_map)

            if batch_events:
                self._wire_narrative_next_chains(ns, batch_events, batch_map)

            for raw in getattr(result, "raw_sourceless", []) or []:
                payload = raw if isinstance(raw, dict) else {"item": str(raw)}
                self._candidate_inbox.add_sourceless(ns, payload)

    def _auto_derive_edges(
        self, ns: str, fact: AtomicFact, state_record: Any, batch_map: dict[str, str]
    ) -> None:
        import time

        obj_str = str(fact.object).strip()
        if obj_str and len(obj_str.split()) < 4:
            pred_clean = fact.predicate.strip().lower()
            relation = _PREDICATE_TO_RELATION.get(pred_clean, MemoryRelation.RELATED_TO)
            source_id = state_record.state_id
            target_id = batch_map.get(obj_str)
            if not target_id:
                active_target = self._entity_state.get_active(ns, obj_str)
                if active_target:
                    target_id = active_target[0].state_id
                else:
                    if not _is_valid_identity_candidate(obj_str):
                        return  # Skip creating identity anchor and auto-derived edge for noisy target objects
                    # Create identity anchor for the target object
                    logger.info(
                        "Auto-deriving identity anchor for target object entity: %s", obj_str
                    )
                    target_rec = self._entity_state.upsert(
                        ns,
                        obj_str,
                        "identity",
                        obj_str,
                        certainty=Certainty.CERTAIN,
                        valid_from=fact.valid_from or time.time(),
                        source_unit_id=fact.source_anchor,
                        qualifiers=None,
                    )
                    target_id = target_rec.state_id
                    batch_map[obj_str] = target_id

            if source_id and target_id and source_id != target_id:
                edge_key = f"edge:{ns}|{source_id}|{target_id}|{relation.value}"
                if edge_key not in batch_map:
                    auto_edge = MemoryEdge(
                        namespace=ns,
                        source_unit_id=source_id,
                        target_unit_id=target_id,
                        source_type="state",
                        target_type="state",
                        relation=relation,
                        valid_from=fact.valid_from or time.time(),
                        provenance=fact.source_anchor,
                    )
                    self._backend.add_edge(auto_edge)
                    batch_map[edge_key] = "1"

    def _wire_event_edges(self, ns: str, event: MemoryEvent, batch_map: dict[str, str]) -> None:
        import time

        event_id = event.event_id
        subject = event.subject.strip()

        # Edge 1: Entity(state) -> PARTICIPATES_IN -> Event
        subject_state_id = batch_map.get(subject)
        if not subject_state_id:
            active = self._entity_state.get_active(ns, subject)
            if active:
                subject_state_id = active[0].state_id
                batch_map[subject] = subject_state_id
            else:
                if _is_valid_identity_candidate(subject):
                    logger.info("Creating identity anchor for event subject: %s", subject)
                    anchor_rec = self._entity_state.upsert(
                        ns,
                        subject,
                        "identity",
                        subject,
                        certainty=Certainty.CERTAIN,
                        valid_from=event.valid_from or time.time(),
                        source_unit_id=event.source_anchor,
                        qualifiers=None,
                    )
                    subject_state_id = anchor_rec.state_id
                    batch_map[subject] = subject_state_id

        if subject_state_id:
            edge_key = f"edge:{ns}|{subject_state_id}|{event_id}|participates_in"
            if edge_key not in batch_map:
                participates_edge = MemoryEdge(
                    namespace=ns,
                    source_unit_id=subject_state_id,
                    target_unit_id=event_id,
                    source_type="state",
                    target_type="event",
                    relation=MemoryRelation.PARTICIPATES_IN,
                    valid_from=event.valid_from or time.time(),
                    provenance=event.source_anchor,
                )
                self._backend.add_edge(participates_edge)
                batch_map[edge_key] = "1"

        # Edge 2: Event -> INVOLVES -> Entity(state) (only if object resolves to existing entity)
        obj_str = event.object.strip()
        object_state_id = batch_map.get(obj_str)
        if not object_state_id:
            active_obj = self._entity_state.get_active(ns, obj_str)
            if active_obj:
                object_state_id = active_obj[0].state_id
                batch_map[obj_str] = object_state_id

        if object_state_id:
            edge_key = f"edge:{ns}|{event_id}|{object_state_id}|involves"
            if edge_key not in batch_map:
                involves_edge = MemoryEdge(
                    namespace=ns,
                    source_unit_id=event_id,
                    target_unit_id=object_state_id,
                    source_type="event",
                    target_type="state",
                    relation=MemoryRelation.INVOLVES,
                    valid_from=event.valid_from or time.time(),
                    provenance=event.source_anchor,
                )
                self._backend.add_edge(involves_edge)
                batch_map[edge_key] = "1"

    def _wire_narrative_next_chains(
        self, ns: str, events: list[MemoryEvent], batch_map: dict[str, str]
    ) -> None:
        import time
        from collections import defaultdict

        groups = defaultdict(list)
        for event in events:
            groups[(ns, event.subject.strip())].append(event)

        for (_group_ns, _subject), group_events in groups.items():
            if len(group_events) < 2:
                continue
            # Sort by occurrence time (event.timestamp) rather than system
            # insertion time (valid_from). When timestamps are free-form
            # strings that cannot be meaningfully compared, preserve the
            # extraction order (which often reflects narrative sequence).
            sorted_events = sorted(
                group_events,
                key=lambda e: _normalize_timestamp_for_sort(e.timestamp),
            )
            for i in range(len(sorted_events) - 1):
                source_event = sorted_events[i]
                target_event = sorted_events[i + 1]
                edge_key = (
                    f"edge:{ns}|{source_event.event_id}|{target_event.event_id}|narrative_next"
                )
                if edge_key in batch_map:
                    continue
                next_edge = MemoryEdge(
                    namespace=ns,
                    source_unit_id=source_event.event_id,
                    target_unit_id=target_event.event_id,
                    source_type="event",
                    target_type="event",
                    relation=MemoryRelation.NARRATIVE_NEXT,
                    valid_from=source_event.valid_from or time.time(),
                    provenance=source_event.source_anchor,
                )
                self._backend.add_edge(next_edge)
                batch_map[edge_key] = "1"


__all__ = ["ExtractorWorker", "ExtractorWorkerConfig"]
