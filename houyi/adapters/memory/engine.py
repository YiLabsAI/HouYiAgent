"""MemoryEngine: unified industrial-grade memory facade.

The engine bundles the write pipeline (turn ingest, extractor and
embedding-backfill workers), the recall pipeline (RecallOrchestrator),
and the legacy candidate-extraction pipeline behind a single object
with a small public API:

  - write_turn(turn): fast-path L0 write plus L1 enqueue.
  - flush(timeout):   wait until all pending L1 extractions and
                      embedding backfills finish.
  - recall(query):    return MemoryRecall hits via the orchestrator
                      when one is wired, falling back to the legacy
                      MemoryRetriever otherwise.
  - answer(query):    recall plus reasoning policies.
  - start(), stop():  manage the background workers.
  - async with engine: idiomatic lifecycle for callers that do not
                      want to call start/stop manually.

Direct construction of MemoryEngine is reserved for tests and
advanced callers. Production code MUST use
build_memory_engine in houyi.adapters.memory.factory which assembles
the full default stack (backend, workers, orchestrator, embedding).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
import time
from typing import Any

from houyi.adapters.embedding import EmbeddingProvider
from houyi.adapters.memory.answerer import AnswerResult
from houyi.adapters.memory.builder import MemoryCandidateBuilder
from houyi.adapters.memory.classifier import MemoryClassifier
from houyi.adapters.memory.deduplicator import MemoryDeduplicator
from houyi.adapters.memory.event_emitter import MemoryEventEmitter
from houyi.adapters.memory.extractor import MemoryCandidateExtractor
from houyi.adapters.memory.forgetting import apply_forgetting
from houyi.adapters.memory.reasoner import (
    DeterministicReasoningPolicy,
    LLMMemoryReasoningPolicy,
    MemoryReasoner,
    ReasoningPolicy,
)
from houyi.adapters.memory.recall.enumeration import detect_enumeration_category
from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.retriever import MemoryRetriever
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.turn_writer import TurnWriter, WriteResult
from houyi.adapters.memory.types import (
    CandidateStatus,
    ExtractionContext,
    ForgettingPolicy,
    MemoryBuildInput,
    MemoryBuildItem,
    MemoryCandidate,
    MemoryPolicy,
    MemoryProvenance,
    MemoryRecall,
    MemoryRecord,
    MemoryScope,
    MemorySourceKind,
    RawTurn,
    RecallMatchMethod,
    RelevanceDetail,
    SessionContext,
)
from houyi.adapters.memory.workers.embedding_backfill import EmbeddingBackfillWorker
from houyi.adapters.memory.workers.extractor_worker import ExtractorWorker
from houyi.application.evolution.events import EvolutionEventType

# Maps a recall-layer RetrieverKind to the legacy RecallMatchMethod enum
# carried on MemoryRecall. Both axes are coarse; the mapping favors the
# closest semantic neighbor and falls back to HYBRID for anything that
# does not have a clean equivalent.
_RETRIEVER_KIND_TO_MATCH_METHOD: dict[RetrieverKind, RecallMatchMethod] = {
    RetrieverKind.ENTITY_STATE: RecallMatchMethod.RULE,
    RetrieverKind.VECTOR: RecallMatchMethod.EMBEDDING,
    RetrieverKind.RAW_TURN: RecallMatchMethod.LEXICAL,
    RetrieverKind.TIMELINE: RecallMatchMethod.RULE,
    RetrieverKind.ITERATIVE: RecallMatchMethod.HYBRID,
}

# Qualifier keys that are internal bookkeeping, not answer evidence. They
# must never be rendered into the answerer prompt: the LLM echoes whatever
# parenthetical metadata it sees, so leaking these produces answers like
# "... (time: 2023-03-25, compound_type: emotional_transition)". The
# extractor stamps compound facts with compound_type/original_time and the
# promoter copies the raw triple into fact_subject/predicate/object; none of
# these help answer a question and are stripped before prompting.
_INTERNAL_QUALIFIER_KEYS: frozenset[str] = frozenset(
    {
        "compound_type",
        "original_time",
        "fact_subject",
        "fact_predicate",
        "fact_object",
    }
)

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Unified memory facade — write pipeline + recall pipeline.

    Components are optional; the engine degrades gracefully:
    - No embedding_provider → lexical-only retrieval, no semantic dedup
    - No extractor → process_messages() returns empty list
    - No classifier → candidates keep their initial type
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        builder: MemoryCandidateBuilder | None = None,
        extractor: MemoryCandidateExtractor | None = None,
        classifier: MemoryClassifier | None = None,
        deduplicator: MemoryDeduplicator | None = None,
        retriever: MemoryRetriever | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        llm_adapter: Any | None = None,
        reasoner: MemoryReasoner | None = None,
        reasoning_policies: list[ReasoningPolicy] | None = None,
        policy: MemoryPolicy | None = None,
        forgetting_policy: ForgettingPolicy | None = None,
        emitter: MemoryEventEmitter | None = None,
        # New facade dependencies. All optional so existing tests and
        # studio code that build a MemoryEngine by hand still work; the
        # canonical entry point that fills these in is build_memory_engine.
        recall_orchestrator: RecallOrchestrator | None = None,
        turn_writer: TurnWriter | None = None,
        extractor_worker: ExtractorWorker | None = None,
        backfill_worker: EmbeddingBackfillWorker | None = None,
    ):
        self._store = store
        self._builder = builder or MemoryCandidateBuilder(llm_adapter=llm_adapter)
        self._extractor = extractor or MemoryCandidateExtractor(llm_adapter=llm_adapter)
        self._classifier = classifier or MemoryClassifier()
        self._deduplicator = deduplicator or MemoryDeduplicator(embedding_provider)
        self._retriever = retriever or MemoryRetriever(
            store,
            embedding_provider,
            policy,
        )
        self._embedding = embedding_provider
        self._policy = policy or MemoryPolicy()
        self._forgetting = forgetting_policy or ForgettingPolicy()
        self._reasoner = reasoner or self._build_reasoner(
            llm_adapter=llm_adapter,
            policies=reasoning_policies,
        )
        # Optional hot-path event emitter. The engine emits RECALL_FAILURE
        # when its retriever returns no recalls so the legacy retrieval
        # path still produces evolution signals on top of the new
        # RecallOrchestrator path. Production wiring may share a single
        # MemoryEventEmitter across both surfaces.
        self._emitter = emitter or MemoryEventEmitter()

        # Facade members. The orchestrator wins over the legacy retriever
        # whenever both are available; the workers are optional and only
        # affect the lifecycle methods (start, stop, flush, write_turn).
        self._recall_orchestrator: RecallOrchestrator | None = recall_orchestrator
        self._turn_writer: TurnWriter | None = turn_writer
        self._extractor_worker: ExtractorWorker | None = extractor_worker
        self._backfill_worker: EmbeddingBackfillWorker | None = backfill_worker
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._worker_stops: list[asyncio.Event] = []
        # The flush implementation needs to query the SQLite-level extract
        # queue. The legacy MemoryStore wraps the backend via a private
        # attribute; cache the handle once so the hot path stays cheap.
        self._backend = getattr(store, "_backend", None)

    @staticmethod
    def _build_reasoner(
        *,
        llm_adapter: Any | None,
        policies: list[ReasoningPolicy] | None,
    ) -> MemoryReasoner:
        if policies is not None:
            return MemoryReasoner(policies=policies)

        default_policies: list[ReasoningPolicy] = [DeterministicReasoningPolicy()]
        if llm_adapter is not None:
            default_policies.append(LLMMemoryReasoningPolicy(llm_adapter))
        return MemoryReasoner(default_policies)

    @property
    def store(self) -> MemoryStore:
        """Access the underlying store directly."""
        return self._store

    # ------------------------------------------------------------------
    # Write pipeline
    # ------------------------------------------------------------------

    async def process_messages(
        self,
        messages: list[dict],
        context: ExtractionContext | None = None,
    ) -> list[MemoryCandidate]:
        """Extract → classify → dedup → optionally store.

        Returns candidates (approved ones are already written to store).
        """
        ctx = context or ExtractionContext()
        memory_input = MemoryBuildInput(
            source_type=MemorySourceKind.CONVERSATION,
            scope=MemoryScope.USER,
            source_context=f"turn:{ctx.turn_index}",
            items=[
                MemoryBuildItem(
                    content=str(message.get("content", "")),
                    role=str(message.get("role", "")),
                    source_ids=[str(message.get("id", ""))] if message.get("id") else [],
                )
                for message in messages
            ],
            metadata={"suggested_tags": ctx.active_tags},
        )
        return await self.process_input(memory_input, ctx)

    async def process_input(
        self,
        memory_input: MemoryBuildInput,
        context: ExtractionContext | None = None,
    ) -> list[MemoryCandidate]:
        """Build → classify → dedup → optionally store."""
        existing = self._store.all_records()
        candidates = await self._builder.build(memory_input, context)
        if not candidates:
            return []

        for candidate in candidates:
            candidate.memory_type = await self._classifier.classify(candidate)
            candidate.dedup_matches = await self._deduplicator.check(
                candidate,
                existing,
            )

            if candidate.dedup_matches:
                has_dup = any(m.relation == "duplicate" for m in candidate.dedup_matches)
                if has_dup:
                    candidate.status = CandidateStatus.MERGED
                    continue

            if self._policy.auto_approve:
                candidate.status = CandidateStatus.APPROVED
                await self._store_candidate(candidate)
            else:
                candidate.status = CandidateStatus.PENDING

        logger.debug(
            "Processed %d input items → %d candidates",
            len(memory_input.items),
            len(candidates),
        )
        return candidates

    async def approve_candidate(self, candidate: MemoryCandidate) -> MemoryRecord:
        """Approve and persist a pending candidate."""
        candidate.status = CandidateStatus.APPROVED
        return await self._store_candidate(candidate)

    async def _store_candidate(self, candidate: MemoryCandidate) -> MemoryRecord:
        """Convert an approved candidate into a MemoryRecord and store it."""
        record = MemoryRecord(
            scope=candidate.scope,
            key=self._derive_key(candidate),
            content=candidate.content,
            memory_type=candidate.memory_type,
            metadata=candidate.metadata,
            tags=candidate.suggested_tags,
            confidence=candidate.confidence,
            provenance=MemoryProvenance(
                source_type=candidate.source_type,
                source_ids=candidate.source_message_ids,
                extracted_by="memory_candidate_builder",
                extraction_timestamp=candidate.extracted_at,
            ),
            embedding=None,
        )
        self._store.put_record(record)

        return record

    def _cache_embedding(
        self,
        record_id: str,
        provider: EmbeddingProvider,
        embedding: list[float],
    ) -> None:
        """Write embedding to the backend's embedding_cache if supported."""
        backend = getattr(self._store, "_backend", None)
        if backend is None:
            return
        provider_name = type(provider).__name__
        model_name = str(provider.dimension())
        try:
            backend.put_embedding(record_id, provider_name, model_name, embedding)
        except Exception:
            logger.debug("Embedding cache write skipped", exc_info=True)

    @staticmethod
    def _derive_key(candidate: MemoryCandidate) -> str:
        """Derive a storage key from candidate content."""
        words = candidate.content.split()[:5]
        key = "_".join(w.lower().strip(".,!?:;") for w in words if w.strip())
        return key or candidate.candidate_id

    # ------------------------------------------------------------------
    # Recall pipeline
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        session_context: SessionContext | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecall]:
        """Retrieve relevant memories for a query.

        When a RecallOrchestrator is wired into the engine the call is
        delegated there and the resulting RecallCandidate list is
        adapted to MemoryRecall so existing callers stay unchanged.
        Otherwise the legacy MemoryRetriever path is used.
        """
        if self._recall_orchestrator is not None:
            recalls = await self._recall_via_orchestrator(query, top_k, session_context)
        else:
            recalls = await self._retriever.retrieve(query, session_context, top_k)
        if not recalls:
            # Empty recall = retrieval miss on the legacy path. Surface to
            # the evolution control plane the same way the new orchestrator
            # does, so signal mining is path-uniform.
            self._emitter.emit(
                EvolutionEventType.RECALL_FAILURE,
                target="memory_engine",
                payload={
                    "query_preview": query[:200],
                    "reason": "no_candidates",
                },
                metrics={"top_k": float(top_k)},
            )
        return recalls

    async def build_context(
        self,
        query: str,
        session_context: SessionContext | None = None,
        top_k: int = 5,
    ) -> str | None:
        """Recall relevant memories and format as context text.

        Returns None if no relevant memories found, allowing callers
        (e.g. ContextPlanner) to omit the memory block entirely.
        """
        recalls = await self.recall(query, session_context, top_k)
        if not recalls:
            return None
        text = self.recall_as_context_text(recalls)
        return text or None

    async def answer(
        self,
        query: str,
        session_context: SessionContext | None = None,
        top_k: int = 5,
    ) -> AnswerResult:
        """Answer a query directly from memory recall + reasoning policies.

        The caller's top_k is honored verbatim: it is both the recall
        breadth and the recall@k metric basis. The reasoner applies
        its own max_facts cap downstream, so the facts the LLM reads
        are a prefix of the same ranked set the caller measured.
        """
        recall_t0 = time.perf_counter()

        # Dynamic Recall Scaling for aggregation/enumeration queries
        # (e.g. how many, what books, list all). We temporarily expand the
        # underlying recall bandwidth so the full member set for the subject
        # reaches fusion/coverage selection, but we slice the returned recalls
        # back to top_k to keep evaluation metrics (recall@k, mrr) clean.
        #
        # Enumeration intent is detected structurally (subject + plural
        # category, "kinds/types of", "how many", "list ...") by the same
        # detector the recall coverage path uses, so the budget and the
        # coverage selector stay in lockstep instead of drifting apart on two
        # separate keyword lists. The remaining literals are NOT category
        # enumerations and therefore have no plural category head for the
        # structural detector to find ("dietary restrictions", "allergies",
        # frequency "how often"); they stay as explicit fallbacks.
        is_agg = detect_enumeration_category(query) is not None or any(
            w in query.lower()
            for w in (
                "how often",
                "dietary restrictions",
                "allergic to",
                "allergies",
                "sensitivities",
            )
        )
        recall_top_k = max(top_k * 3, 30) if is_agg else top_k
        recalls = await self.recall(query, session_context, recall_top_k)
        recall_ms = (time.perf_counter() - recall_t0) * 1000.0

        # Build the record index once instead of scanning all_records()
        # per recall: turns O(top_k * N) into O(N) + O(top_k) lookups.
        record_index = self._build_record_index()
        records: list[MemoryRecord] = []
        for recall in recalls:
            record = record_index.get(recall.memory_id)
            if record is not None:
                content = self._reformat_recall_content(record.content, recall.qualifiers)
                record = record.model_copy(update={"content": content})
                records.append(record)
        records = self._prepare_reasoner_records(records)

        obs_date = (
            getattr(session_context, "current_observation_date", None) if session_context else None
        )
        sys_date = (
            getattr(session_context, "current_system_date", None) if session_context else None
        )

        res = await self._reasoner.answer(
            query, recalls, records, current_observation_date=obs_date, current_system_date=sys_date
        )
        import dataclasses

        # Strictly slice the recalls back to standard top_k to keep evaluation clean
        evaluation_recalls = recalls[:top_k]
        return dataclasses.replace(
            res, extras={**res.extras, "recalls": evaluation_recalls, "recall_ms": recall_ms}
        )

    _CONTENT_FREE_RELATION_RE = re.compile(
        r"^\s*[\w'.-]+\s+"
        r"(?:shares?\s+(?:interests?|activit(?:y|ies))\s+with|(?:is\s+)?related\s+to|knows)"
        r"\s+[\w'.-]+\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def _prepare_reasoner_records(cls, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Deduplicate ranked records and demote content-free relational facts.

        Recall frequently surfaces the same rendered fact several times
        (one hit per retriever route) plus bare relational facts such as
        "A shares interest with B" that carry no answerable content.
        Both waste the reasoner's bounded fact budget and crowd out
        substantive facts. Duplicates (by normalized content) are dropped
        keeping the highest-ranked copy; content-free relational facts
        are kept but moved behind every substantive fact so they only
        consume budget last. Ranking order is otherwise preserved.
        """
        deduped: list[MemoryRecord] = []
        seen: set[str] = set()
        for record in records:
            normalized = re.sub(r"\s+", " ", (record.content or "").strip().lower())
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(record)
        substantive = [
            r for r in deduped if not cls._CONTENT_FREE_RELATION_RE.match(r.content or "")
        ]
        relational = [r for r in deduped if cls._CONTENT_FREE_RELATION_RE.match(r.content or "")]
        return substantive + relational

    def recall_as_context_text(
        self,
        recalls: list[MemoryRecall],
    ) -> str:
        """Render recall results as context text for ContextPlanner."""
        if not recalls:
            return ""
        # Build the record index once instead of an all_records() scan per
        # recall, matching answer()'s O(N) + O(top_k) resolution.
        record_index = self._build_record_index()
        lines = []
        for r in recalls:
            record = record_index.get(r.memory_id)
            if record:
                lines.append(
                    f"- [{record.memory_type.value}] {record.key}: "
                    f"{record.content} (score={r.score:.2f})"
                )
        return "\n".join(lines)

    def _build_record_index(self) -> dict[str, MemoryRecord]:
        """Map every record_id and its fact-id aliases to the record.

        Built once per answer() so recall resolution is O(1) per hit
        instead of an all_records() scan. Exact record_id keys take
        precedence over derived fact-id aliases on collision.
        """
        records = list(self._store.all_records())
        index: dict[str, MemoryRecord] = {}
        # First pass: derived fact-id aliases (lower precedence).
        for record in records:
            for alias in (
                self._record_to_fact_id(record, "A"),
                self._record_to_fact_id(record, "B"),
            ):
                index.setdefault(alias, record)
        # Second pass: exact record_id wins on any collision.
        for record in records:
            index[record.record_id] = record
        return index

    @staticmethod
    def _reformat_recall_content(content: str, quals: dict[str, str] | None) -> str:
        m = re.search(r"\((?:time|date):\s*([^)]+)\)", content)
        cal_time = m.group(1).strip() if m else None
        if not quals and not cal_time:
            return content

        parts = []
        if cal_time:
            parts.append(f"time: {cal_time}")
        for qk, qv in sorted((quals or {}).items()):
            if not qv or (qk == "date" and cal_time):
                continue
            if qk in _INTERNAL_QUALIFIER_KEYS:
                # Internal bookkeeping (compound_type, original_time, raw
                # triple copies) is never answer evidence; rendering it only
                # leaks metadata into the LLM's echoed answer.
                continue
            lbl = "time" if qk == "date" else qk
            parts.append(f"{lbl}: {qv}")

        if parts:
            content_stripped = re.sub(r"\s*\([^)]*\)\s*$", "", content).strip()
            return f"{content_stripped} ({', '.join(parts)})"
        return content

    def _record_to_fact_id(self, record: MemoryRecord, strategy: str = "A") -> str:
        if strategy == "B":
            # Strategy B uses key as subject, "content" as predicate,
            # full content as object — a coarse fallback.
            subject = record.key
            predicate = "content"
            content = record.content
            anchor = record.record_id
        else:
            # Strategy A: derive subject|predicate|object|anchor from
            # the record to match the hash format used by
            # _candidate_to_memory_recall. When fact_promoter stored
            # the original AtomicFact fields in metadata, use those
            # directly — they are the authoritative source and avoid
            # fragile content-prefix-stripping heuristics.
            meta = record.metadata or {}
            if meta.get("fact_subject") and meta.get("fact_predicate") and meta.get("fact_object"):
                subject = meta["fact_subject"]
                predicate = meta["fact_predicate"]
                content = meta["fact_object"]
            else:
                # Fallback: reconstruct from key and content for
                # records created by _store_candidate (UUID record_id)
                # that lack metadata fact fields.
                parts = record.key.split(".", 2)
                subject = parts[0] if len(parts) > 1 else ""
                predicate = parts[1] if len(parts) > 1 else record.key
                content = record.content
                # Strip trailing parenthesized qualifiers like (time: ...)
                content = re.sub(r"\s*\([^)]*\)\s*$", "", content).strip()
                # Strip subject prefix so content matches AtomicFact.object
                if subject:
                    prefix = f"{subject} "
                    if content.lower().startswith(prefix.lower()):
                        content = content[len(prefix) :].lstrip()
                # Strip predicate prefix so content matches AtomicFact.object.
                # The predicate in the key uses underscores (e.g. lost_job)
                # but the content renders them as spaces (e.g. lost job).
                # Replace underscores before prefix matching so the strip
                # succeeds for multi-word predicates.
                if predicate:
                    display_pred = predicate.replace("_", " ")
                    prefix = f"{display_pred} "
                    if content.lower().startswith(prefix.lower()):
                        content = content[len(prefix) :].lstrip()
            anchor = (
                record.provenance.source_ids[0]
                if (record.provenance and record.provenance.source_ids)
                else ""
            )

        digest = hashlib.sha256(f"{subject}|{predicate}|{content}|{anchor}".encode()).hexdigest()[
            :24
        ]
        return f"fact:{digest}"

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def run_forgetting(self) -> int:
        """Apply forgetting policy and remove stale records.

        Returns the number of records evicted.
        """
        all_records = self._store.all_records()
        before = len(all_records)
        survivors = apply_forgetting(all_records, self._forgetting)
        evicted = before - len(survivors)

        if evicted > 0:
            survivor_ids = {r.record_id for r in survivors}
            for record in all_records:
                if record.record_id not in survivor_ids:
                    self._store.delete(record.key, record.scope)
            logger.info("Forgetting: evicted %d records", evicted)

        return evicted

    # ------------------------------------------------------------------
    # Facade — turn write, lifecycle, flush
    # ------------------------------------------------------------------

    async def write_turn(
        self,
        turn: RawTurn,
        *,
        schedule_extract: bool = True,
    ) -> WriteResult:
        """Persist one conversation turn through the layered write tiers.

        Delegates to the wired TurnWriter. Returns the WriteResult so
        callers can read back turn_index and queue_id without reaching
        for the underlying backend. Raises RuntimeError when no
        TurnWriter was supplied at construction time.
        """
        if self._turn_writer is None:
            raise RuntimeError(
                "MemoryEngine has no TurnWriter; build it via "
                "build_memory_engine or pass turn_writer=... explicitly."
            )
        # The fast_path is sync but does blocking SQLite work; dispatch
        # off-thread so the event loop is not blocked under load.
        return await asyncio.to_thread(
            self._turn_writer.fast_path,
            turn,
            schedule_extract=schedule_extract,
        )

    async def start(self) -> None:
        """Launch the background workers if any are wired. Idempotent.

        Safe to call multiple times. When the engine has no workers the
        method is a no-op so callers can use the same lifecycle code
        regardless of how the engine was assembled.
        """
        if self._worker_tasks:
            return
        if self._extractor_worker is not None:
            stop = asyncio.Event()
            self._worker_stops.append(stop)
            self._worker_tasks.append(
                asyncio.create_task(
                    self._extractor_worker.run_forever(stop),
                    name="memory-extractor-worker",
                )
            )
        if self._backfill_worker is not None:
            stop = asyncio.Event()
            self._worker_stops.append(stop)
            self._worker_tasks.append(
                asyncio.create_task(
                    self._backfill_worker.run_forever(stop),
                    name="memory-backfill-worker",
                )
            )

    async def stop(self) -> None:
        """Signal each running worker to stop and await graceful shutdown.

        Idempotent. Cancels tasks that ignore the stop event so a hung
        worker cannot block the shutdown path indefinitely.
        """
        try:
            if self._worker_tasks:
                for stop in self._worker_stops:
                    stop.set()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._worker_tasks, return_exceptions=True),
                        timeout=5.0,
                    )
                except TimeoutError:
                    for task in self._worker_tasks:
                        task.cancel()
                    await asyncio.gather(*self._worker_tasks, return_exceptions=True)
                finally:
                    self._worker_tasks.clear()
                    self._worker_stops.clear()
        finally:
            if self._backend is not None and hasattr(self._backend, "close"):
                with contextlib.suppress(Exception):
                    self._backend.close()

    async def __aenter__(self) -> MemoryEngine:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.stop()

    async def flush(self, *, timeout: float = 60.0) -> dict[str, int]:
        """Wait until the write pipeline is idle, then return.

        Snapshots the current extract-queue and embedding-pending state
        on entry, then polls (50ms backing off to 200ms) until both
        snapshots are fully drained. Returns counts of how many items
        finished while flush was waiting. Raises asyncio.TimeoutError
        when the deadline is reached (built-in TimeoutError) so callers
        can decide whether to retry or surface the failure.

        Safe to call concurrently from multiple tasks: each caller
        observes its own snapshot and the underlying workers are
        idempotent against repeated polls.
        """
        if self._backend is None:
            return {"extracted": 0, "backfilled": 0}
        extract_pending = await asyncio.to_thread(self._extract_pending_count)
        embedding_pending = await asyncio.to_thread(self._embedding_pending_count)
        extracted = 0
        backfilled = 0
        deadline = time.monotonic() + max(0.0, timeout)
        sleep_s = 0.05
        while True:
            current_extract = await asyncio.to_thread(self._extract_pending_count)
            current_embedding = await asyncio.to_thread(self._embedding_pending_count)
            if current_extract == 0 and current_embedding == 0:
                extracted = max(extracted, extract_pending)
                backfilled = max(backfilled, embedding_pending)
                return {"extracted": extracted, "backfilled": backfilled}
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"MemoryEngine.flush timed out after {timeout:.1f}s; "
                    f"extract_pending={current_extract}, "
                    f"embedding_pending={current_embedding}"
                )
            await asyncio.sleep(sleep_s)
            sleep_s = min(sleep_s * 1.5, 0.2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _recall_via_orchestrator(
        self,
        query: str,
        top_k: int,
        session_context: SessionContext | None = None,
    ) -> list[MemoryRecall]:
        assert self._recall_orchestrator is not None
        namespace = (session_context.session_id if session_context else None) or "default"
        recall_query = RecallQuery(text=query, top_k=top_k, namespace=namespace)
        result = await self._recall_orchestrator.recall(recall_query, RetrieverContext())
        return [_candidate_to_memory_recall(c) for c in result.candidates]

    def _extract_pending_count(self) -> int:
        """Return the number of extract-queue rows still in flight.

        pending plus in_progress: anything else (done, failed) is final
        and does not need a flush wait.
        """
        if self._backend is None:
            return 0
        with contextlib.suppress(Exception):
            stats = self._backend.extract_queue_stats()
            return int(stats.get("pending", 0)) + int(stats.get("in_progress", 0))
        return 0

    def _embedding_pending_count(self) -> int:
        """Return how many MemoryRecord rows still need a vector backfill.

        Uses a small fetch (limit=128) and a one-shot SELECT count fallback
        so the hot path stays cheap; the polling cadence keeps the cost
        bounded even when the backlog is large.
        """
        if self._backend is None:
            return 0
        with contextlib.suppress(Exception):
            pending = self._backend.list_pending_embeddings(limit=1)
            if not pending:
                return 0
        with contextlib.suppress(Exception):
            conn = self._backend._conn()
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE embedding_pending = 1"
            ).fetchone()
            return int(row["n"]) if row is not None else 0
        return 0


def _candidate_to_memory_recall(candidate: RecallCandidate) -> MemoryRecall:
    fact = candidate.fact
    anchor = fact.source_anchor or ""
    plain = f"{fact.subject}|{fact.predicate}|{fact.object}|{anchor}"
    digest = hashlib.sha256(plain.encode()).hexdigest()[:24]
    memory_id = f"fact:{digest}"
    matched_by = _RETRIEVER_KIND_TO_MATCH_METHOD.get(candidate.matched_by, RecallMatchMethod.HYBRID)
    quals = dict(fact.qualifiers) if fact.qualifiers else {}
    if fact.event_time and "date" not in quals:
        quals["date"] = fact.event_time
    # NOTE: deliberately no valid_from fallback here. valid_from is a
    # system timestamp (when the fact was recorded), not a fact-relevant
    # time (when the event occurred). Rendering a Unix epoch as a date
    # qualifier produces meaningless values like (time: 1781184471.94)
    # that the LLM cannot interpret. If no event_time is available, the
    # fact simply has no date qualifier — correct and better than
    # misleading the LLM.
    return MemoryRecall(
        memory_id=memory_id,
        score=float(candidate.score),
        matched_by=matched_by,
        explanation=candidate.explanation or f"{fact.subject} {fact.predicate} {fact.object}",
        relevance_detail=RelevanceDetail(),
        qualifiers=quals,
    )
