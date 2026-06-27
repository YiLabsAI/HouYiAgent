"""Failure-anchored, source-grounded, self-retrievability-judged reflection.

The generic extractor flattens rich source text into subject-scoped triples,
losing co-participation semantics (a turn "my girlfriend and I went to wine
tasting" becomes "Andrew went_to wine tasting" + "girlfriend likes wine
tasting" -- the joint participation is gone, so the answerer correctly
refuses to connect them). The reflector repairs this on demand: when recall
fails for a query, it re-extracts query-answering facts from the SOURCE
turns (whose full text still carries the lost semantics), grounds each
candidate against the source (no hallucination), and promotes only candidates
that are actually retrievable for the failing query.

This is the LLM counterpart to the deterministic consolidator
(dreamer_consolidate). Both run off the hot path inside
MemoryEngine.evolve: consolidate first repairs structural contradictions,
then reflect repairs semantic gaps for failing queries.

The reflection pipeline is fully async (MemoryReflector.reflect is async def)
so all I/O -- the LLM call, recall, and embedding backfill -- shares one
event loop. The engine bridges the async reflect into its sync evolve() via
a single _run_coro call (one thread, one loop), avoiding the per-call
thread spawning that could deadlock under cross-loop resource sharing.

Design references (positioned against memory substrates Mem0/Zep/Graphiti,
not task-agent self-evolution systems like ExpeL/Reflexion):
- Failure-anchored source re-extraction: no major memory substrate re-extracts
  from source turns for a specific failing query.
- Causal self-retrievability judge: inject the candidate, run real recall,
  check whether it surfaces in top-k -- directly measuring the property that
  matters, instead of an LLM opinion or a lexical-coverage tautology.
- Grounding gate: every reflected fact must be supported by a source turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
import time
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from houyi.adapters.memory.types import AtomicFact, Certainty, MemoryRecord, RawTurn

if TYPE_CHECKING:
    from houyi.adapters.llm.base import LLMAdapter, LLMMessage
    from houyi.adapters.memory.fact_promoter import FactPromoter
    from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator
    from houyi.adapters.memory.recall.types import RecallCandidate


def _run_coro(coro: Coroutine[Any, Any, Any]) -> Any:
    """Execute a coroutine whether or not an event loop is already running.

    When no loop is running, asyncio.run is used directly. When a loop IS
    running (the bench's async context), a single worker thread runs the
    coroutine in its own loop so the caller's loop is not nested. This is
    called ONCE per reflection run (not per-LLM-call) so all async I/O
    shares one thread/loop and avoids cross-loop deadlocks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: dict[str, Any] = {}

    def _worker() -> None:
        holder["value"] = asyncio.run(coro)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    return holder.get("value")


# ---------------------------------------------------------------------
# Tokenization (shared by the sampler and the grounding verifier)
# ---------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "many",
        "much",
        "of",
        "often",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "you",
        "your",
        "he",
        "her",
        "him",
        "his",
        "i",
        "me",
        "my",
        "our",
        "she",
        "theirs",
        "them",
        "they",
        "us",
        "we",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS)


# ---------------------------------------------------------------------
# Read-only seams (async)
# ---------------------------------------------------------------------


@runtime_checkable
class RecallProbe(Protocol):
    """Async read-only recall for the failing query."""

    async def recall(
        self, query: str, *, namespace: str, top_k: int = 10
    ) -> list[RecallCandidate]: ...


class _SyncRecallProbe:
    """Async recall bridge -- awaits RecallOrchestrator.recall directly."""

    def __init__(self, orchestrator: RecallOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def recall(self, query: str, *, namespace: str, top_k: int = 10) -> list[RecallCandidate]:
        from houyi.adapters.memory.recall.types import RecallQuery, RetrieverContext

        recall_query = RecallQuery(text=query, top_k=top_k, namespace=namespace)
        result = await self._orchestrator.recall(recall_query, RetrieverContext())
        return list(result.candidates)


@runtime_checkable
class SourceReader(Protocol):
    """Read-only access to the raw source turn log (sync -- SQLite is sync)."""

    def list_turns(self, namespace: str) -> list[RawTurn]: ...


class _BackendSourceReader:
    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def list_turns(self, namespace: str) -> list[RawTurn]:
        getter = getattr(self._backend, "list_raw_turns_by_namespace", None)
        if getter is None:
            return []
        return list(getter(namespace))


# ---------------------------------------------------------------------
# Reflected fact (pre-anchor): the reflector's raw output
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReflectedFact:
    subject: str
    predicate: str
    object: str
    event_time: str | None = None


# ---------------------------------------------------------------------
# 1. Sampler
# ---------------------------------------------------------------------


@runtime_checkable
class Sampler(Protocol):
    """Select the source turns a failing query is about."""

    async def sample(
        self,
        query: str,
        *,
        recall: RecallProbe,
        source_reader: SourceReader,
        namespace: str,
        top_k: int = 30,
    ) -> list[RawTurn]: ...


class RecallAnchoredSampler:
    """Find source turns via semantic recall, then per-fact best-turn match.

    For each recalled fact, picks the source turn with the highest fact-token
    overlap. Tiebreaker: highest query-token overlap -- the turn that matches
    BOTH the fact AND the query is the originator (it discusses the fact in
    the query's context, e.g. containing "girlfriend" when the query asks
    about activities "with girlfriend"). This is principled (query-fact
    joint relevance, no speaker heuristics) and picks D25:1 ("My girlfriend
    and I went wine tasting") over D25:2 ("glad you had fun at the wine
    tasting") because D25:1 shares more query tokens.
    """

    def __init__(self, *, max_turns: int = 10) -> None:
        self._max_turns = max_turns

    async def sample(
        self,
        query: str,
        *,
        recall: RecallProbe,
        source_reader: SourceReader,
        namespace: str,
        top_k: int = 30,
    ) -> list[RawTurn]:
        candidates = await recall.recall(query, namespace=namespace, top_k=top_k)
        if not candidates:
            return []
        fact_token_sets = [ts for ts in (_tokens(str(c.fact.object)) for c in candidates) if ts]
        if not fact_token_sets:
            return []
        query_tokens = _tokens(query)
        turns = source_reader.list_turns(namespace)
        best = _best_turn_per_fact(turns, fact_token_sets, query_tokens)
        seen: dict[int, int] = {}
        for overlap, _q, turn in best.values():
            tid = id(turn)
            if tid not in seen or overlap > seen[tid]:
                seen[tid] = overlap
        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        id_to_turn = {id(t): t for t in turns}
        return [id_to_turn[tid] for tid, _ in ranked[: self._max_turns]]


def _best_turn_per_fact(
    turns: list[RawTurn],
    fact_token_sets: list[frozenset[str]],
    query_tokens: frozenset[str],
) -> dict[int, tuple[int, int, RawTurn]]:
    """For each recalled fact, find the source turn with the highest fact-token
    overlap (tiebreaker: query-token overlap). Returns {fact_index: (overlap,
    q_overlap, turn)}."""
    best: dict[int, tuple[int, int, RawTurn]] = {}
    for turn in turns:
        text_tokens = _tokens(turn.content)
        if not text_tokens:
            continue
        q_overlap = len(query_tokens & text_tokens)
        for fi, ftokens in enumerate(fact_token_sets):
            overlap = len(ftokens & text_tokens)
            if overlap == 0:
                continue
            prev = best.get(fi)
            key = (overlap, q_overlap)
            if prev is None or key > (prev[0], prev[1]):
                best[fi] = (overlap, q_overlap, turn)
    return best


# ---------------------------------------------------------------------
# 2. Reflector
# ---------------------------------------------------------------------

_REEXTRACTION_SYSTEM_PROMPT = (
    "You re-extract facts from raw conversation turns to answer a specific "
    "question that earlier extraction failed to answer well. Read every source "
    "turn and extract EVERY fact that directly answers the question. If the "
    "question asks for activities, kinds, or a list, output ONE fact PER "
    "distinct item -- do not stop at the first; scan all turns and gather each "
    "one. Use a single entity as the subject (the main actor named in the "
    "source turn). Put co-participants in the object, not the subject: if a "
    "turn says the subject did something with someone, write the main actor as "
    "subject and the other participant inside the object, e.g. subject=James "
    "and object=the pub with John. Never use a compound subject such as John "
    "and James, because compound subjects break entity lookup and the fact "
    "becomes unfindable. Never invent a fact, name, or date not present in the "
    "source turns. When the question asks for a date or time, extract absolute "
    "dates from the source, not relative expressions like yesterday or today; "
    "if a turn only has a relative date, skip it rather than resolve it, so "
    "the answer stays the sourced absolute date. "
    'Reply with a JSON object {"facts": '
    '[{"subject","predicate","object","event_time"}], "events": []} and '
    "nothing else."
)


@runtime_checkable
class Reflector(Protocol):
    """Re-extract query-answering facts from source turns."""

    async def reflect(
        self,
        query: str,
        source_turns: Sequence[RawTurn],
        *,
        llm: LLMAdapter,
    ) -> list[ReflectedFact]: ...


class QueryFocusedReflector:
    """LLM query-focused re-extraction over the source turns (async)."""

    def __init__(self, *, max_tokens: int = 512) -> None:
        self._max_tokens = max_tokens

    async def reflect(
        self,
        query: str,
        source_turns: Sequence[RawTurn],
        *,
        llm: LLMAdapter,
    ) -> list[ReflectedFact]:
        if not source_turns:
            return []
        body = "\n\n".join(
            f"[turn {i}] speaker={t.role}: {t.content}" for i, t in enumerate(source_turns)
        )
        prompt = (
            f"{_REEXTRACTION_SYSTEM_PROMPT}\n\nQuestion: {query}\n\nSource turns:\n{body}\n\nJSON:"
        )
        try:
            messages: list[LLMMessage | dict[str, Any]] = [{"role": "user", "content": prompt}]
            response = await llm.chat(messages, temperature=0.0, max_tokens=self._max_tokens)
            raw = getattr(response, "content", "") or ""
        except Exception:
            return []
        return _parse_reflected_facts(raw)


def _parse_reflected_facts(raw: str) -> list[ReflectedFact]:
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    facts: list[ReflectedFact] = []
    for item in data.get("facts", []) or []:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip().replace(" ", "_")
        obj = str(item.get("object", "")).strip()
        if not subject or not predicate or not obj:
            continue
        event_time = item.get("event_time")
        facts.append(
            ReflectedFact(
                subject=subject,
                predicate=predicate,
                object=obj,
                event_time=str(event_time) if event_time else None,
            )
        )
    return facts


# ---------------------------------------------------------------------
# 3. Mutator (grounding = the mutation mechanism: raw -> anchored or reject)
# ---------------------------------------------------------------------


@runtime_checkable
class Mutator(Protocol):
    """Mutate a raw ReflectedFact into an anchored AtomicFact, or reject it."""

    def mutate(self, fact: ReflectedFact, source_turns: Sequence[RawTurn]) -> AtomicFact | None: ...


class SourceGroundedMutator:
    """Ground a fact by matching its object tokens to a source turn.

    Grounding (the mutation mechanism): a raw ReflectedFact is mutated into
    an anchored AtomicFact when at least one source turn mentions half of the
    fact's object tokens -- the fact is "grounded" (supported by source text,
    not hallucination). The anchor is the supporting turn's source_anchor.
    If no turn supports the fact, the mutation fails (reject).
    """

    def mutate(self, fact: ReflectedFact, source_turns: Sequence[RawTurn]) -> AtomicFact | None:
        object_tokens = _tokens(fact.object)
        if not object_tokens:
            return None
        threshold = max(1, len(object_tokens) // 2)
        best_turn: RawTurn | None = None
        best_overlap = 0
        for turn in source_turns:
            text_tokens = _tokens(turn.content)
            overlap = len(object_tokens & text_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
        if best_turn is None or best_overlap < threshold:
            return None
        meta = best_turn.metadata if isinstance(best_turn.metadata, dict) else {}
        raw_anchor = str(meta.get("source_anchor", "")).strip()
        anchor = raw_anchor or best_turn.turn_id
        return AtomicFact(
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            certainty=Certainty.CERTAIN,
            source_anchor=anchor,
            event_time=fact.event_time,
        )


# ---------------------------------------------------------------------
# 4. Evaluator
# ---------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Promote a grounded fact iff it is retrievable for the failing query."""

    async def evaluate(
        self,
        query: str,
        fact: AtomicFact,
        source_turn: RawTurn,
        *,
        recall: RecallProbe,
        namespace: str,
        top_k: int = 10,
    ) -> MemoryRecord | None: ...


class RetrievabilityEvaluator:
    """Persist-test-retract: promote, backfill, recall, keep iff it surfaces.

    The candidate is written through the append-only promoter, its embedding
    is backfilled, then the failing query is re-run through real recall. If
    the candidate's object tokens appear in the top-k, it is genuinely
    retrievable and kept; otherwise it is retracted (valid_to set).
    """

    def __init__(
        self,
        promoter: FactPromoter,
        store: Any,
        backfill: Any | None = None,
        entity_state: Any | None = None,
    ) -> None:
        self._promoter = promoter
        self._store = store
        self._backfill = backfill
        self._entity_state = entity_state

    async def evaluate(
        self,
        query: str,
        fact: AtomicFact,
        source_turn: RawTurn,
        *,
        recall: RecallProbe,
        namespace: str,
        top_k: int = 10,
    ) -> MemoryRecord | None:
        record = self._promoter.promote(source_turn, fact)
        if record is None:
            return None
        # The promoter writes the MemoryRecord + FTS + vector, but not the
        # entity_state L2 index. The hot extraction path upserts entity_state
        # separately via the extractor worker; reflection facts skip that
        # worker, so upsert entity_state here or the EntityStateRetriever --
        # the main recall path -- cannot find the new fact and
        # self-retrievability is a false negative (the just-promoted fact is
        # never recalled, so it is retracted and reflection has no effect).
        if self._entity_state is not None:
            with contextlib.suppress(Exception):
                self._entity_state.upsert(
                    namespace,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    certainty=fact.certainty,
                    source_unit_id=fact.source_anchor or None,
                )
        if self._backfill is not None:
            await self._backfill.process_once()
        new_object = str(fact.object)
        candidates = await recall.recall(query, namespace=namespace, top_k=top_k)
        # Judge whether the just-promoted fact itself surfaces in top-k, not
        # whether some other candidate shares object tokens. Token overlap
        # was a false positive: a new fact about baseball with James matched
        # the old fact about John attending baseball on the baseball token,
        # kept a fact that never entered top-k, and the after-answer then hit
        # the LLM cache unchanged so reflection had no effect. Match the exact
        # object string, including compound members (a consolidated compound
        # carries member objects in signals under compound_members).
        in_top_k = any(
            str(c.fact.object) == new_object
            or new_object in (getattr(c, "signals", None) or {}).get("compound_members", [])
            for c in candidates
        )
        if not in_top_k:
            ts = time.time()
            self._store.put_record(record.model_copy(update={"valid_to": ts}))
            # Retract the entity_state row too, or it stays active and the
            # ghost fact keeps surfacing in recall. invalidate closes the
            # active row for this subject+predicate; reflection facts usually
            # have a distinct predicate from existing facts so this does not
            # touch unrelated rows, but a same-predicate retraction would close
            # a prior active row -- a precise state_id close is the future fix.
            if self._entity_state is not None:
                with contextlib.suppress(Exception):
                    self._entity_state.invalidate(
                        namespace, fact.subject, fact.predicate, valid_to=ts
                    )
            return None
        return record


# ---------------------------------------------------------------------
# Report + orchestrator (async)
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReflectionReport:
    queries_reflected: int = 0
    facts_extracted: int = 0
    facts_grounded: int = 0
    facts_kept: int = 0
    facts_retracted: int = 0
    duration_ms: float = 0.0
    kept_records: tuple[MemoryRecord, ...] = ()


@runtime_checkable
class Reflection(Protocol):
    """Reflect on failing queries to repair semantic recall gaps."""

    async def reflect(
        self,
        failing_queries: Sequence[str],
        *,
        namespace: str,
    ) -> ReflectionReport: ...


class MemoryReflector:
    """Failure-anchored source re-extraction orchestrator (async).

    For each failing query: sample source turns (semantic recall + per-fact
    best-turn match), re-extract query-answering facts from those turns (LLM),
    mutate each into an anchored AtomicFact (grounding), and promote only
    facts that are actually retrievable (self-retrievability evaluator).
    All I/O (LLM, recall, backfill) runs in one async event loop.
    """

    def __init__(
        self,
        *,
        sampler: Sampler,
        reflector: Reflector,
        mutator: Mutator,
        evaluator: Evaluator,
        recall: RecallProbe,
        source_reader: SourceReader,
        llm: LLMAdapter,
    ) -> None:
        self._sampler = sampler
        self._reflector = reflector
        self._mutator = mutator
        self._evaluator = evaluator
        self._recall = recall
        self._source_reader = source_reader
        self._llm = llm

    async def reflect(
        self,
        failing_queries: Sequence[str],
        *,
        namespace: str,
    ) -> ReflectionReport:
        started = time.perf_counter()
        extracted = grounded = kept = retracted = 0
        kept_records: list[MemoryRecord] = []
        for query in failing_queries:
            source_turns = await self._sampler.sample(
                query,
                recall=self._recall,
                source_reader=self._source_reader,
                namespace=namespace,
            )
            if not source_turns:
                continue
            reflected = await self._reflector.reflect(query, source_turns, llm=self._llm)
            extracted += len(reflected)
            for rf in reflected:
                fact = self._mutator.mutate(rf, source_turns)
                if fact is None:
                    continue
                grounded += 1
                supporting = _supporting_turn(rf, source_turns) or source_turns[0]
                kept_record = await self._evaluator.evaluate(
                    query,
                    fact,
                    supporting,
                    recall=self._recall,
                    namespace=namespace,
                )
                if kept_record is not None:
                    kept += 1
                    kept_records.append(kept_record)
                else:
                    retracted += 1
        return ReflectionReport(
            queries_reflected=sum(1 for q in failing_queries),
            facts_extracted=extracted,
            facts_grounded=grounded,
            facts_kept=kept,
            facts_retracted=retracted,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            kept_records=tuple(kept_records),
        )


def _supporting_turn(fact: ReflectedFact, turns: Sequence[RawTurn]) -> RawTurn | None:
    object_tokens = _tokens(fact.object)
    if not object_tokens:
        return None
    threshold = max(1, len(object_tokens) // 2)
    best: RawTurn | None = None
    best_overlap = 0
    for turn in turns:
        overlap = len(object_tokens & _tokens(turn.content))
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn
    if best is None or best_overlap < threshold:
        return None
    return best


__all__ = [
    "Evaluator",
    "MemoryReflector",
    "Mutator",
    "QueryFocusedReflector",
    "RecallAnchoredSampler",
    "RecallProbe",
    "ReflectedFact",
    "Reflection",
    "ReflectionReport",
    "Reflector",
    "RetrievabilityEvaluator",
    "Sampler",
    "SourceGroundedMutator",
    "SourceReader",
]
