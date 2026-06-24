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

Design references (positioned against memory substrates Mem0/Zep/Graphiti,
not task-agent self-evolution systems like ExpeL/Reflexion):
- Failure-anchored source re-extraction: no major memory substrate re-extracts
  from source turns for a specific failing query.
- Causal self-retrievability judge: inject the candidate, run real recall,
  check whether it surfaces in top-k -- directly measuring the property that
  matters, instead of an LLM opinion or a lexical-coverage tautology.
- Grounding gate: every reflected fact must be supported by a source turn.

The reflector depends on four narrow, read-only / append-only protocols
(RecallProbe, SourceReader, FactPromoter, LLMAdapter) so it
has no circular coupling with the recall path: it reads recall results and
source turns, and persists candidates through the same append-only write path
every other writer uses.
"""

from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------
# Async bridging (the reflection pipeline is synchronous; recall and the
# LLM adapter are async)
# ---------------------------------------------------------------------


def _run_coro(coro: Coroutine[Any, Any, Any]) -> Any:
    """Execute a coroutine whether or not an event loop is already running."""
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


def _complete_sync(adapter: LLMAdapter, prompt: str, *, max_tokens: int) -> str:
    """Run an adapter chat completion from synchronous code.

    Messages are passed as plain dicts (not LLMMessage) so they are JSON-
    serializable end to end: the adapter contract accepts LLMMessage | dict,
    and some adapter wrappers hash the message list via json.dumps before
    the adapter normalizes it, which would raise on a pydantic LLMMessage.
    """
    messages: list[LLMMessage | dict[str, Any]] = [{"role": "user", "content": prompt}]
    coro = adapter.chat(messages, temperature=0.0, max_tokens=max_tokens)
    response = _run_coro(coro)
    return getattr(response, "content", "") or ""


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
        # pronouns / possessives -- the LLM paraphrases speaker voice
        # (his/her/my) which must not break source grounding.
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
    """Lowercase content tokens of text with stopwords removed."""
    return frozenset(tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS)


# ---------------------------------------------------------------------
# Read-only seams
# ---------------------------------------------------------------------


@runtime_checkable
class RecallProbe(Protocol):
    """Synchronous read-only recall for the failing query."""

    def recall(self, query: str, *, namespace: str, top_k: int = 10) -> list[RecallCandidate]: ...


class _SyncRecallProbe:
    """Bridge the async RecallOrchestrator.recall into a sync call.

    Recall is a pure read (the retriever contract forbids mutation), so this
    is safe to call from the synchronous reflection pipeline.
    """

    def __init__(self, orchestrator: RecallOrchestrator) -> None:
        self._orchestrator = orchestrator

    def recall(self, query: str, *, namespace: str, top_k: int = 10) -> list[RecallCandidate]:
        from houyi.adapters.memory.recall.types import RecallQuery, RetrieverContext

        recall_query = RecallQuery(text=query, top_k=top_k, namespace=namespace)
        result = _run_coro(self._orchestrator.recall(recall_query, RetrieverContext()))
        return list(result.candidates)


@runtime_checkable
class SourceReader(Protocol):
    """Read-only access to the raw source turn log."""

    def list_turns(self, namespace: str) -> list[RawTurn]: ...


class _BackendSourceReader:
    """Wrap a SQLite backend's raw-turn log behind the SourceReader seam."""

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
    """A query-answering triple before it is anchored to a source turn."""

    subject: str
    predicate: str
    object: str
    event_time: str | None = None


# ---------------------------------------------------------------------
# 1. SourceTurnSampler
# ---------------------------------------------------------------------


@runtime_checkable
class SourceTurnSampler(Protocol):
    """Select the source turns a failing query is about."""

    def sample(
        self,
        query: str,
        *,
        recall: RecallProbe,
        source_reader: SourceReader,
        namespace: str,
        top_k: int = 30,
    ) -> list[RawTurn]: ...


class RecallAnchoredSourceSampler:
    """Find source turns via semantic recall, then entity-match to turns.

    Recall (over already-embedded extracted facts) finds the facts semantically
    relevant to the failing query. Each recalled fact's object tokens are then
    matched against raw turn text to locate the source turns that produced
    those facts -- the source turns carry the full semantics the extractor
    flattened. This reuses the existing recall index (no new FTS over raw
    turns) and is robust to the source_anchor format differing from turn_id.
    """

    def __init__(self, *, max_turns: int = 10) -> None:
        self._max_turns = max_turns

    def sample(
        self,
        query: str,
        *,
        recall: RecallProbe,
        source_reader: SourceReader,
        namespace: str,
        top_k: int = 30,
    ) -> list[RawTurn]:
        candidates = recall.recall(query, namespace=namespace, top_k=top_k)
        if not candidates:
            return []
        fact_token_sets = [ts for ts in (_tokens(str(c.fact.object)) for c in candidates) if ts]
        if not fact_token_sets:
            return []
        turns = source_reader.list_turns(namespace)
        # For each turn, the set of recalled-fact indices it touches (any token
        # overlap). A turn "covers" a fact if it shares at least one content
        # token with that fact's object.
        turn_covers: list[tuple[RawTurn, frozenset[int]]] = []
        for turn in turns:
            text_tokens = _tokens(turn.content)
            if not text_tokens:
                continue
            covers = frozenset(i for i, ts in enumerate(fact_token_sets) if (ts & text_tokens))
            if covers:
                turn_covers.append((turn, covers))
        # Greedy max-coverage: pick the turn that covers the most still-uncovered
        # recalled facts, repeat. This surfaces the source turn for each distinct
        # underlying fact (e.g. the wine-tasting turn) even when its total token
        # overlap is low and would lose to busier turns under a sum-of-overlap
        # score.
        uncovered: set[int] = set(range(len(fact_token_sets)))
        selected: list[RawTurn] = []
        pool = list(turn_covers)
        while len(selected) < self._max_turns and uncovered and pool:
            best_idx = max(
                range(len(pool)),
                key=lambda i: len(pool[i][1] & uncovered),
            )
            best_turn, best_covers = pool[best_idx]
            gain = best_covers & uncovered
            if not gain:
                break
            selected.append(best_turn)
            uncovered -= gain
            pool.pop(best_idx)
        return selected


# ---------------------------------------------------------------------
# 2. QueryFocusedReExtractor
# ---------------------------------------------------------------------


_REEXTRACTION_SYSTEM_PROMPT = (
    "You re-extract facts from raw conversation turns to answer a specific "
    "question that earlier extraction failed to answer well. Read every source "
    "turn and extract EVERY fact that directly answers the question. If the "
    "question asks for activities, kinds, or a list, output ONE fact PER "
    "distinct item -- do not stop at the first; scan all turns and gather each "
    "one. Preserve participant relationships verbatim from the source (if a "
    "turn says the subject did something WITH someone, keep that "
    "co-participation in the fact). Never invent a fact, name, or date not "
    'present in the source turns. Reply with a JSON object {"facts": '
    '[{"subject","predicate","object","event_time"}], "events": []} and '
    "nothing else."
)


@runtime_checkable
class QueryFocusedReExtractor(Protocol):
    """Re-extract query-answering facts from source turns."""

    def reflect(
        self,
        query: str,
        source_turns: Sequence[RawTurn],
        *,
        llm: LLMAdapter,
    ) -> list[ReflectedFact]: ...


class LLMReExtractor:
    """LLM query-focused re-extraction over the source turns.

    Produces raw ReflectedFact triples (no source anchor yet); the
    grounding verifier anchors each to the source turn that supports it. Any
    LLM error or malformed JSON degrades to an empty list so the run never
    aborts on a transient model failure.
    """

    def __init__(self, *, max_tokens: int = 512) -> None:
        self._max_tokens = max_tokens

    def reflect(
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
            raw = _complete_sync(llm, prompt, max_tokens=self._max_tokens)
        except Exception:
            return []
        return _parse_reflected_facts(raw)


def _parse_reflected_facts(raw: str) -> list[ReflectedFact]:
    """Parse the LLM JSON response into ReflectedFact triples."""
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
# 3. GroundingVerifier
# ---------------------------------------------------------------------


@runtime_checkable
class GroundingVerifier(Protocol):
    """Anchor a reflected fact to a supporting source turn, or reject it."""

    def verify(self, fact: ReflectedFact, source_turns: Sequence[RawTurn]) -> AtomicFact | None: ...


class TokenOverlapGroundingVerifier:
    """Ground a fact by matching its object tokens to a source turn.

    A fact is grounded (and anchored to the best-supporting source turn) when
    at least one source turn mentions at least half of the fact's object
    content tokens (pronouns are already stripped, so speaker-voice paraphrase
    like his/her/my does not break grounding). Half-coverage (rather than all)
    tolerates the LLM's wording variants while still rejecting fabrication --
    a hallucinated object whose tokens appear in no source turn scores 0. The
    anchor is the supporting turn's source_anchor metadata (falling back to
    turn_id), so the promoted fact stays traceable to its source.
    """

    def verify(self, fact: ReflectedFact, source_turns: Sequence[RawTurn]) -> AtomicFact | None:
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
# 4. RetrievabilityJudge
# ---------------------------------------------------------------------


@runtime_checkable
class RetrievabilityJudge(Protocol):
    """Promote a grounded fact iff it is retrievable for the failing query."""

    def judge(
        self,
        query: str,
        fact: AtomicFact,
        source_turn: RawTurn,
        *,
        recall: RecallProbe,
        namespace: str,
        top_k: int = 10,
    ) -> MemoryRecord | None: ...


class SelfRetrievabilityJudge:
    """Persist-test-retract: promote, recall, keep iff the candidate surfaces.

    The candidate is written through the append-only promoter (the same path
    every writer uses), then the failing query is re-run through real recall.
    If the candidate's (subject, object) appears in the top-k, it is genuinely
    retrievable and is kept (the persisted record is returned); otherwise it is
    retracted by re-putting the record with valid_to set (append-only bi-
    temporal retraction, no delete) and None is returned. This breaks the old
    lexical-coverage tautology: an observation that merely echoes query tokens
    no longer passes by construction -- it must actually surface in retrieval.
    """

    def __init__(self, promoter: FactPromoter, store: Any) -> None:
        self._promoter = promoter
        self._store = store

    def judge(
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
        candidates = recall.recall(query, namespace=namespace, top_k=top_k)
        subject = fact.subject.lower()
        obj_tokens = _tokens(fact.object)
        surfaced = any(
            c.fact.subject.lower() == subject and (obj_tokens & _tokens(str(c.fact.object)))
            for c in candidates
        )
        if not surfaced:
            # Retract: re-put the record with valid_to so recall stops
            # surfacing it. Append-only; no row is deleted.
            self._store.put_record(record.model_copy(update={"valid_to": time.time()}))
            return None
        return record


# ---------------------------------------------------------------------
# Report + orchestrator
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReflectionReport:
    """Result of one reflection run."""

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

    def reflect(
        self,
        failing_queries: Sequence[str],
        *,
        namespace: str,
    ) -> ReflectionReport: ...


class MemoryReflector:
    """Failure-anchored source re-extraction orchestrator.

    For each failing query: sample source turns (semantic recall + entity
    match), re-extract query-answering facts from those turns (LLM), ground
    each against its source (no hallucination), and promote only facts that
    are actually retrievable for the failing query (self-retrievability judge).
    """

    def __init__(
        self,
        *,
        sampler: SourceTurnSampler,
        reextractor: QueryFocusedReExtractor,
        verifier: GroundingVerifier,
        judge: RetrievabilityJudge,
        recall: RecallProbe,
        source_reader: SourceReader,
        llm: LLMAdapter,
    ) -> None:
        self._sampler = sampler
        self._reextractor = reextractor
        self._verifier = verifier
        self._judge = judge
        self._recall = recall
        self._source_reader = source_reader
        self._llm = llm

    def reflect(
        self,
        failing_queries: Sequence[str],
        *,
        namespace: str,
    ) -> ReflectionReport:
        started = time.perf_counter()
        extracted = grounded = kept = retracted = 0
        kept_records: list[MemoryRecord] = []
        for query in failing_queries:
            source_turns = self._sampler.sample(
                query,
                recall=self._recall,
                source_reader=self._source_reader,
                namespace=namespace,
            )
            if not source_turns:
                continue
            reflected = self._reextractor.reflect(query, source_turns, llm=self._llm)
            extracted += len(reflected)
            for rf in reflected:
                fact = self._verifier.verify(rf, source_turns)
                if fact is None:
                    continue
                grounded += 1
                # Anchor the judge's source turn to the one the verifier
                # grounded against: find the supporting turn again.
                supporting = _supporting_turn(rf, source_turns) or source_turns[0]
                kept_record = self._judge.judge(
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
    """Return the turn that best supports the fact (half-coverage, mirroring
    the grounding verifier)."""
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
    "GroundingVerifier",
    "LLMReExtractor",
    "MemoryReflector",
    "QueryFocusedReExtractor",
    "RecallAnchoredSourceSampler",
    "RecallProbe",
    "ReflectedFact",
    "Reflection",
    "ReflectionReport",
    "RetrievabilityJudge",
    "SelfRetrievabilityJudge",
    "SourceReader",
    "SourceTurnSampler",
    "TokenOverlapGroundingVerifier",
]
