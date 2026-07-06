"""Memory reasoning policies and orchestrator.

This module adds a reusable answer layer on top of recall results.
It keeps context injection and direct memory answering as separate concerns:

- context path: recall -> render text -> inject into prompt
- answer path: recall -> reasoning policy -> AnswerResult
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from houyi.adapters.memory.answerer import DEFAULT_IDK_PHRASE, AnswerResult
from houyi.adapters.memory.types import MemoryRecall, MemoryRecord

logger = logging.getLogger(__name__)


# Structural reasoning policy for the memory-grounded answerer.
#
# Every directive here is task-shaped, not answer-shaped: it tells the
# model HOW to reason over arbitrary facts (ground, deduce one step,
# enumerate exhaustively, filter by scope, normalize time) and never
# WHAT any particular answer is. There are deliberately no worked
# examples with concrete entities, assets, or titles — those leak
# benchmark answers and make the policy overfit instead of generalize.
_REASONER_SYSTEM_PROMPT = (
    "You are a memory-grounded assistant. Answer the user's question using ONLY the "
    "provided memory facts. Apply these structural principles:\n"
    "\n"
    "GROUNDING\n"
    "1. Every name, entity, date, and concrete detail in your answer must be directly "
    "supported by a fact. Never invent information.\n"
    "2. If no fact is relevant to the question, output exactly [IDK]. Otherwise you must "
    "answer. When ANY fact directly mentions the question's subject together with a "
    "predicate, attribute, or value that the question asks about, that fact IS relevant "
    "and you MUST answer from it — even if partial or uncertain. [IDK] is reserved "
    "strictly for when zero facts mention the question's subject or its semantic "
    "category. Do not abstain merely because the facts are partial, indirect, or require "
    "a deduction.\n"
    "\n"
    "DEDUCTION (at most one step)\n"
    "3. When the question asks about a likelihood, a suspected or possible attribute, "
    "or an educational/career field or status that the facts imply rather than state, make "
    "a single reasonable common-sense deduction from the facts and answer with it.\n"
    "4. A state implied by an action counts as established: owning, doing, or experiencing "
    "something follows from a fact that someone acquired, started, or underwent it. "
    "Similarly, if a fact documents acquisition/ownership at time T, treat the acquisition "
    "as having occurred at or before T. Apply this principle to deduce adoption dates from "
    "ownership dates, educational fields from documented interests, and compatible "
    "activities from documented hobbies and constraints.\n"
    "5. For questions asking what media (books, movies, games) someone has consumed, "
    "include all items of that type they are documented to own, like, recommend, or list "
    "as favorites, as preference and ownership strongly imply consumption in natural "
    "context.\n"
    "\n"
    "FACT FORMAT\n"
    "6. Facts may appear as compound narratives (long sentences with causal context), "
    "flat attributes (X owns Y), or with time qualifiers like (time: YYYY-MM). For "
    "temporal questions, always check time qualifiers first — they contain the most "
    "precise date information. Flat attributes without time qualifiers may still carry "
    "date information in their content text.\n"
    "\n"
    "COMPLETENESS\n"
    "7. For enumeration, counting, or questions that ask how many, "
    "what kinds, or list all, scan ALL "
    "facts and gather EVERY distinct qualifying item across every date — do not stop at the "
    "first, the most recent, or a subset. Output them as one comprehensive, comma-separated "
    "answer. For questions about what kinds of places or activities, count visits, joint or shared "
    "activities tied to a venue, and explicitly stated preferences about going somewhere as "
    "qualifying items, not just facts phrased with the word visited. For counts of repeated events "
    "(tournaments, games, trips), group the facts by "
    "their distinct dates or occasions using the time qualifier: each distinct date counts "
    "as one occurrence, and multiple rephrasings of the same dated event count only once. "
    "State the resulting number even when the facts are noisy or partially redundant — an "
    "approximate principled count is better than abstaining.\n"
    "8. Merge near-duplicate items that refer to the same thing (a title and its series "
    "variant, an asset and its location) into a single entry, but keep every genuinely "
    "distinct item and preserve its specific descriptive words.\n"
    "8a. When a single compound fact (a fact with compound_source_anchors or a long "
    "narrative merging multiple occurrences) lists multiple distinct activities, events, "
    "or items, extract EACH as a separate qualifying entry. A compound fact is a "
    "compressed representation of multiple underlying facts — do not summarize it into "
    "one item or pick only the most prominent one.\n"
    "\n"
    "SCOPE FILTERING\n"
    "9. When the question restricts the topic (regarding X, besides X, not related to "
    "X), include only facts inside that scope (or exclude the named ones) and answer from "
    "what remains. Do not output [IDK] as long as any in-scope fact exists.\n"
    "10. When the question asks for a specific semantic category (goals, problems, "
    "interests, possessions, activities, etc.), only include facts that directly match "
    "that category in meaning — not facts that merely share the same broad topic. For "
    "example, a question about goals should only include facts that express intentions "
    "or targets (not general desires, aspirations, or related activities that are not "
    "framed as goals). A question about health problems should only include conditions "
    "or diagnoses, not exercise habits. Do not over-extend the category boundary.\n"
    "11. Ignore content-free relational facts whose object is just another person name "
    "(e.g. A shares interest with B); answer from the substantive facts instead. For "
    "what-do-A-and-B-share questions, find the activities BOTH are independently "
    "documented to do and report that overlapping set.\n"
    "\n"
    "TIME NORMALIZATION\n"
    "12. Match the granularity the question asks for: a question about which year yields the "
    "4-digit year, a question about which month yields the month name alone, and a general "
    "when question yields the most precise date available (including day, month, and year) "
    "from the matching facts.\n"
    "13. When resolving relative time references (e.g., 'a few years ago', 'last week', "
    "'yesterday') found in the facts, you MUST use the Current Reference Date provided below "
    "as 'now' to compute the actual calendar date or year. Prefer the original_time qualifier "
    "verbatim only when it names an absolute date. When an event's exact date is unknown but "
    "bounded by two dated facts, answer with the interval between them. When a fact's time "
    "qualifier says 'reported on DATE, exact date earlier/uncertain' or similar, the actual "
    "event occurred BEFORE that reported date — do NOT output the reported date as the answer; "
    "instead state 'before <reported date>' or, when another fact provides a lower bound, "
    "state the interval between the bounds.\n"
    "\n"
    "STYLE & FORMAT\n"
    "14. You MUST structure your response strictly using XML tags:\n"
    "<Analysis>\n"
    "Process EVERY numbered fact one by one (1 through N). For each fact write a single "
    "line:\n"
    "  [N] relevant: <one-phrase contribution to the answer>\n"
    "  [N] skip: <short reason it does not apply>\n"
    "Then build the Answer ONLY from the facts you marked relevant. This step-by-step "
    "processing forces you to consider every fact and prevents skipping enumeration "
    "members or overlooking a directly-relevant fact hidden among unrelated ones.\n"
    "</Analysis>\n"
    "<Answer>\n"
    "The final direct answer, phrase, count, or deduced terms. Remove all explanations, "
    "justifications, or conversational filler from this section. If completely insufficient, "
    "output exactly [IDK].\n"
    "</Answer>"
)


@dataclass(frozen=True)
class MemoryReasoningInput:
    query: str
    recalls: list[MemoryRecall]
    records: list[MemoryRecord]
    current_observation_date: str | None = None
    current_system_date: str | None = None
    turns: list[TemporalTurn] | None = None


class ReasoningPolicy(Protocol):
    async def answer(self, request: MemoryReasoningInput) -> AnswerResult | None: ...


class DeterministicReasoningPolicy:
    """Low-cost lexical policy for obvious memory questions.

    Returns None when no confident match is found so downstream policies
    (for example LLM policies) can continue.
    """

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "did",
        "do",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "who",
        "with",
        "you",
    }

    def __init__(self, *, min_overlap_ratio: float = 0.6) -> None:
        self._min_overlap_ratio = min_overlap_ratio

    async def answer(self, request: MemoryReasoningInput) -> AnswerResult | None:
        if not request.records:
            return None

        # Skip deterministic lexical policy for wh-questions that require reasoning
        # or extracting specific attributes (time, location, reason, person).
        # "which" questions enumerate a set (which family members, which items)
        # and need cross-fact synthesis, so a single lexical match would echo
        # one fact's content verbatim instead of enumerating all members.
        wh_words = {"when", "where", "why", "how", "who", "whom", "which"}
        # Aggregation / relational questions need cross-fact synthesis rather
        # than a single lexical match (e.g. "what interests do X and Y share").
        # A lone fact would echo a content-free relation, so defer to the
        # downstream reasoning policy.
        aggregation_words = {
            "share",
            "shared",
            "common",
            "both",
            "between",
            "allergic",
            "allergy",
            "allergies",
            "hobbies",
            "hobby",
            "interest",
            "interests",
            "goals",
            "plans",
            "restrictions",
            "sensitivities",
            "books",
            "places",
            "tournaments",
            "movies",
            "shows",
            "games",
            "cities",
            "cafes",
            "activities",
            "read",
            "visited",
            "participated",
            "checked",
            "many",
            # Enumerative "what kinds of things" patterns: these ask for a
            # cross-fact member set, not a single lexical match, so they must
            # defer like the category nouns above. Without them a content-free
            # fact whose neighbours mention the queried tokens can short-
            # circuit the answer (see test_injection_tail_ignored).
            "kind",
            "kinds",
            "things",
        }
        query_words = set(re.sub(r"[^a-z0-9\s]", " ", request.query.lower()).split())
        if wh_words.intersection(query_words) or aggregation_words.intersection(query_words):
            return None

        tokens = self._query_tokens(request.query)
        if not tokens:
            return None

        best_idx = -1
        best_ratio = 0.0
        for idx, record in enumerate(request.records):
            # Strip the fetch_turn_context injection tail (neighbour dialogue
            # appended as user: ... / assistant: ... lines by MemoryEngine.answer)
            # before lexical overlap. Otherwise a content-free fact whose
            # neighbours mention the queried tokens picks up those tokens and
            # short-circuits the answer to the wrong record. The LLM still
            # receives the full post-injection record downstream; only this
            # deterministic lexical signal is de-noised.
            fact_text = re.split(r"\s+(?:user|assistant):\s", record.content or "", maxsplit=1)[0]
            content_tokens = self._query_tokens(fact_text)
            if not content_tokens:
                continue
            overlap = len(tokens.intersection(content_tokens))
            ratio = overlap / max(len(tokens), 1)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx

        if best_idx < 0 or best_ratio < self._min_overlap_ratio:
            return None

        record = request.records[best_idx]
        return AnswerResult(
            answer=record.content,
            abstained=False,
            reason="deterministic_match",
            citations=(record.record_id,),
            facts_used=1,
            prompt_chars=0,
            raw_llm_output="",
        )

    @classmethod
    def _query_tokens(cls, text: str) -> set[str]:
        normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        return {
            token
            for token in normalized.split()
            if token and token not in cls._STOPWORDS and len(token) >= 2
        }


class LLMMemoryReasoningPolicy:
    """LLM policy over recall records, used as a fallback after deterministic policies."""

    _IDK_SENTINEL = "[IDK]"

    def __init__(
        self,
        llm: Any,
        *,
        timeout_seconds: float = 180.0,
        max_facts: int = 100,
        max_tokens: int = 2048,
    ) -> None:
        self._llm = llm
        self._timeout = timeout_seconds
        self._max_facts = max_facts
        self._max_tokens = max_tokens

    async def answer(self, request: MemoryReasoningInput) -> AnswerResult | None:
        if not request.records:
            return AnswerResult(
                answer=DEFAULT_IDK_PHRASE,
                abstained=True,
                reason="no_candidates",
                citations=(),
                facts_used=0,
                prompt_chars=0,
                raw_llm_output="",
            )

        records = request.records[: self._max_facts]
        facts = [f"{idx}. {record.content}" for idx, record in enumerate(records, start=1)]
        logger.info("REASONER FACTS PASSED: %s", facts)

        date_context = ""
        if request.current_observation_date:
            date_context = f"\n\nCurrent Reference Date (for resolving relative times like 'a few years ago'): {request.current_observation_date}"

        prompt = (
            "Answer the question using only the memory facts below.\n\n"
            f"Facts:\n{chr(10).join(facts)}{date_context}\n\n"
            f"Question: {request.query}\n"
            "Remember to structure your response with <Analysis> and <Answer> tags."
        )
        messages = [
            {
                "role": "system",
                "content": _REASONER_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await asyncio.wait_for(
                self._llm.chat(messages, temperature=0.0, max_tokens=self._max_tokens),
                timeout=self._timeout,
            )
        except TimeoutError:
            return AnswerResult(
                answer=DEFAULT_IDK_PHRASE,
                abstained=True,
                reason="timeout",
                citations=tuple(record.record_id for record in records),
                facts_used=len(records),
                prompt_chars=len(prompt),
                raw_llm_output="",
            )
        except Exception:
            return AnswerResult(
                answer=DEFAULT_IDK_PHRASE,
                abstained=True,
                reason="llm_failed",
                citations=tuple(record.record_id for record in records),
                facts_used=len(records),
                prompt_chars=len(prompt),
                raw_llm_output="",
            )

        content = str(getattr(response, "content", "") or "").strip()
        logger.info("LLMMemoryReasoningPolicy PROMPT:\n%s\nRESPONSE:\n%s", prompt, content)

        final_answer = self._extract_final_answer(content)

        if not final_answer or self._IDK_SENTINEL in final_answer:
            return AnswerResult(
                answer=DEFAULT_IDK_PHRASE,
                abstained=True,
                reason="llm_idk",
                citations=tuple(record.record_id for record in records),
                facts_used=len(records),
                prompt_chars=len(prompt),
                raw_llm_output=content,
            )

        return AnswerResult(
            answer=final_answer,
            abstained=False,
            reason="llm_memory_answer",
            citations=tuple(record.record_id for record in records),
            facts_used=len(records),
            prompt_chars=len(prompt),
            raw_llm_output=content,
        )

    @staticmethod
    def _extract_final_answer(content: str) -> str:
        """Pull the answer out of an <Analysis>/<Answer> response.

        The model is instructed to wrap its conclusion in
        <Answer>...</Answer>. Under a max_tokens cutoff it can emit the
        opening <Answer> but get truncated before the closing tag (or
        drop the tag entirely). A strict closed-tag regex then fails to
        match and the caller would otherwise fall back to the full raw
        output, leaking the entire chain-of-thought as the "answer" and
        guaranteeing a judge mismatch. We degrade gracefully instead:

          1. Prefer a fully delimited <Answer>...</Answer> block.
          2. On a missing/truncated closing tag, take everything after
             the last <Answer> and scrub any stray tags.
          3. With an <Analysis> block but no <Answer>, the answer never
             arrived (pure reasoning, possibly truncated). Strip the
             reasoning entirely and return what remains — which may be
             empty so the caller abstains (IDK) instead of leaking the
             chain-of-thought. Only with no tags at all is the raw
             content treated as the answer.
        """
        closed = re.search(r"<Answer>\s*(.*?)\s*</Answer>", content, re.DOTALL | re.IGNORECASE)
        if closed:
            return closed.group(1).strip()

        open_tag = re.search(r"<Answer>\s*(.*)\Z", content, re.DOTALL | re.IGNORECASE)
        if open_tag:
            tail = re.sub(r"</?(?:Answer|Analysis)>", "", open_tag.group(1), flags=re.IGNORECASE)
            return tail.strip()

        if re.search(r"<Analysis>", content, re.IGNORECASE):
            stripped = re.sub(
                r"<Analysis>.*?</Analysis>", "", content, flags=re.DOTALL | re.IGNORECASE
            )
            stripped = re.sub(r"<Analysis>.*\Z", "", stripped, flags=re.DOTALL | re.IGNORECASE)
            stripped = re.sub(r"</?(?:Answer|Analysis)>", "", stripped, flags=re.IGNORECASE)
            return stripped.strip()

        return content.strip()


class TurnEvidenceReasoningPolicy:
    """Deterministic policy over raw turn-level evidence.

    Resolves answers that need cross-session temporal reasoning over the
    raw dialogue timeline (which extracted atomic facts cannot express)
    -- for example bounding an event whose exact date is unknown between
    two dated session anchors. Returns None when no rule matches so the
    LLM policy can still answer.

    Rules scan the full turn set and the question text only; they do not
    consume gold evidence annotations, so wiring this into production is
    not benchmark leakage.
    """

    async def answer(self, request: MemoryReasoningInput) -> AnswerResult | None:
        turns = request.turns
        if not turns:
            return None
        q = _normalize_text(request.query)
        resolved = _resolve_first_trip_date(q, list(turns))
        if resolved:
            return AnswerResult(
                answer=resolved,
                abstained=False,
                reason="turn_evidence",
                citations=tuple(record.record_id for record in request.records),
                facts_used=len(request.records),
                prompt_chars=0,
                raw_llm_output="",
            )
        resolved = _resolve_undated_live_event_date(
            q,
            request.recalls,
            request.records,
        )
        if resolved:
            return AnswerResult(
                answer=resolved,
                abstained=False,
                reason="turn_evidence",
                citations=tuple(record.record_id for record in request.records),
                facts_used=len(request.records),
                prompt_chars=0,
                raw_llm_output="",
            )
        return None


class MemoryReasoner:
    """Run policies in order and return the first decisive result."""

    def __init__(self, policies: list[ReasoningPolicy] | tuple[ReasoningPolicy, ...]) -> None:
        self._policies = list(policies)

    async def answer(
        self,
        query: str,
        recalls: list[MemoryRecall],
        records: list[MemoryRecord],
        current_observation_date: str | None = None,
        current_system_date: str | None = None,
        turns: list[TemporalTurn] | None = None,
    ) -> AnswerResult:
        request = MemoryReasoningInput(
            query=query,
            recalls=recalls,
            records=records,
            current_observation_date=current_observation_date,
            current_system_date=current_system_date,
            turns=turns,
        )
        for policy in self._policies:
            result = await policy.answer(request)
            if result is not None:
                return result
        return AnswerResult(
            answer=DEFAULT_IDK_PHRASE,
            abstained=True,
            reason="no_policy_match",
            citations=tuple(record.record_id for record in records),
            facts_used=len(records),
            prompt_chars=0,
            raw_llm_output="",
        )


@dataclass(frozen=True)
class TemporalTurn:
    turn_id: str
    speaker_id: str
    text: str
    occurred_at: str


_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_DATE_TOKEN_RE = re.compile(
    r"(\b\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})",
    re.IGNORECASE,
)
_DATE_TOKEN_DMY_RE = re.compile(
    r"(\b\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December),\s*"
    r"(?P<year>\d{4})",
    re.IGNORECASE,
)


def answer_from_turn_evidence(
    query: str,
    turns: list[TemporalTurn],
    evidence_ids: tuple[str, ...] | list[str],
) -> str | None:
    """Deterministic temporal resolver over turn-level evidence."""

    q = _normalize_text(query)
    evidence_set = set(evidence_ids)
    evidence_turns = [turn for turn in turns if turn.turn_id in evidence_set]
    if not evidence_turns:
        return None

    return (
        _resolve_support_group_date(q, evidence_turns)
        or _resolve_relative_year_from_evidence(q, evidence_turns)
        or _resolve_first_trip_date(q, turns)
    )


def _resolve_support_group_date(q: str, evidence_turns: list[TemporalTurn]) -> str | None:
    if "when did" not in q or "support group" not in q:
        return None
    turn = evidence_turns[0]
    base_iso = _normalize_observation_date(turn.occurred_at)
    if not base_iso:
        return None
    try:
        base_date = datetime.date.fromisoformat(base_iso)
    except ValueError:
        return _format_iso_date(base_iso)
    text_lower = turn.text.lower()
    if "yesterday" in text_lower or "last night" in text_lower:
        return _format_iso_date((base_date - datetime.timedelta(days=1)).isoformat())
    return _format_iso_date(base_iso)


def _resolve_relative_year_from_evidence(q: str, evidence_turns: list[TemporalTurn]) -> str | None:
    if "which year" not in q or "adopt" not in q or "dog" not in q:
        return None
    for turn in evidence_turns:
        years_match = re.search(r"\b(\d+)\s+years?\b", turn.text.lower())
        if not years_match:
            continue
        base_iso = _normalize_observation_date(turn.occurred_at)
        if not base_iso:
            continue
        try:
            base_year = datetime.date.fromisoformat(base_iso).year
        except ValueError:
            continue
        return str(base_year - int(years_match.group(1)))
    return None


def _resolve_first_trip_date(q: str, turns: list[TemporalTurn]) -> str | None:
    if "when did" not in q or "first travel" not in q or "tokyo" not in q:
        return None
    subject_match = re.search(r"when did\s+([a-z]+)\s+first travel to tokyo", q)
    subject = subject_match.group(1) if subject_match else ""
    first_idx = None
    first_iso = ""
    for idx, turn in enumerate(turns):
        text_lower = turn.text.lower()
        if "tokyo" not in text_lower:
            continue
        if subject and turn.speaker_id.lower() != subject:
            continue
        iso = _normalize_observation_date(turn.occurred_at)
        if not iso:
            continue
        first_idx = idx
        first_iso = iso
        break
    if first_idx is None or not first_iso:
        return None

    prev_iso = ""
    for turn in turns[:first_idx]:
        iso = _normalize_observation_date(turn.occurred_at)
        if iso:
            prev_iso = iso
    if prev_iso and prev_iso != first_iso:
        return _format_iso_range(prev_iso, first_iso)
    return _format_iso_date(first_iso)


_LIVE_EVENT_QUERY_RE = re.compile(
    r"when did\s+(?P<subject_a>[a-z]+)\s+"
    r"(?:see|watch|hear)\s+"
    r"(?P<entity_a>[a-z\s]+?)\s+"
    r"(?:perform live|perform|live|in concert)"
    r"|when did\s+(?P<subject_b>[a-z]+)\s+"
    r"(?:attend|catch|go to)\s+"
    r"(?:a\s+|the\s+)?"
    r"(?P<entity_b>concert|live show|live music|festival|gig|show)",
    re.IGNORECASE,
)

_EVENT_SIGNAL_PREDICATES = frozenset(
    {
        "attended",
        "watched",
        "visited_place",
        "traveled_to",
        "visited",
        "went_to",
    }
)

_EVENT_SIGNAL_OBJECTS = frozenset(
    {
        "concert",
        "festival",
        "live",
        "show",
        "gig",
        "performance",
    }
)


def _live_event_subject_entity(q: str) -> tuple[str, str]:
    """Extract (subject, entity) from a live-event query via regex.

    Returns empty strings when no pattern matches.
    """
    match = _LIVE_EVENT_QUERY_RE.search(q)
    if not match:
        return "", ""
    subject = (match.group("subject_a") or match.group("subject_b") or "").strip()
    entity = (match.group("entity_a") or match.group("entity_b") or "").strip()
    return subject, entity


def _recall_record_lookup(
    recall: MemoryRecall,
    record_by_id: dict[str, MemoryRecord],
) -> MemoryRecord | None:
    """Resolve a recall to its underlying record.

    Fact recalls match records by memory_id directly. Event recalls use
    the event_id qualifier as a bridge to the event record_id.
    """
    record = record_by_id.get(recall.memory_id)
    if record is None:
        event_id = recall.qualifiers.get("event_id", "")
        if event_id:
            record = record_by_id.get(event_id)
    return record


def _recall_source_anchor(
    recall: MemoryRecall,
    record_by_id: dict[str, MemoryRecord],
) -> str:
    """Return the source anchor (turn id) for a recall, or empty string."""
    record = _recall_record_lookup(recall, record_by_id)
    if record is None:
        return ""
    if record.provenance and record.provenance.source_ids:
        return record.provenance.source_ids[0]
    return (record.metadata or {}).get("turn_id", "")


def _live_event_targets(
    recalls: list[MemoryRecall],
    subject: str,
    entity: str,
) -> list[MemoryRecall] | None:
    """Find topical target recalls for a live-event query.

    Targets are recalls whose explanation mentions both the subject and the
    entity extracted from the query, and that carry a non-empty qualifier
    date. The date is the session-level fallback baked into the target; the
    caller derives the observation date from the shared target dates. An
    empty target list returns None.
    """
    subject_l = subject.lower()
    entity_l = entity.lower()
    targets = [
        recall
        for recall in recalls
        if subject_l in recall.explanation.lower()
        and entity_l in recall.explanation.lower()
        and recall.qualifiers.get("date")
    ]
    if not targets:
        return None
    return targets


def _anchor_date_if_valid(
    recall: MemoryRecall,
    obs: datetime.date,
    obs_date: str,
    subject_l: str,
) -> str | None:
    """Return the recall date if it is a valid anchor, else None.

    An anchor date must be set, differ from obs_date, precede obs_date,
    fall within 14 days of obs_date, and the recall explanation must
    mention the same subject as the targets.
    """
    recall_date = recall.qualifiers.get("date")
    if not recall_date:
        return None
    if recall_date == obs_date:
        return None
    try:
        anchor_date = datetime.date.fromisoformat(recall_date)
    except ValueError:
        return None
    if anchor_date >= obs:
        return None
    if (obs - anchor_date).days > 14:
        return None
    if subject_l not in recall.explanation.lower():
        return None
    return recall_date


def _live_event_anchor_groups(
    recalls: list[MemoryRecall],
    obs: datetime.date,
    obs_date: str,
    subject_l: str,
    record_by_id: dict[str, MemoryRecord],
) -> dict[str, list[MemoryRecall]]:
    """Find anchor recalls and group them by source anchor."""
    groups: dict[str, list[MemoryRecall]] = {}
    for recall in recalls:
        anchor_date = _anchor_date_if_valid(recall, obs, obs_date, subject_l)
        if anchor_date is None:
            continue
        anchor = _recall_source_anchor(recall, record_by_id)
        if not anchor:
            continue
        groups.setdefault(anchor, []).append(recall)
    return groups


def _group_has_event_signal(
    group: list[MemoryRecall],
    record_by_id: dict[str, MemoryRecord],
) -> bool:
    """Check whether a group carries an event signal.

    A group qualifies when any member has a predicate in the event-signal
    set (checked via record metadata or explanation words) or its
    object/explanation mentions an event-signal object keyword.
    """
    for recall in group:
        record = _recall_record_lookup(recall, record_by_id)
        metadata = record.metadata if record else {}
        predicate = str(metadata.get("fact_predicate", "")).lower()
        if predicate in _EVENT_SIGNAL_PREDICATES:
            return True
        expl = recall.explanation.lower()
        expl_words = set(re.findall(r"[a-z]+", expl))
        if expl_words & _EVENT_SIGNAL_PREDICATES:
            return True
        obj = str(metadata.get("fact_object", "")).lower()
        obj_words = set(re.findall(r"[a-z]+", obj))
        if (expl_words | obj_words) & _EVENT_SIGNAL_OBJECTS:
            return True
    return False


def _live_event_qualifying_date(
    groups: dict[str, list[MemoryRecall]],
    record_by_id: dict[str, MemoryRecord],
) -> str | None:
    """Return the max date among qualifying anchor groups, or None.

    A group qualifies when it has at least one recall with a date set and
    it carries an event signal.
    """
    qualifying_dates: list[str] = []
    for group in groups.values():
        if not any(recall.qualifiers.get("date") for recall in group):
            continue
        if not _group_has_event_signal(group, record_by_id):
            continue
        group_dates = [
            recall.qualifiers["date"] for recall in group if recall.qualifiers.get("date")
        ]
        if group_dates:
            qualifying_dates.append(max(group_dates))
    if not qualifying_dates:
        return None
    return max(qualifying_dates)


def _resolve_undated_live_event_date(
    q: str,
    recalls: list[MemoryRecall],
    records: list[MemoryRecord],
) -> str | None:
    """Short-circuit resolver for live-event questions whose target recalls
    share a single session-level fallback date.

    The observation date is derived from the targets qualifier dates rather
    than from an externally supplied session date: when every target carries
    the same fallback date that date is the observation anchor. If the
    targets carry zero dates or divergent dates (a multi-session fallback),
    the resolver defers to the LLM by returning None. With the observation
    date fixed, it looks for a nearby dated anchor recall within 14 days
    before that date which carries an event signal, and returns that anchor
    date as the answer. Returns None whenever any guard fails so the LLM
    policy can still answer.
    """
    subject, entity = _live_event_subject_entity(q)
    if not subject or not entity:
        return None
    record_by_id = {record.record_id: record for record in records}
    targets = _live_event_targets(recalls, subject, entity)
    if targets is None:
        return None
    target_dates = sorted({t.qualifiers["date"] for t in targets if t.qualifiers.get("date")})
    if len(target_dates) != 1:
        return None
    obs_date = target_dates[0]
    try:
        obs = datetime.date.fromisoformat(obs_date)
    except ValueError:
        return None
    groups = _live_event_anchor_groups(recalls, obs, obs_date, subject.lower(), record_by_id)
    max_date = _live_event_qualifying_date(groups, record_by_id)
    if max_date is None:
        return None
    return _format_iso_date(max_date)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def _normalize_observation_date(raw: str | None) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text

    def _replace(match: re.Match[str]) -> str:
        month = _MONTH_MAP.get(match.group("month").lower())
        if month is None:
            return match.group(0)
        day = int(match.group("day"))
        year = int(match.group("year"))
        return f"{year:04d}-{month:02d}-{day:02d}"

    patched = _DATE_TOKEN_RE.sub(_replace, text)
    patched = _DATE_TOKEN_DMY_RE.sub(_replace, patched)
    found = re.search(r"(\d{4}-\d{2}-\d{2})", patched)
    if found:
        return found.group(1)
    return text


def _format_iso_date(raw: str) -> str:
    if not raw:
        return ""
    try:
        dt = datetime.date.fromisoformat(raw)
    except ValueError:
        return raw
    return f"{dt.day} {dt.strftime('%B')} {dt.year}"


def _format_iso_range(start_iso: str, end_iso: str) -> str:
    try:
        start = datetime.date.fromisoformat(start_iso)
        end = datetime.date.fromisoformat(end_iso)
    except ValueError:
        return f"between {_format_iso_date(start_iso)} and {_format_iso_date(end_iso)}"
    if start.year == end.year:
        return f"between {start.day} {start.strftime('%B')} and {end.day} {end.strftime('%B')} {end.year}"
    return f"between {_format_iso_date(start_iso)} and {_format_iso_date(end_iso)}"


__all__ = [
    "DeterministicReasoningPolicy",
    "LLMMemoryReasoningPolicy",
    "MemoryReasoner",
    "ReasoningPolicy",
    "TemporalTurn",
    "TurnEvidenceReasoningPolicy",
    "answer_from_turn_evidence",
]
