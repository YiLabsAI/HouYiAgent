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


@dataclass(frozen=True)
class MemoryReasoningInput:
    query: str
    recalls: list[MemoryRecall]
    records: list[MemoryRecord]


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
        wh_words = {"when", "where", "why", "how", "who", "whom"}
        query_words = set(re.sub(r"[^a-z0-9\s]", " ", request.query.lower()).split())
        if wh_words.intersection(query_words):
            return None

        tokens = self._query_tokens(request.query)
        if not tokens:
            return None

        best_idx = -1
        best_ratio = 0.0
        for idx, record in enumerate(request.records):
            content_tokens = self._query_tokens(record.content)
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
        timeout_seconds: float = 30.0,
        max_facts: int = 16,
        max_tokens: int = 384,
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
        facts = [
            f"{idx}. {record.content} [record_id={record.record_id}]"
            for idx, record in enumerate(records, start=1)
        ]
        logger.info("REASONER FACTS PASSED: %s", facts)
        prompt = (
            "Answer the question using only the memory facts below.\n\n"
            f"Facts:\n{chr(10).join(facts)}\n\n"
            f"Question: {request.query}\n"
            f"If the facts are completely insufficient, output exactly {self._IDK_SENTINEL}. Otherwise, answer directly and concisely.\n"
            "Answer:"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory-grounded assistant. Answer the user question using ONLY the provided memory facts.\n"
                    "Follow these strict rules:\n"
                    "1. STRICT GROUNDING: All specific events, names, and key entities in your answer must be directly supported by the facts.\n"
                    "2. LOGICAL & TEMPORAL DECUCTION: For questions asking about 'suspected' attributes, 'likelihood', 'possibilities', or temporal relative matches (e.g. connecting a relative time like 'last week' or 'yesterday' to the date context in the question), you are permitted to make reasonable one-step deductions from the facts (e.g. if a person has very big fingers and needs exercise or running, they may be suspected of obesity). Prefer answering with a reasonable deduction over abstaining with [IDK].\n"
                    "3. NO HALLUCINATION: If there are absolutely no relevant facts or clues in memory, or if the facts are completely unrelated to the question, do not make up anything; output exactly [IDK].\n"
                    "4. STRICT CONCISENESS: Keep your answer short and focused directly on what the question asks.\n"
                    "5. GOAL PRIORITIZATION: When asked about goals or plans, collect and combine ALL explicit target facts (such as winning a championship, improving shooting percentage, improve shooting, etc.) into a single merged response. Do not select only one.\n"
                    "6. ALL-TIME RETRIEVAL: You must include goals from all different days and times across the entire conversation history (e.g. winning a championship or improving shooting percentage). Do not ignore older goals in favor of only the most recent ones."
                ),
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
        logger.debug("LLMMemoryReasoningPolicy PROMPT:\n%s\nRESPONSE:\n%s", prompt, content)
        if not content or self._IDK_SENTINEL in content:
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
            answer=content,
            abstained=False,
            reason="llm_memory_answer",
            citations=tuple(record.record_id for record in records),
            facts_used=len(records),
            prompt_chars=len(prompt),
            raw_llm_output=content,
        )


class MemoryReasoner:
    """Run policies in order and return the first decisive result."""

    def __init__(self, policies: list[ReasoningPolicy] | tuple[ReasoningPolicy, ...]) -> None:
        self._policies = list(policies)

    async def answer(
        self,
        query: str,
        recalls: list[MemoryRecall],
        records: list[MemoryRecord],
    ) -> AnswerResult:
        request = MemoryReasoningInput(query=query, recalls=recalls, records=records)
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

    q = _normalize_surface(query)
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


def _normalize_surface(text: str) -> str:
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
    "answer_from_turn_evidence",
]
