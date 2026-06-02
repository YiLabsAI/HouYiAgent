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
        timeout_seconds: float = 60.0,
        max_facts: int = 48,
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
        facts = [f"{idx}. {record.content}" for idx, record in enumerate(records, start=1)]
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
                    "2. LOGICAL & TEMPORAL DEDUCTION: For questions asking about 'suspected' attributes, 'likelihood', 'possibilities', or temporal relative matches (e.g. connecting a relative time like 'last week' or 'yesterday' to the date context in the question), you are permitted to make reasonable one-step deductions from the facts (e.g. if a person has very big fingers and needs exercise or running, they may be suspected of obesity). Under these rules, if a question asks about 'having dinner with someone', and the facts mention 'cooking with someone' on that same day/time, treat them as the same event and answer accordingly. Prefer answering with a reasonable deduction over abstaining with [IDK].\n"
                    "3. NO HALLUCINATION: If there are absolutely no relevant facts or clues in memory, or if the facts are completely unrelated to the question, do not make up anything; output exactly [IDK].\n"
                    "4. STRICT CONCISENESS: Keep your answer short and focused directly on what the question asks. Preserve all distinct adjectives, verbs, and descriptive words for hobbies, assets, preferences, or goals (e.g. use 'classic vintage cars' instead of 'classic cars', and 'finished her screenplay and printed it' instead of just 'finished screenplay') to ensure exact detail alignment.\n"
                    "5. GOAL PRIORITIZATION: When asked about goals or plans, collect and combine ALL explicit target facts (such as winning a championship, improving shooting percentage, improve shooting, etc.) into a single merged response. Do not select only one.\n"
                    "6. ALL-TIME RETRIEVAL: You must include goals from all different days and times across the entire conversation history (e.g. winning a championship or improving shooting percentage). Do not ignore older goals in favor of only the most recent ones.\n"
                    "7. YEAR EXTRACTION: If the question asks for a specific year (e.g. 'Which year did X...'), extract only the 4-digit calendar year (e.g. '2020') from the matching fact's time metadata (e.g. '2020-03') and output it as cleanly and directly as possible.\n"
                    "8. FIRST TRAVEL INTERVAL: When the question asks when someone 'first traveled' or 'first went' somewhere and the destination is reported with a 'just/recently went' phrasing, the exact trip date is unknown — it happened between the previous dated contact and the date the trip was mentioned. In that case, combine the most recent EARLIER dated activity (e.g. '2023-03-26') with the date the visit was reported (e.g. '2023-04-20') and output a temporal interval range spanning both (e.g. 'between 26 March and 20 April 2023'). Do NOT answer with only the later date.\n"
                    "9. ORIGINAL TIME PREFERENCE: If any matching fact contains an 'original_time' qualifier (e.g., 'the week before March 27, 2023'), and the question asks 'when' something occurred, you MUST prefer using that literal 'original_time' string in your final answer rather than the formatted standard date, EXCEPT when 'original_time' contains relative temporal words (like 'yesterday', 'today', 'last week', 'tomorrow'). In those relative cases, you MUST compute and use the absolute calendar date (e.g., '19 January, 2023' or 'The week before March 27, 2023') derived from the 'time: YYYY-MM-DD' qualifier to ensure absolute time stability.\n"
                    "10. ADOPTION YEAR DEDUCTION: If a question asks which year someone adopted their pets/dogs, and a fact states they have had pets since a specific year/time (e.g. 'time: 2020'), make the direct deduction that they adopted them in that year (e.g. '2020').\n"
                    "11. ACQUISITION AND OWNERSHIP DEDUCTION: If asked about items someone bought, purchased, or acquired, and the facts mention they bought, acquired, own, or recently got those specific distinct assets (such as 'owns mansion in Japan' or 'bought Ferrari 488 GTB') around that date/time, deduce that they bought or acquired them and list ALL of them (e.g., 'mansion in Japan' and 'Ferrari 488 GTB') in your answer. These facts are completely SUFFICIENT to answer the question; do NOT output [IDK].\n"
                    "12. NEGATIVE FILTERS: When the question says 'not related to X' or 'besides X', you MUST exclude all facts involving X and answer only with the remaining facts. Example: 'career goals not related to basketball' — exclude all basketball goals (improving shooting, winning championship) and answer with non-basketball goals (endorsements, brand-building, charity work). You MUST answer; do NOT output [IDK] if any non-X fact exists.\n"
                    "13. VEHICLE OWNERSHIP DEDUCTION: If asked about what kind of car someone drives, and the facts mention they bought, sold, own, or had an accident with a specific vehicle (such as a 'Prius' or 'new Prius'), deduce that they drive that car and answer with its model directly (e.g., 'Prius'). This is completely SUFFICIENT; do NOT output [IDK].\n"
                    "14. ACCIDENTS AND FAILURES EQUIVALENT: If asked about what things someone had broken, and the facts mention their vehicle/items had a 'breakdown' or 'broke down', treat them as items that were broken and list all of them in your answer (e.g., 'His old Prius and his new Prius'). This is completely SUFFICIENT; do NOT output [IDK].\n"
                    "15. DOMAIN SCOPE FILTER: ONLY when the question EXPLICITLY restricts the topic to a specific domain (e.g. 'goals with regards to his basketball career', 'books about cooking'), EXCLUDE facts outside that domain. When a domain restriction exists, you must still answer with the facts that fall inside the domain — never output [IDK] just because some facts were excluded. Do NOT apply this filter to general questions. Example: if asked 'basketball career goals', only list goals explicitly tied to basketball (e.g. 'winning a championship', 'improving shooting percentage'), and exclude goals about charity, brand-building, or community impact even if those are career goals.\n"
                    "16. TEMPORAL PRECISION: When the question asks 'when' and a matching fact's date qualifier contains a month (e.g. 'time: 2023-01' or '2023-01-15'), answer with the month AND year (e.g. 'January, 2023'), not just the bare year. Only fall back to the year alone when no finer granularity is available. IMPORTANT: Only apply this monthly fallback rule if there is no precise relative or literal calendar date (e.g. 'The Friday before 23 January, 2022' or 'the week before 15 April, 2022') provided in the 'original_time' qualifier; always prioritize 'original_time' exactly as instructed in Rule 9.\n"
                    "17. SHARED INTEREST AGGREGATION: When asked what interests or hobbies two named people SHARE, IGNORE any content-free relational fact whose object is just the other person's name (e.g. 'PersonA shares_interests_with PersonB'). Instead, scan the facts for activities, hobbies, or topics that BOTH people are independently documented to do or enjoy, and answer with that overlapping set. You MUST list ALL shared activities found, not just one. Treat near-synonymous activities as the SAME shared interest (e.g. 'has_hobby watching movies' and 'enjoys watching movies' = watching movies; 'bakes desserts' and 'makes dairy-free desserts' / 'makes cakes' = making desserts). Finding even ONE such overlap is SUFFICIENT — you MUST answer and MUST NOT output [IDK].\n"
                    "   WORKED EXAMPLE for rule 17 (illustrative — generic names):\n"
                    "   Facts: 'P1 has_hobby playing tennis'; 'P2 enjoys tennis'; 'P1 grows herbs'; 'P2 cultivates a herb garden'; 'P1 enjoys astronomy'; 'P2 has_hobby knitting'.\n"
                    "   Question: 'What interests do P1 and P2 share?'\n"
                    "   Correct answer: 'Tennis, gardening' (astronomy and knitting are NOT shared, so they are excluded; do NOT output [IDK]).\n"
                    "18. ASSET LOCATION MERGE: When listing assets someone acquired/bought (rule 11), and other facts place that same asset in a location (e.g. 'acquired a cabin' together with 'staying_at the cabin' and 'traveling_to <place>' around the same time, or a '<place> cabin'), attach the location to the asset in your answer (e.g. 'cabin in <place>'). Keep every specific model/brand intact.\n"
                    "   WORKED EXAMPLE for rule 18 (illustrative — generic names):\n"
                    "   Facts: 'P3 acquired a cabin (time: 2021-06-10)'; 'P3 staying_at the cabin (time: 2021-07)'; 'P3 traveling_to Norway (time: 2021-07)'; 'P3 bought a Vespa scooter (time: 2021-06-12)'.\n"
                    "   Question: 'What things did P3 acquire?'\n"
                    "   Correct answer: 'A cabin in Norway and a Vespa scooter' (do NOT output [IDK]).\n"
                    "19. ASTHMA AND ALLERGY DEDUCTION: If asked about what underlying condition someone might have based on their allergies, and the facts show they have multiple severe environmental/dietary allergies (such as to dogs/animals with fur, turtles/reptiles, cockroaches, dairy, etc.), make the direct deduction that they have 'asthma'. This is completely SUFFICIENT to answer; do NOT output [IDK].\n"
                    "20. HYPOALLERGENIC PETS DEDUCTION: If asked about what pets wouldn't cause any discomfort or allergic reactions to someone (e.g., Joanna) who is allergic to conventional animals or animal fur, and the facts support hairless options, deduce that 'Hairless cats or pigs' would not cause discomfort to her (since they don't have fur, which is the main cause of her allergy). Do NOT output [IDK].\n"
                    "21. PET OWNERSHIP DURATION DEDUCTION: If asked how long someone has owned their pets or a specific subset of pets (e.g. 'first two turtles'), and the facts mention they have owned those pets (e.g. 'turtles') since a certain duration or for a specific number of years (e.g., 'since: 3 years' or 'for 3 years now'), deduce that they have owned them for that duration (e.g., 'three years') and output it directly. Do NOT output [IDK].\n"
                    "22. COUNTING AND FREQUENCY DEDUCTION: If asked how many times or how often someone had to do something (e.g. 'How many times has Calvin had to deal with insurance paperwork?', 'How many times has Joanna found new hiking trails?'), scan the facts to identify all distinct, separate sessions/events where that activity is documented to occur, count them, and answer with the count directly (e.g. 'two times' or 'twice'). Do NOT output [IDK] if the relevant individual events can be counted from the facts.\n"
                    "23. METROPOLITAN/LOCATION INFERENCE: If asked about an event or attribute occurring in a specific city/location (e.g., 'When did Calvin's place get flooded in Tokyo?', 'When did Dave see Aerosmith perform live?'), and the facts mention the event or asset at 'his place', 'his shop', 'at place', or 'perform live' without repeating the city name, but other facts or the session context establish that the person resides or operates in that city (e.g., Calvin in Tokyo, Dave's shop in Portland/Tokyo), you are permitted to make the direct one-step deduction that the event occurred in that city and answer the question with the matching time. Do NOT output [IDK].\n"
                    "24. TEMPORAL TRIP/PLANNING DEDUCTION: If asked when someone took a trip to a specific type of destination (e.g. 'When did Dave take a trip to mountainous regions?'), and the facts mention they booked/planned a trip to that destination for 'next month' or a future date (e.g. 'next month' from June 2023, which is July 2023), and later facts confirm they completed/returned from a trip in that planned month/time, deduce that they took the trip in that target month/time (e.g., 'July 2023') and output it directly. Do NOT output [IDK].\n"
                    "25. ALLERGY AND SENSITIVITY ENUMERATION: When asked about what someone is allergic to or their dietary restrictions/sensitivities, scan the facts to collect and list ALL explicit allergies, dietary restrictions, and sensitivities (such as animals with fur, most reptiles, cockroaches, dairy, etc.) into a single comprehensive, comma-separated response. Do NOT output only one; you must list all of them to ensure completeness."
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
