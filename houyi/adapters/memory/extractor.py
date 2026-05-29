"""Memory extractors.

Two extractor flavours co-exist in this module:

1. MemoryCandidateExtractor — heuristic + LLM extractor that
 produces MemoryCandidate objects (loose payload + scope + type).
 Used by the candidate ingestion path that classifies/dedup/promotes
 candidates downstream.
2. AtomicFactExtractor — LLM-driven extractor that produces
 strict AtomicFact 6-tuples (subject/predicate/object + certainty
 + source_anchor + qualifiers). Used by the new write path for
 entity-state materialization and bi-temporal updates.

The two surfaces target different downstream contracts and intentionally
do not share state; pick the one matching the caller's pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    ExtractionContext,
    MemoryCandidate,
    MemorySourceKind,
    MemoryType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based patterns (English-only fallback)
# ---------------------------------------------------------------------------

_EXPLICIT_MEMORY_PATTERN = re.compile(
    r"(?i)(?:remember (?:that |this[: ])?|note (?:that |this[: ])?|"
    r"keep in mind[: ]|don't forget[: ])(.*)",
)

_IDENTITY_PATTERN = re.compile(
    r"(?i)(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)

_PREFERENCE_PATTERN = re.compile(
    r"(?i)(?:i (?:prefer|like|love|enjoy|hate|dislike|always use))\s+(.+?)(?:\.|$)",
)

_CONSTRAINT_PATTERN = re.compile(
    r"(?i)(?:don't|do not|never|avoid|stop)\s+(.+?)(?:\.|$)",
)

# Generic words that indicate the LLM replaced a specific term with a vague
# category.  Matched against the *object* field after extraction.  When a
# generic word is detected the certainty is downgraded so the fact enters the
# candidate inbox instead of the main store, reducing noise.
_GENERIC_OBJECT_WORDS: frozenset[str] = frozenset(
    {
        "ride",
        "car",
        "vehicle",
        "place",
        "city",
        "house",
        "friend",
        "person",
        "thing",
        "item",
        "job",
        "pet",
        "animal",
    },
)


def _detect_generic_object(item: dict[str, Any]) -> bool:
    """Return True if the *object* field looks like a generic placeholder."""
    obj = str(item.get("object", "")).strip().lower()
    if not obj:
        return False
    # Exact match against the blacklist (e.g. "ride", "car").
    if obj in _GENERIC_OBJECT_WORDS:
        return True
    # "new car", "my ride" — leading modifier + generic core word.
    tokens = obj.split()
    return len(tokens) > 1 and tokens[-1] in _GENERIC_OBJECT_WORDS


class MemoryCandidateExtractor:
    """Extract MemoryCandidate rows via LLM (primary) or rules (fallback)."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.6,
        llm_adapter: Any | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._llm = llm_adapter

    async def extract(
        self,
        messages: list[dict[str, Any]],
        *,
        context: ExtractionContext | None = None,
    ) -> list[MemoryCandidate]:
        ctx = context or ExtractionContext()
        if self._llm is not None:
            candidates = await self._extract_via_llm(messages, ctx)
            if candidates:
                return candidates
        return self._extract_via_rules(messages, ctx)

    async def _extract_via_llm(
        self,
        messages: list[dict[str, Any]],
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        if self._llm is None:
            return []
        user_messages = [m for m in messages if str(m.get("role", "")).lower() == "user"]
        if not user_messages:
            return []
        user_text = "\n".join(str(m.get("content", "")) for m in user_messages if m.get("content"))
        if not user_text.strip():
            return []

        prompt = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        try:
            response = await self._llm.chat(prompt, temperature=0.1, max_tokens=512)
        except Exception:
            return self._extract_via_rules(messages, ctx)
        content = str(getattr(response, "content", "") or "")
        return self._parse_llm_response(content, ctx, source_messages=user_messages)

    def _parse_llm_response(
        self,
        content: str,
        ctx: ExtractionContext,
        *,
        source_messages: list[dict[str, Any]] | None = None,
    ) -> list[MemoryCandidate]:
        decoded = self._decode_candidate_json(content)
        if not decoded:
            return []
        msg_id = self._first_message_id(source_messages)
        out: list[MemoryCandidate] = []
        for item in decoded:
            candidate = self._candidate_from_item(item, ctx, msg_id)
            if candidate is not None:
                out.append(candidate)
        return out

    @staticmethod
    def _decode_candidate_json(content: str) -> list[dict[str, Any]]:
        cleaned = _ATOMIC_FENCE_RE.sub("", content or "").strip()
        if not cleaned:
            return []
        try:
            decoded = json.loads(cleaned)
        except Exception:
            span = _extract_json_span(cleaned)
            if not span:
                return []
            try:
                decoded = json.loads(span)
            except Exception:
                return []
        if not isinstance(decoded, list):
            return []
        return [item for item in decoded if isinstance(item, dict)]

    @staticmethod
    def _first_message_id(source_messages: list[dict[str, Any]] | None) -> str:
        if not source_messages:
            return ""
        for msg in source_messages:
            raw = str(msg.get("id", "") or "").strip()
            if raw:
                return raw
        return ""

    def _candidate_from_item(
        self,
        item: dict[str, Any],
        ctx: ExtractionContext,
        msg_id: str,
    ) -> MemoryCandidate | None:
        text = str(item.get("content", "") or "").strip()
        if not text:
            return None
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if confidence < self._min_confidence:
            return None
        memory_type = self._parse_memory_type(str(item.get("type", "fact") or "fact"))
        return self._make_candidate(
            content=text,
            memory_type=memory_type,
            confidence=confidence,
            message_id=msg_id,
            ctx=ctx,
        )

    def _extract_via_rules(
        self,
        messages: list[dict[str, Any]],
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        out: list[MemoryCandidate] = []
        for msg in messages:
            if str(msg.get("role", "")).lower() != "user":
                continue
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            message_id = str(msg.get("id", "") or "")

            explicit = _EXPLICIT_MEMORY_PATTERN.search(content)
            if explicit and explicit.group(1).strip():
                out.append(
                    self._make_candidate(
                        content=explicit.group(1).strip(),
                        memory_type=MemoryType.FACT,
                        confidence=0.95,
                        message_id=message_id,
                        ctx=ctx,
                    )
                )

            for m in _IDENTITY_PATTERN.finditer(content):
                value = m.group(1).strip()
                out.append(
                    self._make_candidate(
                        content=f"User name: {value}",
                        memory_type=MemoryType.PROFILE,
                        confidence=0.9,
                        message_id=message_id,
                        ctx=ctx,
                    )
                )

            for m in _PREFERENCE_PATTERN.finditer(content):
                value = m.group(1).strip()
                out.append(
                    self._make_candidate(
                        content=f"User preference: {value}",
                        memory_type=MemoryType.PREFERENCE,
                        confidence=0.7,
                        message_id=message_id,
                        ctx=ctx,
                    )
                )

            for m in _CONSTRAINT_PATTERN.finditer(content):
                value = m.group(1).strip()
                out.append(
                    self._make_candidate(
                        content=f"User constraint: {value}",
                        memory_type=MemoryType.CONSTRAINT,
                        confidence=0.9,
                        message_id=message_id,
                        ctx=ctx,
                    )
                )

        return [c for c in out if c.confidence >= self._min_confidence]

    @staticmethod
    def _parse_memory_type(raw: str) -> MemoryType:
        value = raw.strip().lower()
        if value == "profile":
            return MemoryType.PROFILE
        if value == "preference":
            return MemoryType.PREFERENCE
        if value == "constraint":
            return MemoryType.CONSTRAINT
        return MemoryType.FACT

    @staticmethod
    def _make_candidate(
        *,
        content: str,
        memory_type: MemoryType,
        confidence: float,
        message_id: str,
        ctx: ExtractionContext,
    ) -> MemoryCandidate:
        source_ids = [message_id] if message_id else []
        return MemoryCandidate(
            content=content,
            memory_type=memory_type,
            confidence=confidence,
            source_type=MemorySourceKind.CONVERSATION.value,
            source_message_ids=source_ids,
            source_context=f"turn:{ctx.turn_index}" if ctx.turn_index else "",
            metadata={},
        )


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction assistant. Analyze user messages and extract \
memorable personal information — facts, preferences, constraints, and \
identity details.

Return a JSON array of objects. Each object has:
- "content": a concise sentence describing the memory
- "type": one of "fact", "preference", "constraint", "profile"
- "confidence": a float 0.0-1.0

Rules:
- Only extract from USER messages, ignore assistant messages
- Prefer the user's own language for the content field
- "profile" = name, role, title, location, employer, organization
- "preference" = things the user likes, prefers, or habitually uses
- "constraint" = things the user dislikes, wants to avoid, or forbids
- "fact" = explicit instructions to remember, deadlines, project details
- Any statement revealing the user's identity (name, job, affiliation) \
MUST be typed as "profile", regardless of language
- Skip small talk, greetings, and questions that reveal no personal info
- If nothing memorable, return []

Example 1:
 User: "My name is Alice. I prefer Python for data science. Never use tabs."
Output:
 [
 {"content": "User name: Alice", "type": "profile", "confidence": 0.95},
 {"content": "User prefers Python for data science", "type": "preference", "confidence": 0.85},
 {"content": "Never use tabs", "type": "constraint", "confidence": 0.85}
 ]

Example 2:
 User: "I work at Google as a senior engineer."
Output:
 [
 {"content": "User works at Google as a senior engineer", "type": "profile", "confidence": 0.90}
 ]
"""

_ATOMIC_FACT_BATCH_SYSTEM_PROMPT = """You are an information extraction engine.
Extract atomic facts from multiple turns. Each turn is prefixed by:
<<TURN id=...>>
The turn content is a JSON object with:
 - text: The conversation turn text.
 - speaker_name: The canonical name of the speaker (use this as the subject of extracted facts about the speaker).
 - observation_date: The date when the conversation occurred.
 - system_date: Today's date.

Return JSON in this shape:
{
  "items": [
    {
      "source_anchor": "<turn-id>",
      "facts": [
        {
          "subject": "...",
          "predicate": "...",
          "object": "...",
          "certainty": "certain|probable|vague",
          "accumulate": false,
          "qualifiers": {"k": "v"}
        }
      ]
    }
  ]
}

Rules:
1) Keep each fact under the correct source_anchor matching the <<TURN id=...>> prefix.
2) If no fact is present for a turn, return that source_anchor with an empty facts array.
3) Use the person's actual speaker_name if the fact is about the speaker.
4) Return only JSON, no markdown or prose.
5) PROPER NOUN & SPECIFIC DETAIL PRESERVATION: Do not generalize, abstract, or drop specific proper nouns, names, brands, models, objects, or locations. For example, "Ferrari 488 GTB" must NOT be simplified to "new car" or "luxury car"; keep "Ferrari 488 GTB" intact in the "object". Preserve specific names of people, places, and distinct assets (e.g. "mansion", "Karlie") rather than replacing them with generic categories like "friend" or "house".
6) OBJECT SPECIFICITY RULE: The "object" field MUST preserve the exact specific term used in the original text. NEVER replace a concrete brand, model, name, or location with a generic category word. Forbidden generic replacements: ride, car, vehicle, place, city, house, friend, person, thing, item, job, pet, animal. WRONG: {"object": "new car"} RIGHT: {"object": "Ferrari 488 GTB"} WRONG: {"object": "ride"} RIGHT: {"object": "new Prius"} WRONG: {"object": "city"} RIGHT: {"object": "San Francisco"}
7) TEMPORAL LITERAL PRESERVATION: When resolving relative time references (e.g. "the week before March 27, 2023", "last week") and computing standard dates for "qualifiers.date" or "qualifiers.since", you MUST ALSO record the exact original relative/literal time string (e.g., "the week before March 27, 2023") under a key "original_time" inside the "qualifiers" object. Do not lose the literal wording of the temporal event.
8) JOINT ENTITY RESOLUTION (WE/US/OUR): If the speaker refers to themselves and their partner/friend/family as "we", "us", or "our" when discussing shared interests, activities, preferences, or states (e.g., "We love making desserts"), you MUST extract symmetrical individual facts for BOTH people. Extract one fact with the speaker's name as subject, and another identical fact with the partner's name as subject. Do NOT output a vague subject like "we".
9) GOALS/CAREER ASPIRATIONS ALIGNMENT: To ensure reliable recall of future intentions, any career goals, life ambitions, or mid-to-long term plans (e.g., "making a difference away from the court through charity", "getting endorsements", "building a brand") MUST be extracted using the predicate "has_goal" or "career_goal", rather than vague predicates like "wants_to", "likes", or "dislikes".
10) LIFETIME EVENT & TRANSITION PRESERVATION: You MUST explicitly extract active milestone transition actions (e.g. "adopted", "bought", "started_job", "lost_job") and record the exact year/month of occurrence in the qualifiers (e.g., "date: 2020", "date: 2023-03"). Do NOT collapse active transitions into static states (e.g. do not turn "adopted 3 dogs in 2020" into just "owns dogs"). Preserve both the action and its exact date/year.
11) EXHAUSTIVE ENUMERATION: Extract EVERY distinct fact stated in a turn. When a turn names multiple items of the same kind (several books, hobbies, activities, authors, places, foods), emit a SEPARATE fact for EACH item with "accumulate": true. NEVER collapse a list into one representative item, and NEVER drop the "minor" items. If a speaker names five books, output five separate facts — one per book.
12) LIFE EVENTS ARE MANDATORY: Always extract one-time life events even when stated briefly or in passing, AND even when they concern the speaker's own family member or friend (e.g. a parent's or friend's death, a donation, selling/disposing of an item, acquiring a specific asset, losing a job). Use the named person or the speaker's relative as subject (e.g. subject "Deborah" with predicate "lost_family_member", object "father"). Suggested predicates: lost_family_member, lost_friend, donated, sold, bought, acquired, lost_job. Preserve the proper noun (the named friend, the specific asset) and the date. A relative's or friend's death reported by the speaker is part of the speaker's own life and MUST be extracted — it is NOT "unrelated third-party information".
13) DESCRIPTIVE FIDELITY: Reproduce the speaker's exact descriptive words. NEVER replace a term with its opposite or a near-synonym (do not turn "sunrise" into "sunset", "old" into "new", "donated" into "owns"). Keep stated time expressions (e.g. "last year") and ALSO run them through DATE HANDLING to record a qualifier date.

EXAMPLES:

Input:
<<TURN id=conv-47:D1:27>>
{"observation_date": "2023-03-16", "system_date": "2024-01-15", "text": "I can't bowl, my fingers are too big. Perhaps I should take up exercise, at least start going for a run in the morning. And I also don't like bowling itself, to be honest.", "speaker_name": "John"}

Output:
{
  "items": [
    {
      "source_anchor": "conv-47:D1:27",
      "facts": [
        {"subject": "John", "predicate": "dislikes", "object": "bowling", "certainty": "certain"},
        {"subject": "John", "predicate": "has_attribute", "object": "big fingers", "certainty": "certain"},
        {"subject": "John", "predicate": "considers", "object": "exercise", "certainty": "certain"},
        {"subject": "John", "predicate": "considers", "object": "running in the morning", "certainty": "certain"},
        {"subject": "John", "predicate": "suspected_health_issue", "object": "obesity", "certainty": "probable"}
      ]
    }
  ]
}

Input:
<<TURN id=conv-43:D1:9>>
{"observation_date": "2023-01-15", "system_date": "2024-01-15", "text": "Yeah, my goal is to improve my shooting percentage. Been practicing hard and gonna make it happen.", "speaker_name": "John"}

<<TURN id=conv-43:D6:15>>
{"observation_date": "2023-01-22", "system_date": "2024-01-15", "text": "Yeah! Winning a championship is my number one goal. But I also want to make a difference away from the court, like through charity.", "speaker_name": "John"}

Output:
{
  "items": [
    {
      "source_anchor": "conv-43:D1:9",
      "facts": [
        {"subject": "John", "predicate": "has_goal", "object": "improve shooting percentage", "certainty": "certain"},
        {"subject": "John", "predicate": "practices", "object": "basketball", "certainty": "certain"}
      ]
    },
    {
      "source_anchor": "conv-43:D6:15",
      "facts": [
        {"subject": "John", "predicate": "primary_goal", "object": "winning a championship", "certainty": "certain"},
        {"subject": "John", "predicate": "wants_to", "object": "make a difference through charity", "certainty": "certain"}
      ]
    }
  ]
}

Input:
<<TURN id=ex:D1:14>>
{"observation_date": "2023-05-08", "system_date": "2024-01-15", "text": "Yeah, I painted that lake sunrise last year! It's special to me.", "speaker_name": "Melanie"}

<<TURN id=ex:D2:1>>
{"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "Sorry to tell you my dad passed away two days ago. I still really miss my friend Karlie too.", "speaker_name": "Deborah"}

<<TURN id=ex:D3:1>>
{"observation_date": "2023-05-04", "system_date": "2024-01-15", "text": "I donated my old car to a homeless shelter yesterday. And last week I finally got a new Ferrari!", "speaker_name": "Maria"}

<<TURN id=ex:D4:1>>
{"observation_date": "2023-06-01", "system_date": "2024-01-15", "text": "I love fantasy. I've read Harry Potter, The Hobbit, and A Dance with Dragons.", "speaker_name": "Tim"}

Output:
{
  "items": [
    {
      "source_anchor": "ex:D1:14",
      "facts": [
        {"subject": "Melanie", "predicate": "painted", "object": "lake sunrise", "certainty": "certain", "qualifiers": {"date": "2022", "original_time": "last year"}}
      ]
    },
    {
      "source_anchor": "ex:D2:1",
      "facts": [
        {"subject": "Deborah", "predicate": "lost_family_member", "object": "father", "certainty": "certain", "qualifiers": {"date": "2023-05-01"}},
        {"subject": "Deborah", "predicate": "lost_friend", "object": "Karlie", "certainty": "certain"}
      ]
    },
    {
      "source_anchor": "ex:D3:1",
      "facts": [
        {"subject": "Maria", "predicate": "donated", "object": "old car", "certainty": "certain", "qualifiers": {"date": "2023-05-03", "recipient": "homeless shelter"}},
        {"subject": "Maria", "predicate": "bought", "object": "Ferrari", "certainty": "certain", "qualifiers": {"date": "2023-04-27", "original_time": "last week"}}
      ]
    },
    {
      "source_anchor": "ex:D4:1",
      "facts": [
        {"subject": "Tim", "predicate": "reads_book", "object": "Harry Potter", "certainty": "certain", "accumulate": true},
        {"subject": "Tim", "predicate": "reads_book", "object": "The Hobbit", "certainty": "certain", "accumulate": true},
        {"subject": "Tim", "predicate": "reads_book", "object": "A Dance with Dragons", "certainty": "certain", "accumulate": true}
      ]
    }
  ]
}

DATE HANDLING:
Relative time mappings (use observation_date as anchor):
 - yesterday -> observation_date minus 1 day
 - today -> observation_date
 - tomorrow -> observation_date plus 1 day
 - last week -> observation_date minus 7 days
 - last weekend -> most recent Saturday/Sunday before observation_date
 - next week -> observation_date plus 7 days
 - last Monday -> Most recent Monday before observation_date
 - next Monday -> First Monday after observation_date
 - two days ago -> observation_date minus 2 days
 - N years ago -> observation_date minus N years
 - N months ago -> observation_date minus N months
 - for N years -> started at observation_date minus N years; record qualifier since: that computed year
 - for N months -> started at observation_date minus N months; record qualifier since: that computed year-month
 - last year -> year of (observation_date minus 1 year); record qualifier date: that year
 - last month -> year-month of (observation_date minus 1 month); record qualifier date: that year-month
 - just / recently / no explicit time -> the event happened at observation_date; record qualifier date: observation_date

CERTAINTY RUBRIC:
 - certain: Direct factual statements without hedging.
 - probable: Hedged but concrete statements (e.g. I think..., probably).
 - vague: Non-committal or evasive (kind of stuck, maybe, not sure).

ACCUMULATE FLAG:
 - accumulate: true - for open-ended sets (visited places, collected items, read books, known bands, past jobs, recurring activities).
 - accumulate: false - for single current values (current job, current address, current relationship status, name, age).

PREDICATE CONSISTENCY:
 - Books read/mentioned: always use predicate reads_book
 - Bands/artists enjoyed: always use predicate likes_band
 - Places visited: always use predicate visited_place
 - Items collected: always use predicate collects
 - Past jobs/roles: always use predicate had_job
"""

_ATOMIC_FACT_SYSTEM_PROMPT = """You are an information extraction engine.
Extract discrete factual claims from user messages as structured facts.

OUTPUT SCHEMA (JSON array):
Each item MUST have these fields:
 - "subject": the entity the fact is about. Use the person's actual name if known (e.g., "Caroline", "Bob"), otherwise "user" for the speaker, or the entity identifier (e.g., "project_x")
 - "predicate": short snake_case attribute name (e.g., "lives_in", "went_to", "has_meeting")
 - "object": the value as a short string
 - "certainty": one of "certain", "probable", "vague"
 - "qualifiers": OPTIONAL object with extra attributes like "since", "date", "location"

DATE HANDLING (CRITICAL):
Two dates are provided in each input:
 - "observation_date": The date when the conversation occurred. Use THIS to resolve relative time references.
 - "system_date": Today's date (for context only, do NOT use for resolving "yesterday", "last week", etc.)

Relative time mappings (use observation_date as anchor):
 - "yesterday" → observation_date minus 1 day
 - "today" → observation_date
 - "tomorrow" → observation_date plus 1 day
 - "last week" → observation_date minus 7 days
 - "last weekend" → most recent Saturday/Sunday before observation_date
 - "next week" → observation_date plus 7 days
 - "last Monday" → Most recent Monday before observation_date
 - "next Monday" → First Monday after observation_date
 - "two days ago" → observation_date minus 2 days
 - "N years ago" → observation_date minus N years (year only)
 - "N months ago" → observation_date minus N months
 - "for N years" (current possession/state) → started at observation_date minus N years; record as qualifier "since": that computed year
 - "for N months" → started at observation_date minus N months; record as qualifier "since": that computed year-month
 - "last year" → year of (observation_date minus 1 year); record as qualifier "date": that year
 - "last month" → year-month of (observation_date minus 1 month); record as qualifier "date": that year-month
 - "just" / "recently" / no explicit time → the event happened at observation_date; record as qualifier "date": observation_date

CERTAINTY RUBRIC:
 - "certain": Direct factual statements without hedging ("I live in Beijing", "My name is Alice")
 - "probable": Hedged but concrete statements ("I think we deploy on Friday", "Probably Postgres")
 - "vague": Non-committal or evasive ("kind of stuck", "maybe", "I'm not sure")

ACCUMULATE FLAG:
 - "accumulate": true  — the fact is ONE ITEM in an open-ended set that grows over time.
   Use for: visited places, collected items, read books, known bands/artists, past jobs,
   recurring activities, things the speaker has done more than once across conversations.
   Each item from each turn is a separate extraction; the engine will merge them.
 - "accumulate": false (default) — the fact is a SINGLE CURRENT VALUE that supersedes any
   previous value for the same attribute. Use for: current job, current address,
   current relationship status, name, age, single preference ("prefers Python").

PREDICATE CONSISTENCY (use these exact predicates for common accumulate categories):
 - Books read or mentioned: always use predicate "reads_book"
 - Bands/artists enjoyed: always use predicate "likes_band"
 - Places visited: always use predicate "visited_place"
 - Items collected: always use predicate "collects"
 - Past jobs/roles: always use predicate "had_job"

EXTRACTION BOUNDARIES:
EXTRACT:
 - Identity: name, role, employer, location
 - Preferences: likes, dislikes, habits, preferred tools/methods
 - Events with dates: meetings, appointments, past occurrences
 - Relationships: knows, works with, friends with
 - Facts about projects, tasks, deadlines

DO NOT EXTRACT:
 - Greetings and small talk ("Hello", "How are you?", "Good morning")
 - Questions asked by the user
 - Opinions about external topics not related to the speaker
 - Information about unrelated public figures or strangers the speaker has no connection to
 - General knowledge facts ("The sky is blue")

ALWAYS EXTRACT (these are part of the speaker's own life, never skip them):
 - Life events about the speaker's family or friends (e.g. a parent's or friend's death, marriage, birth) — use the named person or relative as subject (e.g. "Deborah lost_family_member father", "Deborah lost_friend Karlie").
 - One-time transactions: donating, selling, buying, or acquiring a specific item — preserve the specific object and the date.
 - Every distinct item when the speaker enumerates several of the same kind (books, hobbies, places, foods): emit one fact per item with "accumulate": true; never collapse the list.

PROPER NOUN & SPECIFIC DETAIL PRESERVATION:
- Do not generalize, abstract, or drop specific proper nouns, names, brands, models, objects, or locations. For example, "Ferrari 488 GTB" must NOT be simplified to "new car" or "luxury car"; keep "Ferrari 488 GTB" intact in the "object" field. Preserve specific names of people, places, and distinct assets (e.g. "mansion", "Karlie") rather than replacing them with generic categories like "friend" or "house".

OBJECT SPECIFICITY RULE (CRITICAL):
- The "object" field MUST preserve the exact specific term used in the original text. NEVER replace a concrete brand, model, name, or location with a generic category word.
- Forbidden generic replacements for "object": ride, car, vehicle, place, city, house, friend, person, thing, item, job, pet, animal
- If the original text mentions a specific brand/model/name, the "object" MUST contain that specific term, not a vague substitute.

WRONG vs RIGHT examples:
  WRONG: {"subject": "John", "predicate": "bought", "object": "new car"}
  RIGHT: {"subject": "John", "predicate": "bought", "object": "Ferrari 488 GTB"}

  WRONG: {"subject": "Jon", "predicate": "bought", "object": "ride"}
  RIGHT: {"subject": "Jon", "predicate": "bought", "object": "new Prius"}

  WRONG: {"subject": "Caroline", "predicate": "lives_in", "object": "city"}
  RIGHT: {"subject": "Caroline", "predicate": "lives_in", "object": "San Francisco"}

TEMPORAL LITERAL PRESERVATION:
- When resolving relative time references (e.g. "the week before March 27, 2023", "last week") and computing standard dates for "qualifiers.date" or "qualifiers.since", you MUST ALSO record the exact original relative/literal time string (e.g., "the week before March 27, 2023") under a key "original_time" inside the "qualifiers" object. Do not lose the literal wording of the temporal event.

---

FEW-SHOT EXAMPLES:

Example 1 - Identity (name becomes subject):
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "My name is Alice and I work at Google."}
Output:
[
  {"subject": "Alice", "predicate": "name_is", "object": "Alice", "certainty": "certain"},
  {"subject": "Alice", "predicate": "employer", "object": "Google", "certainty": "certain"}
]

Example 1b - Self-reference with known identity:
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "I went to the store yesterday.", "speaker_name": "Caroline"}
Output:
[
  {"subject": "Caroline", "predicate": "went_to", "object": "store", "certainty": "certain", "qualifiers": {"date": "2023-05-02"}}
]

Example 2 - Preferences:
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "I prefer Python over JavaScript. I hate waiting in lines."}
Output:
[
  {"subject": "user", "predicate": "prefers", "object": "Python over JavaScript", "certainty": "certain"},
  {"subject": "user", "predicate": "dislikes", "object": "waiting in lines", "certainty": "certain"}
]

Example 3 - Temporal (yesterday, speaker is Caroline):
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "I went to the gym yesterday and worked out for an hour.", "speaker_name": "Caroline"}
Output:
[
  {"subject": "Caroline", "predicate": "went_to", "object": "gym", "certainty": "certain", "qualifiers": {"date": "2023-05-02", "duration": "1 hour"}}
]

Example 4 - Temporal (last week):
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "Last week I met with Bob to discuss the project."}
Output:
[
  {"subject": "user", "predicate": "met_with", "object": "Bob", "certainty": "certain", "qualifiers": {"date": "2023-04-26", "topic": "project"}}
]

Example 5 - Temporal (next Monday):
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "I have a dentist appointment next Monday."}
Output:
[
  {"subject": "user", "predicate": "has_appointment", "object": "dentist", "certainty": "certain", "qualifiers": {"date": "2023-05-08"}}
]

Example 6 - Vague information:
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "The project is kind of stuck, maybe we need to pivot."}
Output:
[
  {"subject": "project", "predicate": "status", "object": "stuck", "certainty": "vague"},
  {"subject": "project", "predicate": "may_need", "object": "pivot", "certainty": "vague"}
]

Example 7 - Negative (do not extract):
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "Hello! How is the weather today? I read that Paris is beautiful."}
Output: []

Example 8 - Relationships:
Input: {"observation_date": "2023-05-03", "system_date": "2024-01-15", "text": "My friend Sarah introduced me to her colleague Tom."}
Output:
[
  {"subject": "user", "predicate": "knows", "object": "Sarah", "certainty": "certain", "qualifiers": {"relationship": "friend"}},
  {"subject": "Sarah", "predicate": "knows", "object": "Tom", "certainty": "certain", "qualifiers": {"relationship": "colleague"}}
]

Example 9 - Job event with "yesterday" and role detail:
Input: {"observation_date": "2023-01-20", "system_date": "2024-01-15", "text": "Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.", "speaker_name": "Jon"}
Output:
[
  {"subject": "Jon", "predicate": "lost_job", "object": "banker", "certainty": "certain", "qualifiers": {"date": "2023-01-19", "former_role": "banker"}},
  {"subject": "Jon", "predicate": "plans_to", "object": "start own business", "certainty": "certain"}
]

Example 10 - Duration-based year back-calculation ("for N years"):
Input: {"observation_date": "2023-01-15", "system_date": "2024-01-15", "text": "I've had my cat for 3 years! Her name is Mimi.", "speaker_name": "Audrey"}
Output:
[
  {"subject": "Audrey", "predicate": "has_pet", "object": "cat", "certainty": "certain", "qualifiers": {"name": "Mimi", "since": "2020"}},
  {"subject": "Audrey", "predicate": "has_pet_name", "object": "Mimi", "certainty": "certain"}
]

Example 11 - Implied physical condition (probable certainty):
Input: {"observation_date": "2023-03-16", "system_date": "2024-01-15", "text": "I can't bowl, my fingers are too big. Perhaps I should start going for a run in the morning.", "speaker_name": "John"}
Output:
[
  {"subject": "John", "predicate": "dislikes", "object": "bowling", "certainty": "certain"},
  {"subject": "John", "predicate": "considers_starting", "object": "going for a run in the morning", "certainty": "certain"},
  {"subject": "John", "predicate": "suspected_health_issue", "object": "obesity", "certainty": "probable"}
]

Example 12 - Accumulate (open-ended set of visited places):
Input: {"observation_date": "2023-03-10", "system_date": "2024-01-15", "text": "My girlfriend and I tried out that new cafe scene in the city last weekend!", "speaker_name": "Andrew"}
Output:
[
  {"subject": "Andrew", "predicate": "visited_place", "object": "cafe", "certainty": "certain", "accumulate": true, "qualifiers": {"date": "2023-03-04", "with": "girlfriend"}}
]

Example 12b - reads_book and likes_band as canonical predicates:
Input: {"observation_date": "2023-08-11", "system_date": "2024-01-15", "text": "I just finished reading The Name of the Wind. My favorite band is Radiohead.", "speaker_name": "Tim"}
Output:
[
  {"subject": "Tim", "predicate": "reads_book", "object": "The Name of the Wind", "certainty": "certain", "accumulate": true},
  {"subject": "Tim", "predicate": "likes_band", "object": "Radiohead", "certainty": "certain", "accumulate": true}
]

Example 13 - Single-valued vs accumulate contrast:
Input: {"observation_date": "2023-06-01", "system_date": "2024-01-15", "text": "I just moved to Seattle. I collect sneakers and jerseys.", "speaker_name": "Jon"}
Output:
[
  {"subject": "Jon", "predicate": "lives_in", "object": "Seattle", "certainty": "certain", "accumulate": false},
  {"subject": "Jon", "predicate": "collects", "object": "sneakers", "certainty": "certain", "accumulate": true},
  {"subject": "Jon", "predicate": "collects", "object": "jerseys", "certainty": "certain", "accumulate": true}
]

Example 14 - "last weekend" — resolve to most recent Saturday/Sunday before observation_date:
Input: {"observation_date": "2023-03-26", "system_date": "2024-01-15", "text": "Last weekend, I went to a music festival in Boston.", "speaker_name": "Dave"}
Output:
[
  {"subject": "Dave", "predicate": "attended_event", "object": "music festival", "certainty": "certain", "qualifiers": {"date": "2023-03-18", "location": "Boston"}}
]

Example 15 - "I just went/traveled to X" — use observation_date as qualifier.date:
Input: {"observation_date": "2023-04-20", "system_date": "2024-01-15", "text": "I just went to an awesome music festival in Tokyo!", "speaker_name": "Calvin"}
Output:
[
  {"subject": "Calvin", "predicate": "visited_place", "object": "Tokyo", "certainty": "certain", "accumulate": true, "qualifiers": {"date": "2023-04-20"}}
]

Example 17b - Implicit identity from community context: infer identity predicate with probable certainty:
Input: {"observation_date": "2023-05-08", "system_date": "2024-01-15", "text": "The transgender stories were so inspiring! I was so happy and thankful for all the support.", "speaker_name": "Caroline"}
Output:
[
  {"subject": "Caroline", "predicate": "identity", "object": "transgender", "certainty": "probable"},
  {"subject": "Caroline", "predicate": "found_inspiring", "object": "transgender stories", "certainty": "certain"}
]

Example 17 - Indirect collection mention should extract the item as collects with accumulate=true:
Input: {"observation_date": "2023-05-21", "system_date": "2024-01-15", "text": "I love talking to people about my sneaker collection.", "speaker_name": "John"}
Output:
[
  {"subject": "John", "predicate": "collects", "object": "sneakers", "certainty": "certain", "accumulate": true}
]

Example 16 - "last year / last month" — subtract from observation_date:
Input: {"observation_date": "2023-01-23", "system_date": "2024-01-15", "text": "My mother also passed away last year.", "speaker_name": "Jolene"}
Output:
[
  {"subject": "Jolene", "predicate": "family_loss", "object": "mother", "certainty": "certain", "accumulate": true, "qualifiers": {"date": "2022"}}
]

---

INPUT FORMAT:
{"observation_date": "YYYY-MM-DD", "system_date": "YYYY-MM-DD", "text": "user message text", "speaker_name": "OptionalName"}

Note: speaker_name is optional. When provided, use it as the subject for facts about the speaker (instead of "user").

OUTPUT FORMAT:
JSON array of fact objects. Return [] if no extractable facts.
Each fact object fields: subject, predicate, object, certainty, accumulate (bool, default false), qualifiers (optional object).
"""


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a single AtomicFact extraction call.

    - facts - successfully validated AtomicFact list.
    - raw_sourceless - LLM items parsed but rejected because the
    caller could not supply a source_anchor;
    the raw payloads are forwarded so the
    candidate inbox can park them for later
    replay once a source becomes available.
    - invalid_dropped - count of items the LLM produced that failed
    schema validation (bad certainty value,
    empty subject, etc.). Surfaced for telemetry
    but the items themselves are intentionally
    discarded — replaying malformed extractions
    would just recreate the same failure.
    - generic_dropped - count of facts whose *object* was detected as a
    generic placeholder word, triggering a certainty
    downgrade (certain → probable → vague).
    """

    facts: list[AtomicFact] = field(default_factory=list)
    raw_sourceless: list[dict[str, Any]] = field(default_factory=list)
    invalid_dropped: int = 0
    generic_dropped: int = 0


_ATOMIC_FENCE_RE = re.compile(r"^`(?:json)?\s*|\s*`\s*$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

_ATOMIC_FACT_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "atomic_fact_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "certainty": {
                                "type": "string",
                                "enum": ["certain", "probable", "vague"],
                            },
                            "accumulate": {"type": "boolean"},
                            "qualifiers": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["subject", "predicate", "object", "certainty"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    },
}


class AtomicFactExtractor:
    """Convert free text into validated AtomicFact instances.

    Construction takes any object exposing an awaitable chat(messages,
    temperature, max_tokens) returning an object with a content
    string attribute — the same minimal shape used by the rest of the
    memory adapter, so existing LLM client wrappers drop in unchanged.
    """

    def __init__(
        self,
        llm_adapter: Any,
        *,
        max_tokens: int = 1024,
        batch_max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 1,
        prefer_json_mode: bool = True,
    ) -> None:
        if llm_adapter is None:
            raise ValueError("llm_adapter is required")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._llm = llm_adapter
        self._max_tokens = max_tokens
        self._batch_max_tokens = batch_max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._prefer_json_mode = prefer_json_mode

    async def extract(
        self,
        text: str,
        source_anchor: str | None,
    ) -> ExtractionResult:
        """Extract atomic facts from text.

        source_anchor is the caller-supplied provenance handle (chunk
        id, message id, etc.). When it is None or empty/whitespace,
        every extracted item is routed to raw_sourceless instead of
        being assembled into an AtomicFact (the schema would refuse
        an empty anchor anyway, and silently dropping would lose data).
        """
        if not text or not text.strip():
            return ExtractionResult()

        items = await self._call_llm(text)
        if not items:
            return ExtractionResult()

        anchor = (source_anchor or "").strip()
        if not anchor:
            return ExtractionResult(raw_sourceless=list(items))

        facts: list[AtomicFact] = []
        invalid = 0
        generic_dropped = 0
        for item in items:
            fact, was_generic = self._build_fact(item, anchor)
            if fact is None:
                invalid += 1
            else:
                if was_generic:
                    generic_dropped += 1
                facts.append(fact)
        return ExtractionResult(
            facts=facts, invalid_dropped=invalid, generic_dropped=generic_dropped
        )

    async def extract_batch(
        self,
        turns: list[tuple[str, str | None]],
    ) -> list[ExtractionResult]:
        """Extract facts for multiple turns with one LLM call.

        Args:
            turns: Ordered list of (text, source_anchor).

        Returns:
            One ExtractionResult per input element, preserving order.
        """
        if not turns:
            return []

        normalized = self._normalize_batch_turns(turns)
        prompt_parts = [f"<<TURN id={anchor}>>\n{text}" for text, anchor in normalized if text]
        if not prompt_parts:
            return [ExtractionResult() for _ in turns]

        payload = "\n\n".join(prompt_parts)
        parsed = await self._call_llm_batch(payload)
        if parsed is None:
            logger.warning(
                "AtomicFactExtractor batch parse failed; fallback to single-turn extraction for %d turns",
                len(turns),
            )
            return await self._fallback_single_turn(normalized)

        by_anchor = {source_anchor: list(items) for source_anchor, items in parsed if source_anchor}
        return self._results_from_batch_parse(normalized, by_anchor)

    @staticmethod
    def _normalize_batch_turns(turns: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for idx, (text, source_anchor) in enumerate(turns):
            text_norm = (text or "").strip()
            anchor = (source_anchor or "").strip()
            if text_norm and not anchor:
                anchor = f"batch-turn-{idx}"
            normalized.append((text_norm, anchor))
        return normalized

    async def _fallback_single_turn(
        self, normalized: list[tuple[str, str]]
    ) -> list[ExtractionResult]:
        out: list[ExtractionResult] = []
        for text, anchor in normalized:
            if not text:
                out.append(ExtractionResult())
                continue
            out.append(await self.extract(text, anchor))
        return out

    def _results_from_batch_parse(
        self,
        normalized: list[tuple[str, str]],
        by_anchor: dict[str, list[dict[str, Any]]],
    ) -> list[ExtractionResult]:
        out: list[ExtractionResult] = []
        for text, anchor in normalized:
            if not text:
                out.append(ExtractionResult())
                continue
            items = by_anchor.get(anchor, [])
            if not anchor or anchor.startswith("batch-turn-"):
                out.append(ExtractionResult(raw_sourceless=list(items)))
                continue

            facts: list[AtomicFact] = []
            invalid = 0
            generic_dropped = 0
            for item in items:
                fact, was_generic = self._build_fact(item, anchor)
                if fact is None:
                    invalid += 1
                else:
                    if was_generic:
                        generic_dropped += 1
                    facts.append(fact)
            out.append(
                ExtractionResult(
                    facts=facts, invalid_dropped=invalid, generic_dropped=generic_dropped
                )
            )
        return out

    async def _call_llm(self, text: str) -> list[dict[str, Any]]:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            content = await self._request_llm(text, retry=attempt > 0)
            if content is None:
                return []
            items = self._parse_json_array(content)
            if items is not None:
                return items
        return []

    async def _request_llm(self, text: str, *, retry: bool) -> str | None:
        user_content = text
        if retry:
            user_content = (
                "The previous response was invalid JSON. Re-read the original text "
                "and return only a valid JSON array, with no markdown or prose.\n\n"
                f"Original text:\n{text}"
            )
        messages = [
            {"role": "system", "content": _ATOMIC_FACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            response = await self._llm.chat(
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **self._json_kwargs(),
            )
        except TypeError as exc:
            if not self._prefer_json_mode:
                logger.warning("AtomicFactExtractor LLM call failed", exc_info=True)
                return None
            try:
                response = await self._llm.chat(
                    messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except Exception:
                logger.warning("AtomicFactExtractor LLM call failed", exc_info=True)
                return None
            logger.debug("AtomicFactExtractor JSON mode fallback after TypeError: %s", exc)
        except Exception:
            # The extractor's contract is "best-effort, never raise":
            # downstream pipelines must keep flowing even if the LLM is
            # offline. The detailed traceback is logged for ops.
            logger.warning("AtomicFactExtractor LLM call failed", exc_info=True)
            return None

        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return None
        return content

    async def _call_llm_batch(self, text: str) -> list[tuple[str, list[dict[str, Any]]]] | None:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            content = await self._request_llm_batch(text, retry=attempt > 0)
            if content is None:
                return None
            items = self._parse_json_batch(content)
            if items is not None:
                return items
        return None

    async def _request_llm_batch(self, text: str, *, retry: bool) -> str | None:
        user_content = text
        if retry:
            user_content = (
                "The previous response was invalid JSON. Re-read the original batch and return only valid JSON.\n\n"
                f"Original turns:\n{text}"
            )
        messages = [
            {"role": "system", "content": _ATOMIC_FACT_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            response = await self._llm.chat(
                messages,
                temperature=self._temperature,
                max_tokens=self._batch_max_tokens,
                **self._json_kwargs(),
            )
        except TypeError:
            if not self._prefer_json_mode:
                logger.warning("AtomicFactExtractor batch LLM call failed", exc_info=True)
                return None
            try:
                response = await self._llm.chat(
                    messages,
                    temperature=self._temperature,
                    max_tokens=self._batch_max_tokens,
                )
            except Exception:
                logger.warning("AtomicFactExtractor batch LLM call failed", exc_info=True)
                return None
        except Exception:
            logger.warning("AtomicFactExtractor batch LLM call failed", exc_info=True)
            return None

        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return None
        return content

    def _json_kwargs(self) -> dict[str, Any]:
        if not self._prefer_json_mode:
            return {}
        return {"response_format": _ATOMIC_FACT_RESPONSE_FORMAT}

    @staticmethod
    def _parse_json_array(content: str) -> list[dict[str, Any]] | None:
        candidates = _json_candidates(content)
        if not candidates:
            return []
        last_candidate = candidates[-1]
        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
                return _atomic_items_from_decoded(decoded)
            except json.JSONDecodeError:
                continue
        logger.warning("AtomicFactExtractor got non-JSON response: %s", last_candidate[:200])
        return None

    @staticmethod
    def _parse_json_batch(content: str) -> list[tuple[str, list[dict[str, Any]]]] | None:
        candidates = _json_candidates(content)
        if not candidates:
            return []
        last_candidate = candidates[-1]
        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
                return _batch_items_from_decoded(decoded)
            except json.JSONDecodeError:
                continue
        logger.warning("AtomicFactExtractor got non-JSON batch response: %s", last_candidate[:200])
        return None

    @staticmethod
    def _build_fact(item: dict[str, Any], anchor: str) -> tuple[AtomicFact | None, bool]:
        """Assemble one validated AtomicFact or None on schema failure.

        Returns (fact, was_generic) where *was_generic* is True when the
        object field was detected as a generic placeholder and the certainty
        was downgraded accordingly.
        """
        try:
            certainty_raw = str(item.get("certainty", "")).strip().lower()
            certainty = Certainty(certainty_raw)
        except ValueError:
            return None, False

        # Generic-object downgrade: certain → probable, probable → vague.
        was_generic = False
        if _detect_generic_object(item):
            was_generic = True
            if certainty == Certainty.CERTAIN:
                certainty = Certainty.PROBABLE
            elif certainty == Certainty.PROBABLE:
                certainty = Certainty.VAGUE

        qualifiers_raw = item.get("qualifiers")
        qualifiers: dict[str, str] | None = None
        if isinstance(qualifiers_raw, dict):
            try:
                qualifiers = {str(k): str(v) for k, v in qualifiers_raw.items()}
            except (TypeError, ValueError):
                qualifiers = None

        accumulate = bool(item.get("accumulate", False))

        try:
            return AtomicFact(
                subject=str(item.get("subject", "")),
                predicate=str(item.get("predicate", "")),
                object=str(item.get("object", "")),
                certainty=certainty,
                source_anchor=anchor,
                qualifiers=qualifiers,
                accumulate=accumulate,
            ), was_generic
        except ValueError:
            return None, False


def _json_candidates(content: str) -> list[str]:
    cleaned = _ATOMIC_FENCE_RE.sub("", content).strip()
    if not cleaned:
        return []
    candidates = [cleaned]
    span = _extract_json_span(cleaned)
    if span and span != cleaned:
        candidates.append(span)
    for i in range(len(candidates)):
        repaired = _TRAILING_COMMA_RE.sub(r"\1", candidates[i])
        if repaired != candidates[i]:
            candidates.append(repaired)
    return candidates


def _extract_json_span(text: str) -> str | None:
    starts = [pos for pos in (text.find("["), text.find("{")) if pos >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    if end <= start:
        return None
    return text[start : end + 1].strip()


def _atomic_items_from_decoded(decoded: Any) -> list[dict[str, Any]]:
    if isinstance(decoded, dict):
        facts = decoded.get("facts")
        if isinstance(facts, list):
            return [item for item in facts if isinstance(item, dict)]
        return []
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, dict)]


def _batch_items_from_decoded(decoded: Any) -> list[tuple[str, list[dict[str, Any]]]]:
    if isinstance(decoded, dict):
        items = decoded.get("items")
        if not isinstance(items, list):
            return []
        out_dict: list[tuple[str, list[dict[str, Any]]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            source_anchor = str(item.get("source_anchor", "")).strip()
            facts = _atomic_items_from_decoded(item.get("facts", []))
            out_dict.append((source_anchor, facts))
        return out_dict
    if isinstance(decoded, list):
        out_list: list[tuple[str, list[dict[str, Any]]]] = []
        for item in decoded:
            if not isinstance(item, dict):
                continue
            source_anchor = str(item.get("source_anchor", "")).strip()
            facts = _atomic_items_from_decoded(item.get("facts", []))
            out_list.append((source_anchor, facts))
        return out_list
    return []
