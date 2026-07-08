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

from houyi.adapters.memory.timestamp_resolver import (
    extract_observation_date,
    resolve_relative_timestamp,
)
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    ExtractionContext,
    MemoryCandidate,
    MemoryEvent,
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
Extract atomic facts and events from multiple turns. Each turn is prefixed by:
<<TURN id=...>>
The turn content is a JSON object with:
 - text: The conversation turn text.
 - speaker_name: The canonical name of the speaker (use this as the subject of extracted facts/events about the speaker).
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
      ],
      "events": [
        {
          "subject": "...",
          "action": "...",
          "object": "...",
          "timestamp": "...",
          "context": "...",
          "certainty": "certain|probable|vague"
        }
      ],
      "edges": []
    }
  ]
}

STATE vs EVENT DISTINCTION:
Output each extracted semantic unit as either a FACT (state) or an EVENT (action):

FACT (State): When the turn describes a static attribute, preference, or ongoing condition.
  Example: "I love coffee" -> FACT(subject="user", predicate="likes", object="coffee", accumulate=false)
  Example: "My job is designer" -> FACT(subject="user", predicate="job", object="designer")

EVENT (Action): When the turn describes a specific action, occurrence, or transition that
happened at a point in time. MUST include a timestamp.
  Example: "I watched Eternal Sunshine in 2019" -> EVENT(subject="user", action="watched",
    object="Eternal Sunshine", timestamp="2019")
  Example: "I moved to Shanghai last year" -> EVENT(subject="user", action="moved_to",
    object="Shanghai", timestamp="last year")

Rules for EVENT extraction:
1. Every EVENT MUST have a non-empty timestamp. Preserve relative times verbatim ("last week", "a few years ago").
2. The action MUST be a specific verb (watched, adopted, moved_to, purchased, started, lost,
   passed_away, married, traveled_to). Never use generic state verbs (likes, enjoys, has, is).
3. Multiple events by the same subject: output EACH as a separate EVENT. This includes
   SECONDARY or EMBEDDED activities. Invitations, plans, and proposals ("we're going to X",
   "want to join?", "let's do Y next Saturday", "I went to Z") MUST each produce their own
   EVENT even when they are not the main point of the sentence. Never drop a planned/proposed
   activity just because the sentence's primary clause is about a different topic.
4. When a turn describes both a state and an action, output BOTH as separate items.
5. Edges are always an empty array. The system computes edges from EVENT fields.
6. Compound facts (_compound) are ONLY for static enumeration lists. NEVER use _compound
   for action sequences or temporal transitions.

Rules:
1) Keep each fact and event under the correct source_anchor matching the <<TURN id=...>> prefix.
2) If no fact or event is present for a turn, return that source_anchor with empty arrays.
3) Use the person's actual speaker_name if the fact/event is about the speaker.
4) Return only JSON, no markdown or prose.
5) PROPER NOUN & SPECIFIC DETAIL PRESERVATION: Do not generalize, abstract, or drop specific proper nouns, names, brands, models, objects, or locations. For example, "Ferrari 488 GTB" must NOT be simplified to "new car" or "luxury car"; keep "Ferrari 488 GTB" intact in the "object". Preserve specific names of people, places, and distinct assets (e.g. "mansion", "Karlie") rather than replacing them with generic categories like "friend" or "house".
6) OBJECT SPECIFICITY RULE: The "object" field MUST preserve the exact specific term used in the original text. NEVER replace a concrete brand, model, name, or location with a generic category word. Forbidden generic replacements: ride, car, vehicle, place, city, house, friend, person, thing, item, job, pet, animal. WRONG: {"object": "new car"} RIGHT: {"object": "Ferrari 488 GTB"} WRONG: {"object": "ride"} RIGHT: {"object": "new Prius"} WRONG: {"object": "city"} RIGHT: {"object": "San Francisco"}
7) TEMPORAL LITERAL PRESERVATION: When resolving relative time references (e.g. "the week before March 27, 2023", "last week") and computing standard dates for "qualifiers.date" or "qualifiers.since", you MUST ALSO record the exact original relative/literal time string (e.g., "the week before March 27, 2023") under a key "original_time" inside the "qualifiers" object. Do not lose the literal wording of the temporal event.
8) JOINT ENTITY RESOLUTION (WE/US/OUR): If the speaker refers to themselves and their partner/friend/family as "we", "us", or "our" when discussing shared interests, activities, preferences, or states (e.g., "We love making desserts"), you MUST extract symmetrical individual facts for BOTH people. Extract one fact with the speaker's name as subject, and another identical fact with the partner's name as subject. Do NOT output a vague subject like "we". Additionally, when two named participants BOTH engage in or enjoy the same activity (e.g. both watch movies, both make desserts), extract a shared-activity fact: subject = one person, predicate = "shares_interest_with" or "shares_activity_with", object = the other person's name, qualifiers = {"activity": "making desserts"}. This makes shared interests explicitly searchable alongside the individual facts.
9) GOALS/CAREER ASPIRATIONS ALIGNMENT: To ensure reliable recall of future intentions, any career goals, life ambitions, or mid-to-long term plans (e.g., "making a difference away from the court through charity", "getting endorsements", "building a brand") MUST be extracted using the predicate "has_goal" or "career_goal", rather than vague predicates like "wants_to", "likes", or "dislikes".
10) LIFETIME EVENT & TRANSITION PRESERVATION: You MUST explicitly extract active milestone transition actions (e.g. "adopted", "bought", "started_job", "lost_job") and record the exact year/month of occurrence in the qualifiers (e.g., "date: 2020", "date: 2023-03"). Do NOT collapse active transitions into static states (e.g. do not turn "adopted 3 dogs in 2020" into just "owns dogs"). Preserve both the action and its exact date/year.
11) EXHAUSTIVE ENUMERATION: Extract EVERY distinct fact stated in a turn. When a turn names multiple items of the same kind (several books, hobbies, activities, authors, places, foods), emit a SEPARATE fact for EACH item with "accumulate": true. NEVER collapse a list into one representative item, and NEVER drop the "minor" items. If a speaker names five books, output five separate facts — one per book.
12) LIFE EVENTS ARE MANDATORY: Always extract one-time life events even when stated briefly or in passing, AND even when they concern the speaker's own family member or friend (e.g. a parent's or friend's death, a donation, selling/disposing of an item, acquiring a specific asset, losing a job). Trigger keywords include: donated, gave away, disposed of, sold, got rid of, passed away, died, lost (a person). Use the named person or the speaker's relative as subject (e.g. subject "Deborah" with predicate "lost_family_member", object "father"). Suggested predicates: lost_family_member, lost_friend, donated, sold, bought, acquired, lost_job. Preserve the proper noun (the named friend, the specific asset) and the date. A relative's or friend's death reported by the speaker is part of the speaker's own life and MUST be extracted — it is NOT "unrelated third-party information".
13) DESCRIPTIVE FIDELITY: Reproduce the speaker's exact descriptive words. NEVER replace a term with its opposite or a near-synonym (do not turn "sunrise" into "sunset", "old" into "new", "donated" into "owns"). Keep stated time expressions (e.g. "last year") and ALSO run them through DATE HANDLING to record a qualifier date.
14) SPEAKER TEXT OVER IMAGE CAPTION: When a speaker explicitly names or describes something that contradicts an image caption in the same turn (e.g. speaker says "sunrise" but image caption says "sunset"), the fact MUST use the speaker's own stated word. The speaker's explicit text is primary evidence; image captions are secondary context only. However, when the image caption (or "blip_caption" / "Image caption:") provides non-contradictory, unique descriptive facts about the scene, assets, or people present (such as "a photo of a group of people standing in front of a car" or specific background details), you MUST extract those unique facts as valid memories (e.g., subject "Dave's shop", predicate "has_attribute", object "group of people standing in front of it" or "employs a group of people", with "certainty": "certain").
15) SPECIFIC NAME OVERRIDE: When a speaker uses a generic term (e.g. "my ride", "new car", "the house") in one turn, but the SAME speaker previously or later mentions a specific proper name/model for the same referent (e.g. "Ferrari 488 GTB", "Prius", "mansion"), the object MUST use the specific name rather than the generic term. Cross-turn consistency requires the specific name to override generic references within the same batch.

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

Input:
<<TURN id=conv-100:D1:1>>
{"observation_date": "2023-06-15", "system_date": "2024-01-15", "text": "I switched from drinking cow milk to oat milk last month because I developed lactose intolerance.", "speaker_name": "Evan"}

<<TURN id=conv-100:D2:5>>
{"observation_date": "2023-06-20", "system_date": "2024-01-15", "text": "Joanna works as a Designer. She knows Evan because they went to college together.", "speaker_name": "Sam"}

Output:
{
  "items": [
    {
      "source_anchor": "conv-100:D1:1",
      "facts": [
        {"subject": "Evan", "predicate": "dislikes", "object": "cow milk", "certainty": "certain"},
        {"subject": "Evan", "predicate": "drinks", "object": "oat milk", "certainty": "certain", "qualifiers": {"since": "2023-05", "original_time": "last month"}},
        {"subject": "Evan", "predicate": "has_health_issue", "object": "lactose intolerance", "certainty": "certain", "qualifiers": {"since": "2023-05", "original_time": "last month"}},
        {
          "subject": "Evan",
          "predicate": "_compound",
          "object": "Evan switched from cow milk to oat milk in May 2023 because of developing lactose intolerance.",
          "certainty": "certain",
          "qualifiers": {"compound_type": "causal_transition", "date": "2023-05", "original_time": "last month"}
        }
      ]
    },
    {
      "source_anchor": "conv-100:D2:5",
      "facts": [
        {"subject": "Joanna", "predicate": "job", "object": "Designer", "certainty": "certain"},
        {"subject": "Joanna", "predicate": "knows", "object": "Evan", "certainty": "certain", "qualifiers": {"reason": "went to college together"}}
      ],
      "events": [],
      "edges": []
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
 - no explicit time on a STATIC STATE (likes, owns, lives in) -> the state holds as of observation_date; record qualifier date: observation_date
 - just / recently / the other day / vague recency on a PAST ACTION or EVENT (e.g. "I just went to X", "I recently saw Y") -> the action happened at an UNKNOWN point BEFORE observation_date. observation_date is only the REPORT/UPPER-BOUND date, NOT the precise event date. Record qualifier date: observation_date (the upper bound) AND qualifier date_certainty: "approximate" AND qualifier original_time: the verbatim cue (e.g. "recently"). Never present observation_date as the exact event date.

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

COMPOUND FACTS EXTRACTION (CRITICAL):
 - When a user's message contains complex relational, causal, or temporal transitions (e.g. "switched/stopped/replaced", future plans/bookings, or multiple connected facts like "switched from almond to oat milk because of sensitivity last month"), you MUST extract one rich COMPOUND fact in addition to individual atomic facts.
 - A compound fact MUST use the predicate "_compound", the entity as subject, and the complete, contextually-rich text statement (with relative time words like "next month" or "last week" resolved to absolute dates/months based on observation_date) as "object".
 - The "qualifiers" for a compound fact MUST include "compound_type" (e.g., "causal_transition", "scheduled_plan") and computed absolute "date" or "original_time" (e.g., "July 2023").

RELATION EDGE EXTRACTION:
 - When a turn expresses explicit or strong implicit relationships between extracted facts or entities (e.g., person A works with person B, or person A knows person B, or event A causes state B), you MUST extract a list of directed relation edges under the "edges" array.
   - Use standard entity names (e.g. "Caroline", "Joanna") or attribute paths (e.g. "Caroline.job") as symbolic source_unit_id and target_unit_id.
   - Set source_type and target_type to "state" (for entity-attribute states) or "fact" (if referencing a specific content statement).
   - Use one of these exact relation values: "derived_from", "source_of", "same_as", "updates", "replaces", "invalidates", "supports", "contradicts", "precedes", "causes", "related_to", "strategy_for", "readable_by".
"""

_ATOMIC_FACT_SYSTEM_PROMPT = """You are an information extraction engine.
Extract discrete factual claims and explicit/implicit relation edges between them from user messages.

OUTPUT SCHEMA (JSON object):
{
  "facts": [
    {
      "subject": "the entity the fact is about. Use actual name if known, otherwise 'user'",
      "subject_type": "person|organization|vehicle|location|activity|item|concept|unknown",
      "predicate": "snake_case attribute",
      "object": "value as short string",
      "object_type": "person|organization|vehicle|location|activity|item|concept|unknown",
      "certainty": "certain|probable|vague",
      "qualifiers": {"since": "...", "date": "..."}
    }
  ],
  "events": [
    {
      "subject": "who performed the action",
      "action": "a specific verb (watched, adopted, moved_to, purchased, started, lost, passed_away, married, traveled_to)",
      "object": "what the action targets",
      "timestamp": "when it happened (year, relative time, date string)",
      "context": "supplementary details",
      "certainty": "certain|probable|vague"
    }
  ],
  "edges": []
}

STATE vs EVENT DISTINCTION:
Output each extracted semantic unit as either a FACT (state) or an EVENT (action):

FACT (State): When the turn describes a static attribute, preference, or ongoing condition.
  Example: "I love coffee" -> FACT(subject="user", predicate="likes", object="coffee", accumulate=false)
  Example: "My job is designer" -> FACT(subject="user", predicate="job", object="designer")

EVENT (Action): When the turn describes a specific action, occurrence, or transition that
happened at a point in time. MUST include a timestamp.
  Example: "I watched Eternal Sunshine in 2019" -> EVENT(subject="user", action="watched",
    object="Eternal Sunshine", timestamp="2019")
  Example: "I moved to Shanghai last year" -> EVENT(subject="user", action="moved_to",
    object="Shanghai", timestamp="last year")
  Example: "I adopted a dog named Pepper last month" -> EVENT(subject="user", action="adopted",
    object="dog named Pepper", timestamp="last month")

Rules for EVENT extraction:
1. Every EVENT MUST have a non-empty timestamp field. If the turn gives a relative time
   ("last week", "a few years ago", "yesterday"), preserve it verbatim in timestamp.
2. The action field MUST be a specific verb (watched, adopted, moved_to, purchased,
   started, lost, passed_away, married, divorced, enrolled, quit, traveled_to).
   Never use generic state verbs (likes, enjoys, has, is, wants).
3. When a turn mentions multiple events by the same subject, output EACH as a separate EVENT.
   This includes SECONDARY or EMBEDDED activities: invitations, plans, and proposals
   ("we're going to X", "want to join?", "let's do Y next Saturday", "I went to Z") MUST each
   produce their own EVENT even when they are not the main point of the sentence. Never drop a
   planned/proposed activity just because the sentence's primary clause is about a different topic.
4. When a turn describes both a state and an action, output BOTH. "I started painting
   because I enjoy art" -> FACT(subject="user", predicate="enjoys", object="art") AND
   EVENT(subject="user", action="started", object="painting", timestamp="...").
5. Edges are always returned as an empty array. The system computes edges from
   EVENT fields deterministically; do not generate edges yourself.

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
 - no explicit time on a STATIC STATE (likes, owns, lives in) → the state holds as of observation_date; record as qualifier "date": observation_date
 - "just" / "recently" / "the other day" / vague recency on a PAST ACTION or EVENT (e.g. "I just went to X", "I recently saw Y") → the action happened at an UNKNOWN point BEFORE observation_date. observation_date is only the REPORT/UPPER-BOUND date, NOT the precise event date. Record qualifier "date": observation_date (the upper bound) AND qualifier "date_certainty": "approximate" AND qualifier "original_time": the verbatim cue (e.g. "recently"). Never present observation_date as the exact event date.

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
 - Shared activities: when two named participants both enjoy or engage in the same activity (e.g. both watch movies, both make desserts), extract a fact with subject = one person, predicate = "shares_interest_with", object = the other person's name, qualifiers = {"activity": "<the activity>"}
 - Facts about projects, tasks, deadlines

DO NOT EXTRACT:
 - Greetings and small talk ("Hello", "How are you?", "Good morning")
 - Questions asked by the user
 - Opinions about external topics not related to the speaker
 - Information about unrelated public figures or strangers the speaker has no connection to
 - General knowledge facts ("The sky is blue")

ALWAYS EXTRACT (these are part of the speaker's own life, never skip them):
 - Life events about the speaker's family or friends (e.g. a parent's or friend's death, marriage, birth) — use the named person or relative as subject (e.g. "Deborah lost_family_member father", "Deborah lost_friend Karlie"). Trigger keywords: passed away, died, lost (a person), donated, gave away, disposed of, sold, got rid of.
 - One-time transactions: donating, selling, buying, or acquiring a specific item — preserve the specific object and the date. Trigger keywords: donated, gave away, disposed of, sold, got rid of.
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

SPEAKER TEXT OVER IMAGE CAPTION:
- When a speaker explicitly names or describes something that contradicts an image caption in the same turn (e.g. speaker says "sunrise" but image caption says "sunset"), the fact MUST use the speaker's own stated word. The speaker's explicit text is primary evidence; image captions are secondary context only.

SPECIFIC NAME OVERRIDE:
- When a speaker uses a generic term (e.g. "my ride", "new car", "the house") in one turn, but the SAME speaker previously or later mentions a specific proper name/model for the same referent (e.g. "Ferrari 488 GTB", "Prius", "mansion"), the object MUST use the specific name rather than the generic term. Cross-turn consistency requires the specific name to override generic references within the same batch.

TEMPORAL LITERAL PRESERVATION:
- When resolving relative time references (e.g. "the week before March 27, 2023", "last week") and computing standard dates for "qualifiers.date" or "qualifiers.since", you MUST ALSO record the exact original relative/literal time string (e.g., "the week before March 27, 2023") under a key "original_time" inside the "qualifiers" object. Do not lose the literal wording of the temporal event.

COMPOUND FACTS EXTRACTION (CRITICAL):
- When a user's message contains complex relational, causal, or temporal transitions (e.g. "switched/stopped/replaced", future plans/bookings, or multiple connected facts like "switched from almond to oat milk because of sensitivity last month"), you MUST extract one rich COMPOUND fact in addition to individual atomic facts.
- A compound fact MUST use the predicate "_compound", the entity as subject, and the complete, contextually-rich text statement (with relative time words like "next month" or "last week" resolved to absolute dates/months based on observation_date) as "object".
- The "qualifiers" for a compound fact MUST include "compound_type" (e.g., "causal_transition", "scheduled_plan") and computed absolute "date" or "original_time" (e.g., "July 2023").

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

Example 18 - Compound facts with event:
Input: {"observation_date": "2023-04-10", "system_date": "2024-01-15", "text": "I switched from drinking cow milk to oat milk last month because of my lactose intolerance.", "speaker_name": "Evan"}
Output:
{
  "facts": [
    {"subject": "Evan", "predicate": "dislikes", "object": "cow milk", "certainty": "certain"},
    {"subject": "Evan", "predicate": "drinks", "object": "oat milk", "certainty": "certain", "qualifiers": {"since": "2023-03", "original_time": "last month"}},
    {"subject": "Evan", "predicate": "has_health_issue", "object": "lactose intolerance", "certainty": "certain", "qualifiers": {"since": "2023-03", "original_time": "last month"}},
    {
      "subject": "Evan",
      "predicate": "_compound",
      "object": "Evan switched from cow milk to oat milk in March 2023 because of lactose intolerance.",
      "certainty": "certain",
      "qualifiers": {"compound_type": "causal_transition", "date": "2023-03", "original_time": "last month"}
    }
  ],
  "events": [
    {"subject": "Evan", "action": "switched_to", "object": "oat milk", "timestamp": "last month", "context": "from cow milk because of lactose intolerance", "certainty": "certain"}
  ],
  "edges": []
}

---

INPUT FORMAT:
{"observation_date": "YYYY-MM-DD", "system_date": "YYYY-MM-DD", "text": "user message text", "speaker_name": "OptionalName"}

Note: speaker_name is optional. When provided, use it as the subject for facts about the speaker (instead of "user").

OUTPUT FORMAT:
JSON object with "facts" (array of facts), "events" (array of events), and "edges" (always empty array). Return {"facts": [], "events": [], "edges": []} if no extractable facts/events.
"""


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a single AtomicFact extraction call.

    - facts - successfully validated AtomicFact list.
    - events - successfully validated MemoryEvent list (temporal occurrences).
    - edges - successfully parsed semantic MemoryEdge candidate relationships.
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
    """

    facts: list[AtomicFact] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    raw_sourceless: list[dict[str, Any]] = field(default_factory=list)
    invalid_dropped: int = 0


_ATOMIC_FENCE_RE = re.compile(r"^`(?:json)?\s*|\s*`\s*$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

_ATOMIC_FACT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
        "certainty": {"type": "string", "enum": ["certain", "probable", "vague"]},
        "accumulate": {"type": "boolean"},
        "qualifiers": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["subject", "predicate", "object", "certainty"],
    "additionalProperties": False,
}

_ATOMIC_EVENT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "action": {"type": "string"},
        "object": {"type": "string"},
        "timestamp": {"type": "string"},
        "context": {"type": "string"},
        "certainty": {"type": "string", "enum": ["certain", "probable", "vague"]},
    },
    "required": ["subject", "action", "object", "timestamp"],
    "additionalProperties": False,
}

_ATOMIC_EDGE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_unit_id": {"type": "string"},
        "target_unit_id": {"type": "string"},
        "relation": {"type": "string"},
        "source_type": {"type": "string", "enum": ["state", "fact", "event"]},
        "target_type": {"type": "string", "enum": ["state", "fact", "event"]},
    },
    "required": ["source_unit_id", "target_unit_id", "relation"],
    "additionalProperties": False,
}

_ATOMIC_FACT_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "atomic_fact_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "facts": {"type": "array", "items": _ATOMIC_FACT_ITEM_SCHEMA},
                "events": {"type": "array", "items": _ATOMIC_EVENT_ITEM_SCHEMA},
                "edges": {"type": "array", "items": _ATOMIC_EDGE_ITEM_SCHEMA},
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    },
}


def _atomic_fact_batch_response_format(n_turns: int) -> dict[str, Any]:
    """JSON schema constraining batch extraction output.

    minItems forces the LLM to return one item per input turn -- a turn with
    no facts comes back as an item with empty arrays rather than being
    silently omitted -- so long-batch attention loss can no longer drop turns
    from the response. Each item requires a source_anchor so facts stay
    grouped to their turn. The per-anchor recovery in extract_batch remains
    as a safety net for providers that do not enforce minItems.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "atomic_fact_batch_extraction",
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": n_turns,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_anchor": {"type": "string"},
                                "facts": {"type": "array", "items": _ATOMIC_FACT_ITEM_SCHEMA},
                                "events": {"type": "array", "items": _ATOMIC_EVENT_ITEM_SCHEMA},
                                "edges": {"type": "array", "items": _ATOMIC_EDGE_ITEM_SCHEMA},
                            },
                            "required": ["source_anchor", "facts", "events", "edges"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


# Structured JSON extraction is pattern matching against a schema, not
# chain-of-thought reasoning. On providers whose thinking mode is on by
# default, reasoning_content can consume the entire max_tokens budget and
# leave message.content empty with finish_reason "length". Every batch and
# single-turn extract call explicitly disables thinking so the budget goes
# to the JSON payload.
_EXTRACT_THINKING_OFF: dict[str, Any] = {"enable_thinking": False}


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
        *,
        namespace: str = "default",
    ) -> ExtractionResult:
        """Extract atomic facts from text.

        source_anchor is the caller-supplied provenance handle (chunk
        id, message id, etc.). When it is None or empty/whitespace,
        every extracted item is routed to raw_sourceless instead of
        being assembled into an AtomicFact (the schema would refuse
        an empty anchor anyway, and silently dropping would lose data).

        namespace is applied to extracted events so the EventRetriever
        (which queries by the same namespace) can find them -- mirrors
        the batch path. Without it, single-turn recovery would write
        events under the default namespace and the retriever would miss
        them.
        """
        if not text or not text.strip():
            return ExtractionResult()

        facts_raw, edges_raw, events_raw = await self._call_llm(text)
        if not facts_raw and not events_raw:
            return ExtractionResult()

        anchor = (source_anchor or "").strip()
        if not anchor:
            return ExtractionResult(raw_sourceless=list(facts_raw))

        facts: list[AtomicFact] = []
        events: list[Any] = []
        invalid = 0
        for item in facts_raw:
            fact = self._build_fact(item, anchor)
            if fact is None:
                invalid += 1
            else:
                facts.append(fact)
        # Build events from raw event dicts -- mirrors the batch parse path.
        # Without this loop the single-turn path silently dropped every event
        # (events_raw was read but never assembled), so any anchor recovered
        # via single-turn re-extraction lost its time-anchored events.
        for ev_item in events_raw:
            event = self._build_event(ev_item, anchor, namespace, extract_observation_date(text))
            if event is None:
                invalid += 1
            else:
                events.append(event)
        facts = self._restore_generic_objects(facts)
        return ExtractionResult(
            facts=facts, events=events, edges=list(edges_raw), invalid_dropped=invalid
        )

    async def extract_batch(
        self,
        turns: list[tuple[str, str | None]],
        namespace: str = "default",
    ) -> list[ExtractionResult]:
        """Extract facts for multiple turns with one LLM call.

        Args:
            turns: Ordered list of (text, source_anchor).
            namespace: Namespace applied to extracted events. Events are
                written alongside entity_state facts and must share the query
                namespace, otherwise the EventRetriever (which queries by the
                same namespace) never sees them. Default "default" preserves
                backward compatibility for callers without a namespace.

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
        parsed = await self._call_llm_batch(payload, n_turns=len(prompt_parts))
        if parsed is None:
            logger.warning(
                "AtomicFactExtractor batch parse failed; fallback to single-turn extraction for %d turns",
                len(turns),
            )
            return await self._fallback_single_turn(normalized)

        by_anchor = {
            source_anchor: (list(facts), list(edges), list(events))
            for source_anchor, facts, edges, events in parsed
            if source_anchor
        }
        # Guarantee per-anchor coverage. The batch LLM response is unreliable
        # at the per-turn level in TWO ways, both caused by long-batch
        # attention loss: (1) it silently OMITS a turn entirely, and (2) it
        # returns the turn PRESENT BUT EMPTY (an item with facts=[] and
        # events=[]). Empirically the empty case is NOT "the model deliberately
        # found nothing" -- the same turn extracted singly (no batch attention
        # loss) yields its salient facts, while the batch drops them. Treating
        # a batch empty as authoritative makes a gold evidence turn score 0
        # facts forever: it vanishes from the DB and recall has nothing to
        # surface (root-caused via debug-trace: gold evidence turns came back
        # batch-empty and were never recovered without this re-extraction). So
        # we re-extract via the single-turn path BOTH absent anchors AND
        # present-but-empty ones. Single-turn re-extraction of a genuinely
        # empty turn simply returns empty again -- a wasted call, never a
        # correctness loss.
        single_results: dict[str, ExtractionResult] = {}
        dropped = 0
        dropped_raw_empty = 0
        dropped_build_invalid = 0
        for text, anchor in normalized:
            if not text or not anchor or anchor.startswith("batch-turn-"):
                continue
            covered = by_anchor.get(anchor)
            # Re-extract when the batch omitted the anchor OR returned it with
            # no item that survives schema validation. The earlier check only
            # tested raw-list emptiness, which missed the case where the batch
            # returns a non-empty but schema-invalid item list (e.g. an
            # unrecognized certainty value) that _build_fact/_build_event drop
            # entirely -- the anchor then stored empty with no recovery, even
            # though the single-turn path would have produced valid facts.
            # Trial-building here reuses the same pure static builders that
            # _results_from_batch_parse applies downstream, so the validity
            # check is exact rather than a raw-emptiness heuristic. The second
            # build downstream is cheap (no LLM, pure compute).
            if covered is None:
                needs_reextract = True
                reason = "raw_empty"
            else:
                raw_facts, _raw_edges, raw_events = covered
                obs_date = extract_observation_date(text)
                has_valid = any(
                    self._build_fact(item, anchor) is not None for item in raw_facts
                ) or any(
                    self._build_event(item, anchor, namespace, obs_date) is not None
                    for item in raw_events
                )
                needs_reextract = not has_valid
                reason = "build_invalid"
            if needs_reextract:
                dropped += 1
                if reason == "raw_empty":
                    dropped_raw_empty += 1
                else:
                    dropped_build_invalid += 1
                single_results[anchor] = await self.extract(text, anchor, namespace=namespace)
        if dropped:
            logger.info(
                "AtomicFactExtractor batch dropped/emptied %d/%d anchors; "
                "re-extracted singly (raw_empty=%d, build_invalid=%d)",
                dropped,
                len(normalized),
                dropped_raw_empty,
                dropped_build_invalid,
            )
        return self._results_from_batch_parse(
            normalized, by_anchor, single_results, namespace=namespace
        )

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
        by_anchor: dict[
            str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
        ],
        single_results: dict[str, ExtractionResult] | None = None,
        namespace: str = "default",
    ) -> list[ExtractionResult]:
        single_results = single_results or {}
        out: list[ExtractionResult] = []
        for text, anchor in normalized:
            if not text:
                out.append(ExtractionResult())
                continue
            # Anchors the batch dropped were re-extracted singly in extract_batch;
            # use that result directly instead of the empty by_anchor fallback.
            if anchor in single_results:
                out.append(single_results[anchor])
                continue
            items, raw_edges, events_raw = by_anchor.get(anchor, ([], [], []))
            if not anchor or anchor.startswith("batch-turn-"):
                out.append(ExtractionResult(raw_sourceless=list(items)))
                continue

            facts: list[AtomicFact] = []
            events: list[Any] = []
            invalid = 0
            for item in items:
                fact = self._build_fact(item, anchor)
                if fact is None:
                    invalid += 1
                else:
                    facts.append(fact)
            # Build events from raw event dicts
            for ev_item in events_raw:
                event = self._build_event(
                    ev_item, anchor, namespace, extract_observation_date(text)
                )
                if event is None:
                    invalid += 1
                else:
                    events.append(event)
            # Apply post-extraction restoration
            facts = self._restore_generic_objects(facts)
            out.append(
                ExtractionResult(
                    facts=facts, events=events, edges=list(raw_edges), invalid_dropped=invalid
                )
            )
        return out

    async def _call_llm(
        self, text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            content = await self._request_llm(text, retry=attempt > 0)
            if content is None:
                # _request_llm returns None on exception/timeout (it catches
                # all errors internally). Previously this short-circuited to
                # empty with zero retries, which permanently lost the anchor
                # when this single-turn call is the last-resort fallback after
                # a batch omission. Continue the retry loop instead.
                if attempt < attempts - 1:
                    logger.warning(
                        "AtomicFactExtractor single-turn LLM call returned None "
                        "(exception/timeout), retrying (attempt %d/%d)",
                        attempt + 1,
                        attempts,
                    )
                continue
            res = self._parse_json_array(content)
            if res is not None:
                return res
        return [], [], []

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
                **_EXTRACT_THINKING_OFF,
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

    async def _call_llm_batch(
        self, text: str, *, n_turns: int
    ) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] | None:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            content = await self._request_llm_batch(text, n_turns=n_turns, retry=attempt > 0)
            if content is None:
                # Same zero-retry gap as _call_llm: _request_llm_batch returns
                # None on exception/timeout. Continue the retry loop instead
                # of short-circuiting, so a transient network failure does not
                # lose the entire batch.
                if attempt < attempts - 1:
                    logger.warning(
                        "AtomicFactExtractor batch LLM call returned None "
                        "(exception/timeout), retrying (attempt %d/%d)",
                        attempt + 1,
                        attempts,
                    )
                continue
            items = self._parse_json_batch(content)
            if items is not None:
                return items
        return None

    async def _request_llm_batch(self, text: str, *, n_turns: int, retry: bool) -> str | None:
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
            # Pass a batch-specific json_schema (items[] with minItems = n_turns
            # and required source_anchor per item) so the LLM is constrained to
            # return one item per input turn instead of silently dropping turns
            # to long-batch attention loss. The single-turn schema cannot be
            # reused here: it has no items[]/source_anchor wrapper, and passing
            # it would flatten the output and drop every fact. The per-anchor
            # recovery in extract_batch is the safety net for providers that do
            # not enforce minItems.
            response = await self._llm.chat(
                messages,
                temperature=self._temperature,
                max_tokens=self._batch_max_tokens,
                **_EXTRACT_THINKING_OFF,
                **self._batch_json_kwargs(n_turns),
            )
        except TypeError as exc:
            if not self._prefer_json_mode:
                logger.warning("AtomicFactExtractor batch LLM call failed", exc_info=True)
                return None
            # Adapter rejected response_format; retry unstructured. The batch
            # prompt still requests JSON, and extract_batch recovers dropped
            # anchors singly.
            try:
                response = await self._llm.chat(
                    messages,
                    temperature=self._temperature,
                    max_tokens=self._batch_max_tokens,
                )
            except Exception:
                logger.warning("AtomicFactExtractor batch LLM call failed", exc_info=True)
                return None
            logger.debug("AtomicFactExtractor batch JSON mode fallback after TypeError: %s", exc)
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

    def _batch_json_kwargs(self, n_turns: int) -> dict[str, Any]:
        if not self._prefer_json_mode:
            return {}
        return {"response_format": _atomic_fact_batch_response_format(n_turns)}

    @staticmethod
    def _parse_json_array(
        content: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
        candidates = _json_candidates(content)
        if not candidates:
            return [], [], []
        last_candidate = candidates[-1]
        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
                if isinstance(decoded, dict):
                    facts = decoded.get("facts")
                    edges = decoded.get("edges")
                    events = decoded.get("events")
                    facts_list = (
                        [item for item in facts if isinstance(item, dict)]
                        if isinstance(facts, list)
                        else []
                    )
                    edges_list = (
                        [item for item in edges if isinstance(item, dict)]
                        if isinstance(edges, list)
                        else []
                    )
                    events_list = (
                        [item for item in events if isinstance(item, dict)]
                        if isinstance(events, list)
                        else []
                    )
                    return facts_list, edges_list, events_list
                elif isinstance(decoded, list):
                    # Fallback if it returned a flat array of facts
                    facts_list = [item for item in decoded if isinstance(item, dict)]
                    return facts_list, [], []
            except json.JSONDecodeError:
                continue
        logger.warning("AtomicFactExtractor got non-JSON response: %s", last_candidate[:200])
        return None

    @staticmethod
    def _parse_json_batch(
        content: str,
    ) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] | None:
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
    def _build_fact(item: dict[str, Any], anchor: str) -> AtomicFact | None:
        """Assemble one validated AtomicFact or None on schema failure."""
        try:
            certainty_raw = str(item.get("certainty", "")).strip().lower()
            certainty = Certainty(certainty_raw)
        except ValueError:
            return None

        qualifiers_raw = item.get("qualifiers")
        qualifiers: dict[str, str] | None = None
        if isinstance(qualifiers_raw, dict):
            try:
                qualifiers = {str(k): str(v) for k, v in qualifiers_raw.items()}
            except (TypeError, ValueError):
                qualifiers = None

        subj_type = item.get("subject_type")
        obj_type = item.get("object_type")
        if subj_type or obj_type:
            if qualifiers is None:
                qualifiers = {}
            if subj_type and "subject_type" not in qualifiers:
                qualifiers["subject_type"] = str(subj_type).strip().lower()
            if obj_type and "object_type" not in qualifiers:
                qualifiers["object_type"] = str(obj_type).strip().lower()

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
            )
        except ValueError:
            return None

    @staticmethod
    def _build_event(
        item: dict[str, Any],
        anchor: str,
        namespace: str,
        observation_date: str | None = None,
    ) -> MemoryEvent | None:
        """Assemble one validated MemoryEvent or None on schema failure."""

        subject = str(item.get("subject", "")).strip()
        action = str(item.get("action", "")).strip()
        obj = str(item.get("object", "")).strip()
        timestamp = str(item.get("timestamp", "")).strip()
        # Resolve relative times (e.g. "around 3 years ago") to an absolute
        # value anchored on the per-turn observation_date. The LLM is
        # non-deterministic here (the prompt both tells it to preserve the
        # phrase verbatim and to resolve it), so a relative timestamp would
        # otherwise be stored verbatim and lose its reference date forever.
        timestamp = resolve_relative_timestamp(timestamp, observation_date)
        context = str(item.get("context", "")).strip()

        if not subject or not action or not obj or not timestamp:
            return None

        try:
            certainty_raw = str(item.get("certainty", "")).strip().lower()
            certainty = Certainty(certainty_raw)
        except ValueError:
            certainty = Certainty.CERTAIN

        qualifiers_raw = item.get("qualifiers")
        qualifiers: dict[str, str] | None = None
        if isinstance(qualifiers_raw, dict):
            try:
                qualifiers = {str(k): str(v) for k, v in qualifiers_raw.items()}
            except (TypeError, ValueError):
                qualifiers = None

        try:
            return MemoryEvent(
                namespace=namespace,
                subject=subject,
                action=action,
                object=obj,
                timestamp=timestamp,
                context=context,
                certainty=certainty,
                source_anchor=anchor,
                qualifiers=qualifiers,
            )
        except ValueError:
            return None

    @staticmethod
    def _is_generic_object(val: str, forbidden: set[str]) -> bool:
        v_low = val.lower().strip()
        if v_low in forbidden:
            return True
        return bool(set(v_low.split()).intersection(forbidden))

    @staticmethod
    def _build_type_to_specifics(facts: list[AtomicFact], forbidden: set[str]) -> dict[str, str]:
        type_to_specifics: dict[str, str] = {}
        for f in facts:
            obj_val = f.object.strip()
            if obj_val and not AtomicFactExtractor._is_generic_object(obj_val, forbidden):
                obj_type = f.qualifiers.get("object_type") if f.qualifiers else None
                if obj_type:
                    type_to_specifics[obj_type.lower().strip()] = obj_val

            subj_val = f.subject.strip()
            if (
                subj_val
                and not AtomicFactExtractor._is_generic_object(subj_val, forbidden)
                and subj_val.lower() != "user"
            ):
                subj_type = f.qualifiers.get("subject_type") if f.qualifiers else None
                if subj_type:
                    type_to_specifics[subj_type.lower().strip()] = subj_val
        return type_to_specifics

    @staticmethod
    def _restore_generic_objects(facts: list[AtomicFact]) -> list[AtomicFact]:
        """Look for generic/forbidden words in fact objects and restore them.

        If a fact has a generic object (e.g. "new ride", "city"), we look at
        other facts in the same turn/list. If there is a fact of matching type
        with a specific object, we substitute it. If no substitution can be done,
        we downgrade the certainty to PROBABLE.
        """
        forbidden_generics = {
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
        }

        type_to_specifics = AtomicFactExtractor._build_type_to_specifics(facts, forbidden_generics)
        restored_facts: list[AtomicFact] = []
        for f in facts:
            restored_obj = f.object
            restored_cert = f.certainty

            obj_type = f.qualifiers.get("object_type") if f.qualifiers else None
            if AtomicFactExtractor._is_generic_object(f.object, forbidden_generics):
                matched = False
                if obj_type:
                    specific_val = type_to_specifics.get(obj_type.lower().strip())
                    if specific_val:
                        restored_obj = specific_val
                        matched = True

                if not matched and restored_cert == Certainty.CERTAIN:
                    restored_cert = Certainty.PROBABLE

            if restored_obj != f.object or restored_cert != f.certainty:
                restored_facts.append(
                    AtomicFact(
                        subject=f.subject,
                        predicate=f.predicate,
                        object=restored_obj,
                        certainty=restored_cert,
                        source_anchor=f.source_anchor,
                        qualifiers=f.qualifiers,
                        accumulate=f.accumulate,
                    )
                )
            else:
                restored_facts.append(f)

        return restored_facts


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


def _event_items_from_decoded(decoded: Any) -> list[dict[str, Any]]:
    if isinstance(decoded, list):
        return [item for item in decoded if isinstance(item, dict)]
    return []


def _batch_items_from_decoded(
    decoded: Any,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    if isinstance(decoded, dict):
        items = decoded.get("items")
        if not isinstance(items, list):
            return []
        out_dict: list[
            tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
        ] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            source_anchor = str(item.get("source_anchor", "")).strip()
            facts = _atomic_items_from_decoded(item.get("facts", []))
            edges = [e for e in item.get("edges", []) if isinstance(e, dict)]
            events = _event_items_from_decoded(item.get("events", []))
            out_dict.append((source_anchor, facts, edges, events))
        return out_dict
    if isinstance(decoded, list):
        out_list: list[
            tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
        ] = []
        for item in decoded:
            if not isinstance(item, dict):
                continue
            source_anchor = str(item.get("source_anchor", "")).strip()
            facts = _atomic_items_from_decoded(item.get("facts", []))
            edges = [e for e in item.get("edges", []) if isinstance(e, dict)]
            events = _event_items_from_decoded(item.get("events", []))
            out_list.append((source_anchor, facts, edges, events))
        return out_list
    return []
