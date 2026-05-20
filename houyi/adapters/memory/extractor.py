"""Memory extractors.

Two extractor flavours co-exist in this module:

1. MemoryCandidateExtractor — heuristic + LLM extractor that
 produces MemoryCandidate objects (loose payload + scope + type).
 Used by the legacy ingestion path that classifies/dedup/promotes
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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from houyi.adapters.memory.builder import MemoryCandidateBuilder
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    ExtractionContext,
    MemoryBuildInput,
    MemoryBuildItem,
    MemoryCandidate,
    MemoryRecord,
    MemoryScope,
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

_TYPE_MAP: dict[str, MemoryType] = {
    "fact": MemoryType.FACT,
    "preference": MemoryType.PREFERENCE,
    "constraint": MemoryType.CONSTRAINT,
    "profile": MemoryType.PROFILE,
}


class MemoryCandidateExtractor:
    """Extracts memory candidates from conversation messages.

    When an llm_adapter is provided, uses LLM-based extraction
    (language-agnostic, semantic understanding). Falls back to English-only
    rule-based patterns when no LLM is available.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        llm_adapter: Any | None = None,
    ):
        self._min_confidence = min_confidence
        self._llm = llm_adapter
        self._builder = MemoryCandidateBuilder(
            min_confidence=min_confidence,
            llm_adapter=llm_adapter,
        )

    async def extract(
        self,
        messages: list[dict],
        existing_memories: list[MemoryRecord] | None = None,
        context: ExtractionContext | None = None,
    ) -> list[MemoryCandidate]:
        """Extract memory candidates from a message sequence.

        Uses LLM extraction when available, otherwise falls back to
        rule-based patterns (English only).

        Args:
        messages: List of message dicts with 'role' and 'content' keys.
        existing_memories: Current memories for dedup hint.
        context: Extraction context metadata.
        """
        ctx = context or ExtractionContext()
        _ = existing_memories
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
        return await self._builder.build(memory_input, ctx)

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------

    async def _extract_via_llm(
        self,
        messages: list[dict],
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        """Send messages to LLM for structured memory extraction."""
        user_texts = [
            m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")
        ]
        if not user_texts:
            return []

        user_block = "\n".join(f"User: {t}" for t in user_texts)
        llm_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ]

        try:
            llm = self._llm  # type narrowing already checked in caller
            assert llm is not None  # for mypy
            response = await llm.chat(
                llm_messages,
                temperature=0.1,
                max_tokens=1024,
            )
            return self._parse_llm_response(response.content, ctx)
        except Exception:
            logger.warning("LLM extraction failed, falling back to rules", exc_info=True)
            return self._extract_via_rules(
                [{"role": "user", "content": t} for t in user_texts],
                ctx,
            )

    def _parse_llm_response(
        self,
        content: str,
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        """Parse LLM JSON array response into candidates."""
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            items = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON: %s", content[:200])
            return []

        if not isinstance(items, list):
            return []

        candidates: list[MemoryCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            mem_content = item.get("content", "").strip()
            mem_type_str = item.get("type", "fact")
            confidence = float(item.get("confidence", 0.7))

            if not mem_content or confidence < self._min_confidence:
                continue

            candidates.append(
                MemoryCandidate(
                    candidate_id=uuid.uuid4().hex[:12],
                    scope=MemoryScope.USER,
                    content=mem_content,
                    memory_type=_TYPE_MAP.get(mem_type_str, MemoryType.FACT),
                    source_message_ids=[],
                    source_context=f"turn:{ctx.turn_index}",
                    confidence=confidence,
                )
            )
        return candidates

    # ------------------------------------------------------------------
    # Rule-based extraction (English-only fallback)
    # ------------------------------------------------------------------

    def _extract_via_rules(
        self,
        messages: list[dict],
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            msg_id = msg.get("id", "")
            candidates.extend(self._extract_from_text(content, msg_id, ctx))
        return [c for c in candidates if c.confidence >= self._min_confidence]

    def _extract_from_text(
        self,
        text: str,
        message_id: str,
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []

        for m in _EXPLICIT_MEMORY_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    content=m.group(1).strip(),
                    memory_type=MemoryType.FACT,
                    confidence=0.95,
                    message_id=message_id,
                    ctx=ctx,
                )
            )

        for m in _IDENTITY_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    content=f"User name: {m.group(1).strip()}",
                    memory_type=MemoryType.PROFILE,
                    confidence=0.9,
                    message_id=message_id,
                    ctx=ctx,
                )
            )

        for m in _PREFERENCE_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    content=m.group(0).strip(),
                    memory_type=MemoryType.PREFERENCE,
                    confidence=0.8,
                    message_id=message_id,
                    ctx=ctx,
                )
            )

        for m in _CONSTRAINT_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    content=m.group(0).strip(),
                    memory_type=MemoryType.CONSTRAINT,
                    confidence=0.85,
                    message_id=message_id,
                    ctx=ctx,
                )
            )

        return candidates

    @staticmethod
    def _make_candidate(
        *,
        content: str,
        memory_type: MemoryType,
        confidence: float,
        message_id: str,
        ctx: ExtractionContext,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            candidate_id=uuid.uuid4().hex[:12],
            scope=MemoryScope.USER,
            content=content,
            memory_type=memory_type,
            source_message_ids=[message_id] if message_id else [],
            source_context=f"turn:{ctx.turn_index}",
            confidence=confidence,
            extracted_at=time.time(),
        )


# ---------------------------------------------------------------------------
# AtomicFact extractor (6-tuple schema for entity-state write path)
# ---------------------------------------------------------------------------


_ATOMIC_FACT_SYSTEM_PROMPT = """\
You are a fact extraction component for a long-term memory system. \
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
 - Information about other people not connected to the speaker
 - General knowledge facts ("The sky is blue")

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
    """

    facts: list[AtomicFact] = field(default_factory=list)
    raw_sourceless: list[dict[str, Any]] = field(default_factory=list)
    invalid_dropped: int = 0


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
        for item in items:
            fact = self._build_fact(item, anchor)
            if fact is None:
                invalid += 1
            else:
                facts.append(fact)
        return ExtractionResult(facts=facts, invalid_dropped=invalid)

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
