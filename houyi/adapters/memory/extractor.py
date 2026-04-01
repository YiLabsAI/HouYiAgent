"""Memory candidate extractor.

Extracts memory candidates from conversation messages using:
1. LLM-based extraction (primary) — language-agnostic structured output
2. Rule-based detection (fallback when LLM unavailable) — English explicit
   commands only ("remember that ...", "my name is ...")

LLM mode sends recent user messages to the model with a structured
extraction prompt and parses the JSON array response into candidates.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from houyi.adapters.memory.types import (
    ExtractionContext,
    MemoryCandidate,
    MemoryRecord,
    MemoryScope,
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

    When an ``llm_adapter`` is provided, uses LLM-based extraction
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

        if self._llm is not None:
            return await self._extract_via_llm(messages, ctx)

        return self._extract_via_rules(messages, ctx)

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
                    extracted_at=time.time(),
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
