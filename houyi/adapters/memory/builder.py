from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from houyi.adapters.memory.types import (
    ExtractionContext,
    MemoryBuildInput,
    MemoryCandidate,
    MemoryScope,
    MemorySourceKind,
    MemoryType,
)

logger = logging.getLogger(__name__)

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
"""

_TYPE_MAP: dict[str, MemoryType] = {
    "fact": MemoryType.FACT,
    "preference": MemoryType.PREFERENCE,
    "constraint": MemoryType.CONSTRAINT,
    "profile": MemoryType.PROFILE,
}


class MemoryCandidateBuilder:
    def __init__(
        self,
        min_confidence: float = 0.6,
        llm_adapter: Any | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._llm = llm_adapter

    async def build(
        self,
        memory_input: MemoryBuildInput,
        context: ExtractionContext | None = None,
    ) -> list[MemoryCandidate]:
        ctx = context or ExtractionContext()
        if not memory_input.items:
            return []
        if memory_input.source_type == MemorySourceKind.CONVERSATION:
            return await self._build_conversation(memory_input, ctx)
        return self._build_structured(memory_input, ctx)

    async def _build_conversation(
        self,
        memory_input: MemoryBuildInput,
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        if self._llm is not None:
            candidates = await self._build_conversation_via_llm(memory_input, ctx)
            if candidates:
                return candidates
        return self._build_conversation_via_rules(memory_input, ctx)

    async def _build_conversation_via_llm(
        self,
        memory_input: MemoryBuildInput,
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        user_items = [
            item for item in memory_input.items if item.role == "user" and item.content.strip()
        ]
        if not user_items:
            return []

        user_block = "\n".join(f"User: {item.content}" for item in user_items)
        llm_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ]

        try:
            llm = self._llm
            assert llm is not None
            response = await llm.chat(
                llm_messages,
                temperature=0.1,
                max_tokens=1024,
            )
            tags = self._merge_tags(memory_input, user_items)
            return self._parse_llm_response(
                response.content,
                scope=memory_input.scope,
                source_type=memory_input.source_type,
                source_context=memory_input.source_context or f"turn:{ctx.turn_index}",
                suggested_tags=tags,
                metadata=memory_input.metadata,
            )
        except Exception:
            logger.warning("LLM memory build failed, falling back to rules", exc_info=True)
            return []

    def _build_conversation_via_rules(
        self,
        memory_input: MemoryBuildInput,
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for item in memory_input.items:
            if item.role != "user" or not item.content.strip():
                continue
            candidates.extend(self._build_from_text(memory_input, item, ctx))
        return [c for c in candidates if c.confidence >= self._min_confidence]

    def _build_structured(
        self,
        memory_input: MemoryBuildInput,
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for item in memory_input.items:
            content = item.content.strip()
            if not content:
                continue
            candidates.append(
                self._make_candidate(
                    scope=memory_input.scope,
                    content=content,
                    memory_type=item.memory_type
                    or self._default_memory_type(memory_input.source_type),
                    source_type=memory_input.source_type,
                    source_ids=item.source_ids,
                    source_context=item.source_context
                    or memory_input.source_context
                    or f"turn:{ctx.turn_index}",
                    confidence=item.confidence
                    or self._default_confidence(memory_input.source_type),
                    suggested_tags=self._merge_tags(memory_input, [item]),
                    metadata={**memory_input.metadata, **item.metadata, "role": item.role},
                )
            )
        return [c for c in candidates if c.confidence >= self._min_confidence]

    def _build_from_text(
        self,
        memory_input: MemoryBuildInput,
        item: Any,
        ctx: ExtractionContext,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        text = item.content
        for match in _EXPLICIT_MEMORY_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    scope=memory_input.scope,
                    content=match.group(1).strip(),
                    memory_type=MemoryType.FACT,
                    source_type=memory_input.source_type,
                    source_ids=item.source_ids,
                    source_context=item.source_context
                    or memory_input.source_context
                    or f"turn:{ctx.turn_index}",
                    confidence=0.95,
                    suggested_tags=self._merge_tags(memory_input, [item]),
                    metadata={**memory_input.metadata, **item.metadata, "role": item.role},
                )
            )
        for match in _IDENTITY_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    scope=memory_input.scope,
                    content=f"User name: {match.group(1).strip()}",
                    memory_type=MemoryType.PROFILE,
                    source_type=memory_input.source_type,
                    source_ids=item.source_ids,
                    source_context=item.source_context
                    or memory_input.source_context
                    or f"turn:{ctx.turn_index}",
                    confidence=0.9,
                    suggested_tags=self._merge_tags(memory_input, [item]),
                    metadata={**memory_input.metadata, **item.metadata, "role": item.role},
                )
            )
        for match in _PREFERENCE_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    scope=memory_input.scope,
                    content=match.group(0).strip(),
                    memory_type=MemoryType.PREFERENCE,
                    source_type=memory_input.source_type,
                    source_ids=item.source_ids,
                    source_context=item.source_context
                    or memory_input.source_context
                    or f"turn:{ctx.turn_index}",
                    confidence=0.8,
                    suggested_tags=self._merge_tags(memory_input, [item]),
                    metadata={**memory_input.metadata, **item.metadata, "role": item.role},
                )
            )
        for match in _CONSTRAINT_PATTERN.finditer(text):
            candidates.append(
                self._make_candidate(
                    scope=memory_input.scope,
                    content=match.group(0).strip(),
                    memory_type=MemoryType.CONSTRAINT,
                    source_type=memory_input.source_type,
                    source_ids=item.source_ids,
                    source_context=item.source_context
                    or memory_input.source_context
                    or f"turn:{ctx.turn_index}",
                    confidence=0.85,
                    suggested_tags=self._merge_tags(memory_input, [item]),
                    metadata={**memory_input.metadata, **item.metadata, "role": item.role},
                )
            )
        return candidates

    def _parse_llm_response(
        self,
        content: str,
        *,
        scope: MemoryScope,
        source_type: MemorySourceKind,
        source_context: str,
        suggested_tags: list[str],
        metadata: dict[str, Any],
    ) -> list[MemoryCandidate]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON memory payload: %s", text[:200])
            return []
        if not isinstance(items, list):
            return []

        candidates: list[MemoryCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            mem_content = str(item.get("content", "")).strip()
            confidence = float(item.get("confidence", 0.7))
            if not mem_content or confidence < self._min_confidence:
                continue
            candidates.append(
                self._make_candidate(
                    scope=scope,
                    content=mem_content,
                    memory_type=_TYPE_MAP.get(str(item.get("type", "fact")), MemoryType.FACT),
                    source_type=source_type,
                    source_ids=[],
                    source_context=source_context,
                    confidence=confidence,
                    suggested_tags=suggested_tags,
                    metadata=metadata,
                )
            )
        return candidates

    def _default_memory_type(self, source_type: MemorySourceKind) -> MemoryType:
        if source_type == MemorySourceKind.AUTO_DREAM:
            return MemoryType.PROJECT
        return MemoryType.FACT

    def _default_confidence(self, source_type: MemorySourceKind) -> float:
        if source_type == MemorySourceKind.SEARCH:
            return 0.7
        if source_type == MemorySourceKind.AUTO_DREAM:
            return 0.55
        return 0.75

    def _merge_tags(self, memory_input: MemoryBuildInput, items: list[Any]) -> list[str]:
        tags: list[str] = []
        tags.extend(str(tag) for tag in memory_input.metadata.get("suggested_tags", []) if str(tag))
        for item in items:
            tags.extend(tag for tag in item.suggested_tags if tag)
        return list(dict.fromkeys(tags))

    @staticmethod
    def _make_candidate(
        *,
        scope: MemoryScope,
        content: str,
        memory_type: MemoryType,
        source_type: MemorySourceKind,
        source_ids: list[str],
        source_context: str,
        confidence: float,
        suggested_tags: list[str],
        metadata: dict[str, Any],
    ) -> MemoryCandidate:
        return MemoryCandidate(
            candidate_id=uuid.uuid4().hex[:12],
            scope=scope,
            content=content,
            memory_type=memory_type,
            source_type=source_type.value,
            source_message_ids=source_ids,
            source_context=source_context,
            confidence=confidence,
            extracted_at=time.time(),
            suggested_tags=suggested_tags,
            metadata=metadata,
        )
