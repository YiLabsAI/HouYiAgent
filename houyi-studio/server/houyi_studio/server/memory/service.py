"""Memory service — adapts server layer to SDK MemoryEngine.

Handles candidate approval/rejection, record CRUD, and recall history.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from houyi.adapters.memory.answerer import AnswerResult
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.types import (
    CandidateStatus,
    MemoryCandidate,
    MemoryRecord,
    MemoryScope,
    SessionContext,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """Server-side memory management service.

    Maintains a candidate buffer that receives candidates from
    ResearchRuntime.extract_memories() and MemoryEngine.process_messages().
    Provides CRUD for both candidates and records.
    """

    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine
        self._candidates: list[MemoryCandidate] = []

    def add_candidates(self, candidates: list[MemoryCandidate]) -> None:
        """Buffer new candidates from research or chat extraction."""
        self._candidates.extend(candidates)

    def list_candidates(
        self,
        status: CandidateStatus | None = None,
    ) -> list[MemoryCandidate]:
        if status:
            return [c for c in self._candidates if c.status == status]
        return list(self._candidates)

    async def approve_candidate(self, candidate_id: str) -> MemoryRecord | None:
        """Approve a candidate via MemoryEngine and return the record."""
        for c in self._candidates:
            if c.candidate_id == candidate_id:
                record = await self._engine.approve_candidate(c)
                return record
        return None

    async def reject_candidate(self, candidate_id: str) -> bool:
        for c in self._candidates:
            if c.candidate_id == candidate_id:
                c.status = CandidateStatus.REJECTED
                return True
        return False

    async def update_candidate(
        self,
        candidate_id: str,
        content: str | None = None,
        suggested_tags: list[str] | None = None,
    ) -> MemoryCandidate | None:
        for c in self._candidates:
            if c.candidate_id == candidate_id:
                if content is not None:
                    c.content = content
                if suggested_tags is not None:
                    c.suggested_tags = suggested_tags
                return c
        return None

    def list_records(
        self,
        scope: MemoryScope | None = None,
    ) -> list[MemoryRecord]:
        if scope:
            return self._engine.store.list_by_scope(scope)
        return self._engine.store.all_records()

    def get_record(self, record_id: str) -> MemoryRecord | None:
        for r in self._engine.store.all_records():
            if r.record_id == record_id:
                return r
        return None

    async def update_record(
        self,
        record_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryRecord | None:
        record = self.get_record(record_id)
        if not record:
            return None
        if content is not None:
            record.content = content
        if tags is not None:
            record.tags = tags
        self._engine.store.put_record(record)
        return record

    async def delete_record(self, record_id: str) -> bool:
        record = self.get_record(record_id)
        if not record:
            return False
        return self._engine.store.delete(record.key, record.scope)

    async def get_recall_history(self, record_id: str) -> list[dict[str, Any]]:
        """Placeholder — recall tracking to be implemented."""
        return []

    def should_use_memory_answer(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        patterns = (
            r"\bremember\b",
            r"\brecall\b",
            r"\bwhat did i (say|tell)\b",
            r"\bwhat do you remember\b",
            r"\bdo you remember\b",
            r"\bmy (preference|constraint|setting|deadline)\b",
        )
        return any(re.search(pattern, text) is not None for pattern in patterns)

    async def answer_query(
        self,
        query: str,
        *,
        session_context: SessionContext | None = None,
        top_k: int = 5,
    ) -> AnswerResult:
        return await self._engine.answer(
            query=query,
            session_context=session_context,
            top_k=top_k,
        )
