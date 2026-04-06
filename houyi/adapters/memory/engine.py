"""MemoryEngine: unified facade for the memory write and recall pipelines.

Assembles Extractor, Classifier, Deduplicator, Retriever, Store, and
EmbeddingProvider into a single entry point used by the application layer.

Two primary operations:
- process_messages(): write pipeline (extract → classify → dedup → store)
- recall(): read pipeline (retrieve → format)
"""

from __future__ import annotations

import logging
from typing import Any

from houyi.adapters.memory.builder import MemoryCandidateBuilder
from houyi.adapters.memory.classifier import MemoryClassifier
from houyi.adapters.memory.deduplicator import MemoryDeduplicator
from houyi.adapters.memory.embedding import EmbeddingProvider
from houyi.adapters.memory.extractor import MemoryCandidateExtractor
from houyi.adapters.memory.forgetting import apply_forgetting
from houyi.adapters.memory.retriever import MemoryRetriever
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    CandidateStatus,
    ExtractionContext,
    ForgettingPolicy,
    MemoryBuildInput,
    MemoryBuildItem,
    MemoryCandidate,
    MemoryPolicy,
    MemoryProvenance,
    MemoryRecall,
    MemoryRecord,
    MemoryScope,
    MemorySourceKind,
    SessionContext,
)

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Unified memory facade — write pipeline + recall pipeline.

    Components are optional; the engine degrades gracefully:
    - No embedding_provider → lexical-only retrieval, no semantic dedup
    - No extractor → process_messages() returns empty list
    - No classifier → candidates keep their initial type
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        builder: MemoryCandidateBuilder | None = None,
        extractor: MemoryCandidateExtractor | None = None,
        classifier: MemoryClassifier | None = None,
        deduplicator: MemoryDeduplicator | None = None,
        retriever: MemoryRetriever | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        llm_adapter: Any | None = None,
        policy: MemoryPolicy | None = None,
        forgetting_policy: ForgettingPolicy | None = None,
    ):
        self._store = store
        self._builder = builder or MemoryCandidateBuilder(llm_adapter=llm_adapter)
        self._extractor = extractor or MemoryCandidateExtractor(llm_adapter=llm_adapter)
        self._classifier = classifier or MemoryClassifier()
        self._deduplicator = deduplicator or MemoryDeduplicator(embedding_provider)
        self._retriever = retriever or MemoryRetriever(
            store,
            embedding_provider,
            policy,
        )
        self._embedding = embedding_provider
        self._policy = policy or MemoryPolicy()
        self._forgetting = forgetting_policy or ForgettingPolicy()

    @property
    def store(self) -> MemoryStore:
        """Access the underlying store directly."""
        return self._store

    # ------------------------------------------------------------------
    # Write pipeline
    # ------------------------------------------------------------------

    async def process_messages(
        self,
        messages: list[dict],
        context: ExtractionContext | None = None,
    ) -> list[MemoryCandidate]:
        """Extract → classify → dedup → optionally store.

        Returns candidates (approved ones are already written to store).
        """
        ctx = context or ExtractionContext()
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
        return await self.process_input(memory_input, ctx)

    async def process_input(
        self,
        memory_input: MemoryBuildInput,
        context: ExtractionContext | None = None,
    ) -> list[MemoryCandidate]:
        """Build → classify → dedup → optionally store."""
        existing = self._store.all_records()
        candidates = await self._builder.build(memory_input, context)
        if not candidates:
            return []

        for candidate in candidates:
            candidate.memory_type = await self._classifier.classify(candidate)
            candidate.dedup_matches = await self._deduplicator.check(
                candidate,
                existing,
            )

            if candidate.dedup_matches:
                has_dup = any(m.relation == "duplicate" for m in candidate.dedup_matches)
                if has_dup:
                    candidate.status = CandidateStatus.MERGED
                    continue

            if self._policy.auto_approve:
                candidate.status = CandidateStatus.APPROVED
                await self._store_candidate(candidate)
            else:
                candidate.status = CandidateStatus.PENDING

        logger.debug(
            "Processed %d input items → %d candidates",
            len(memory_input.items),
            len(candidates),
        )
        return candidates

    async def approve_candidate(self, candidate: MemoryCandidate) -> MemoryRecord:
        """Approve and persist a pending candidate."""
        candidate.status = CandidateStatus.APPROVED
        return await self._store_candidate(candidate)

    async def _store_candidate(self, candidate: MemoryCandidate) -> MemoryRecord:
        """Convert an approved candidate into a MemoryRecord and store it."""
        embedding = None
        if self._embedding:
            embs = await self._embedding.embed([candidate.content])
            embedding = embs[0] if embs else None

        record = MemoryRecord(
            scope=candidate.scope,
            key=self._derive_key(candidate),
            content=candidate.content,
            memory_type=candidate.memory_type,
            metadata=candidate.metadata,
            tags=candidate.suggested_tags,
            confidence=candidate.confidence,
            provenance=MemoryProvenance(
                source_type=candidate.source_type,
                source_ids=candidate.source_message_ids,
                extracted_by="memory_candidate_builder",
                extraction_timestamp=candidate.extracted_at,
            ),
            embedding=embedding,
        )
        self._store.put_record(record)

        if embedding and self._embedding:
            self._cache_embedding(
                record.record_id,
                self._embedding,
                embedding,
            )

        return record

    def _cache_embedding(
        self,
        record_id: str,
        provider: EmbeddingProvider,
        embedding: list[float],
    ) -> None:
        """Write embedding to the backend's embedding_cache if supported."""
        backend = getattr(self._store, "_backend", None)
        if backend is None:
            return
        provider_name = type(provider).__name__
        model_name = str(provider.dimension())
        try:
            backend.put_embedding(record_id, provider_name, model_name, embedding)
        except Exception:
            logger.debug("Embedding cache write skipped", exc_info=True)

    @staticmethod
    def _derive_key(candidate: MemoryCandidate) -> str:
        """Derive a storage key from candidate content."""
        words = candidate.content.split()[:5]
        key = "_".join(w.lower().strip(".,!?:;") for w in words if w.strip())
        return key or candidate.candidate_id

    # ------------------------------------------------------------------
    # Recall pipeline
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        session_context: SessionContext | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecall]:
        """Retrieve relevant memories for a query."""
        return await self._retriever.retrieve(query, session_context, top_k)

    async def build_context(
        self,
        query: str,
        session_context: SessionContext | None = None,
        top_k: int = 5,
    ) -> str | None:
        """Recall relevant memories and format as context text.

        Returns None if no relevant memories found, allowing callers
        (e.g. ContextPlanner) to omit the memory block entirely.
        """
        recalls = await self.recall(query, session_context, top_k)
        if not recalls:
            return None
        text = self.recall_as_context_text(recalls)
        return text or None

    def recall_as_context_text(
        self,
        recalls: list[MemoryRecall],
    ) -> str:
        """Render recall results as context text for ContextPlanner."""
        if not recalls:
            return ""
        lines = []
        for r in recalls:
            record = self._find_record(r.memory_id)
            if record:
                lines.append(
                    f"- [{record.memory_type.value}] {record.key}: "
                    f"{record.content} (score={r.score:.2f})"
                )
        return "\n".join(lines)

    def _find_record(self, record_id: str) -> MemoryRecord | None:
        for record in self._store.all_records():
            if record.record_id == record_id:
                return record
        return None

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def run_forgetting(self) -> int:
        """Apply forgetting policy and remove stale records.

        Returns the number of records evicted.
        """
        all_records = self._store.all_records()
        before = len(all_records)
        survivors = apply_forgetting(all_records, self._forgetting)
        evicted = before - len(survivors)

        if evicted > 0:
            survivor_ids = {r.record_id for r in survivors}
            for record in all_records:
                if record.record_id not in survivor_ids:
                    self._store.delete(record.key, record.scope)
            logger.info("Forgetting: evicted %d records", evicted)

        return evicted
