"""Cross-encoder reranker + fallback chain for memory recall.

A cross-encoder scores each (query, document) pair jointly -- the model
reads the query against the document and outputs a relevance score. This
is stronger than the bi-encoder vector path (which embeds query and
document separately) or the heuristic EvidenceAwareReranker (which boosts
on entity/answer-type matches without measuring query-document relevance).
For a query like "where has Evan been on roadtrips", the cross-encoder
scores "Evan took his family on a road trip to Jasper" far above "Evan
watercolor painting", where the heuristic (entity-match boost) scores both
equally.

The FallbackReranker chains tiers so a missing model or a transient error
never breaks recall: local cross-encoder -> (optional) LLM reranker ->
heuristic EvidenceAwareReranker. Each tier catches its own failures and
falls through; the first tier that produces a result wins.

Model: BAAI/bge-reranker-base (local, no API cost), cached under the same
HuggingFace cache as the bge-small embedder (~/.cache/huggingface by
default). The model loads lazily on first use and is reused for the
process lifetime.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable, Sequence
from typing import Any

from houyi.adapters.memory.recall.rerank import EvidenceAwareReranker, Reranker
from houyi.adapters.memory.recall.types import QueryType, RecallCandidate

# Pin HuggingFace Hub to offline mode by default so a cached model never
# triggers network HEAD-checks (which spam + time out on every startup).
# Respect an explicit HF_HUB_OFFLINE from the environment. First-time
# download: set HF_HUB_OFFLINE=0 once, run, the model caches under
# ~/.cache/huggingface, then revert.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _candidate_text(c: RecallCandidate) -> str:
    """Render a candidate as the document text the cross-encoder scores."""
    f = c.fact
    text = f"{f.subject} {f.predicate} {f.object}"
    if f.event_time:
        text = f"{text} (time: {f.event_time})"
    return text


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker (bge-reranker-base).

    Scores every (query, candidate_text) pair through a joint model that
    outperforms bi-encoder cosine + heuristic boosts on semantic relevance.
    The model loads lazily (cache-only first, network fallback on miss) so
    a cold cache still works on first run, and a missing/unreachable model
    is caught by the FallbackReranker.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        max_candidates: int = 64,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        # Two-stage retrieval bound: the cross-encoder re-scores only the
        # top-N of the fused pool, not the whole pool. 64 is within standard
        # rerank depth (BEIR/RAG practice 50-100) and bounds the per-query
        # rerank cost. Candidates beyond 64 are kept in the tail (unscored,
        # at their pre-rerank fused score) so downstream boost + MMR stages
        # still see them -- a gold fact that sits in the fused tail stays
        # reachable instead of being silently dropped.
        self._max_candidates = max_candidates
        self._batch_size = batch_size
        self._model: Any | None = None
        self._load_failed = False

    def _load_model(self) -> Any | None:
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("sentence_transformers not installed; cross-encoder rerank disabled")
            self._load_failed = True
            return None
        # Cache-only first (no HuggingFace Hub HEAD-check spam on every startup);
        # fall back to a network fetch on cache miss so the first run still
        # downloads the model. CrossEncoder passes model_kwargs through to the
        # underlying AutoModel, so local_files_only=True disables the hub
        # round-trip without changing the model.
        try:
            return CrossEncoder(
                self._model_name,
                device=self._device,
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            pass
        try:
            return CrossEncoder(self._model_name, device=self._device)
        except Exception as exc:
            logger.warning("cross-encoder model %s failed to load: %s", self._model_name, exc)
            self._load_failed = True
            return None

    def rerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        raise RuntimeError(
            "CrossEncoderReranker requires async; call arerank() instead "
            "(orchestrator does so automatically)."
        )

    async def arerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        ordered = list(candidates)
        if top_k <= 0:
            return []
        if query is None:
            # Without the query text a cross-encoder cannot score; signal the
            # caller (FallbackReranker) to try the next tier.
            raise ValueError("cross-encoder rerank requires the query text")
        model = self._load_model()
        if model is None:
            raise RuntimeError("cross-encoder model unavailable")
        # Score the bounded top-N of the fused pool (already fused-score
        # ordered by the orchestrator). Candidates beyond max_candidates are
        # kept in the tail (unscored) so downstream stages still see them.
        window = ordered[: self._max_candidates]
        pairs = [(query, _candidate_text(c)) for c in window]
        scores = await asyncio.to_thread(model.predict, pairs, batch_size=self._batch_size)
        for c, score in zip(window, scores, strict=False):
            c.signals["rerank_score"] = float(score)
        window.sort(key=lambda c: float(c.signals["rerank_score"]), reverse=True)
        # Preserve unscored tail so downstream stages still see every candidate.
        scored_ids = {id(c) for c in window}
        tail = [c for c in ordered if id(c) not in scored_ids]
        return (window + tail)[:top_k] if top_k else window + tail


class FallbackReranker(Reranker):
    """Chain rerankers so a failure degrades to the next tier.

    Tries each reranker in order; the first that returns a non-empty list
    wins. A tier that raises (model missing, network error, bad input) is
    caught and the next tier runs. The last tier (typically the heuristic
    EvidenceAwareReranker) is the guaranteed fallback.
    """

    def __init__(self, tiers: Sequence[Reranker]) -> None:
        if not tiers:
            raise ValueError("FallbackReranker needs at least one tier")
        self._tiers = list(tiers)

    def rerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        ordered = list(candidates)
        last_exc: Exception | None = None
        for tier in self._tiers:
            try:
                result = tier.rerank(
                    query_type=query_type, candidates=list(ordered), top_k=top_k, query=query
                )
                if result:
                    return result
            except Exception as exc:
                last_exc = exc
                logger.info(
                    "rerank tier %s failed, falling through: %s",
                    type(tier).__name__,
                    exc,
                )
        # All tiers failed or returned empty; return input order as last resort.
        if last_exc is not None:
            logger.warning("all rerank tiers failed; returning input order")
        return ordered[:top_k]

    async def arerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        ordered = list(candidates)
        last_exc: Exception | None = None
        for tier in self._tiers:
            try:
                result = await tier.arerank(
                    query_type=query_type, candidates=list(ordered), top_k=top_k, query=query
                )
                if result:
                    return result
            except Exception as exc:
                last_exc = exc
                logger.info(
                    "rerank tier %s failed, falling through: %s",
                    type(tier).__name__,
                    exc,
                )
        if last_exc is not None:
            logger.warning("all rerank tiers failed; returning input order")
        return ordered[:top_k]


def build_default_reranker(*, llm_adapter: Any | None = None) -> Reranker:
    """Build the reranker chain from the RERANKER_PROVIDER env var.

    - local (default): cross-encoder -> heuristic fallback. Fast, no
      API cost.
    - auto: cross-encoder -> LLM reranker (when an llm_adapter is
      supplied) -> heuristic fallback. Best-effort.

    The heuristic EvidenceAwareReranker is always the last tier so recall
    never breaks on a missing model or transient error.
    """
    import os

    provider = os.environ.get("RERANKER_PROVIDER", "local").strip().lower()
    heuristic = EvidenceAwareReranker()
    cross = CrossEncoderReranker()

    if provider == "auto" and llm_adapter is not None:
        from houyi.adapters.memory.recall.rerank import LLMReranker

        return FallbackReranker([cross, LLMReranker(llm_adapter), heuristic])
    return FallbackReranker([cross, heuristic])


__all__ = [
    "CrossEncoderReranker",
    "FallbackReranker",
    "build_default_reranker",
]
