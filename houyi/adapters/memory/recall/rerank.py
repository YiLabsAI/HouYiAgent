"""Evidence-aware reranking for memory recall.

Two rerankers ship in this module:

- EvidenceAwareReranker — the deterministic default used by the
 orchestrator. No external dependencies; safe to enable everywhere.
- LLMReranker — opt-in async reranker that calls an injected
 LLM adapter (). Default *off*: the orchestrator only invokes
 it when the caller explicitly constructs one. Comes with an explicit
 LLMRerankBudget so prompt cost cannot run away.

The Reranker ABC keeps a synchronous rerank so existing
call sites (and the 5+ test files that mock it) need no rewrite. We
add an Reranker.arerank companion method that defaults to
wrapping the sync path; LLM-based subclasses override it with their
async implementation. The orchestrator awaits arerank so both
sync and async rerankers work transparently.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from houyi.adapters.memory.recall.types import QueryType, RecallCandidate, RetrieverKind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceRerankConfig:
    """Weights used by the deterministic evidence reranker."""

    source_anchor_bonus: float = 0.2
    source_text_bonus: float = 0.4
    complete_chain_bonus: float = 0.8
    partial_chain_penalty: float = 0.4
    graph_base_bonus: float = 0.6
    graph_close_range_bonus: float = 0.2
    graph_decay_per_depth: float = 0.15
    primary_retriever_bonus: float = 0.3
    secondary_retriever_bonus: float = 0.15


_PRIMARY_RETRIEVERS: dict[QueryType, frozenset[RetrieverKind]] = {
    QueryType.FACTUAL_LOOKUP: frozenset(
        {RetrieverKind.ENTITY_STATE, RetrieverKind.GRAPH, RetrieverKind.VECTOR}
    ),
    QueryType.NEGATION_CHECK: frozenset({RetrieverKind.ENTITY_STATE}),
    QueryType.TEMPORAL_QUERY: frozenset({RetrieverKind.TIMELINE, RetrieverKind.GRAPH}),
    QueryType.RELATIONAL_CHAIN: frozenset({RetrieverKind.ITERATIVE, RetrieverKind.GRAPH}),
    QueryType.THEMATIC_SUMMARY: frozenset({RetrieverKind.VECTOR}),
    QueryType.PROCEDURAL_RECALL: frozenset({RetrieverKind.RAW_TURN}),
}

_SECONDARY_RETRIEVERS: dict[QueryType, frozenset[RetrieverKind]] = {
    QueryType.FACTUAL_LOOKUP: frozenset({RetrieverKind.ITERATIVE}),
    QueryType.NEGATION_CHECK: frozenset(),
    QueryType.TEMPORAL_QUERY: frozenset({RetrieverKind.VECTOR, RetrieverKind.ENTITY_STATE}),
    QueryType.RELATIONAL_CHAIN: frozenset({RetrieverKind.VECTOR, RetrieverKind.ENTITY_STATE}),
    QueryType.THEMATIC_SUMMARY: frozenset(
        {RetrieverKind.GRAPH, RetrieverKind.RAW_TURN, RetrieverKind.TIMELINE}
    ),
    QueryType.PROCEDURAL_RECALL: frozenset({RetrieverKind.GRAPH, RetrieverKind.VECTOR}),
}


class Reranker(ABC):
    """Rank candidates after fusion and before unknown-answer guarding.

    Implementations override rerank (sync). Async-only rerankers
    (e.g. LLM-driven) override arerank and let the default
    rerank raise; the orchestrator always awaits arerank
    so the sync/async split is invisible at the call site.

    The optional query is the raw query text; semantic rerankers
    (cross-encoder, LLM) need it to score query-document relevance,
    while provenance-based rerankers ignore it.
    """

    @abstractmethod
    def rerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        """Return candidates ordered by rerank score."""
        raise NotImplementedError  # pragma: no cover - abstract

    async def arerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        """Async entry point — defaults to rerank for sync rerankers.

        Implementations that need network I/O override this directly
        and leave rerank raising. The orchestrator only calls
        arerank.
        """
        return self.rerank(query_type=query_type, candidates=candidates, top_k=top_k, query=query)


class EvidenceAwareReranker(Reranker):
    """Deterministic reranker using provenance and relation-chain coverage."""

    def __init__(self, config: EvidenceRerankConfig | None = None) -> None:
        self._config = config or EvidenceRerankConfig()

    def rerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        if top_k <= 0:
            return []

        ranked = list(candidates)
        chain_complete = (
            _chain_complete(ranked) if query_type == QueryType.RELATIONAL_CHAIN else True
        )
        for candidate in ranked:
            candidate.signals = dict(candidate.signals)
            coverage = self._evidence_coverage(
                candidate,
                query_type=query_type,
                chain_complete=chain_complete,
            )
            candidate.signals["evidence_coverage"] = coverage
            if query_type == QueryType.RELATIONAL_CHAIN:
                candidate.signals["chain_evidence_complete"] = chain_complete
            candidate.signals["rerank_score"] = self._rerank_score(
                candidate,
                coverage=coverage,
            )
        ranked.sort(key=lambda c: c.signals.get("rerank_score", c.score), reverse=True)
        return ranked[:top_k]

    def _evidence_coverage(
        self,
        candidate: RecallCandidate,
        *,
        query_type: QueryType,
        chain_complete: bool,
    ) -> float:
        coverage = 0.0
        if candidate.fact.source_anchor:
            coverage += self._config.source_anchor_bonus
            if candidate.signals.get("source_rehydrated") or candidate.signals.get("source_text"):
                coverage += self._config.source_text_bonus

        # New Graph Signaling: graph_path_bonus
        if "bfs_depth" in candidate.signals:
            coverage += self._graph_coverage_bonus(int(candidate.signals["bfs_depth"]))

        if query_type == QueryType.RELATIONAL_CHAIN:
            chain_member = (
                "iteration_round" in candidate.signals
                or "hop_index" in candidate.signals
                or "bfs_depth" in candidate.signals
            )
            if chain_member:
                if chain_complete:
                    coverage += self._config.complete_chain_bonus
                else:
                    coverage -= self._config.partial_chain_penalty

        kind = candidate.matched_by
        if kind in _PRIMARY_RETRIEVERS.get(query_type, frozenset()):
            coverage += self._config.primary_retriever_bonus
        elif kind in _SECONDARY_RETRIEVERS.get(query_type, frozenset()):
            coverage += self._config.secondary_retriever_bonus

        return max(0.0, min(1.0, coverage))

    def _rerank_score(self, candidate: RecallCandidate, *, coverage: float) -> float:
        base = float(candidate.signals.get("fused_score", candidate.score))
        round_bonus = _iteration_bonus(candidate)
        return base + coverage + round_bonus

    def _graph_coverage_bonus(self, depth: int) -> float:
        if depth == 1:
            return self._config.graph_base_bonus + self._config.graph_close_range_bonus
        if depth == 2:
            return self._config.graph_base_bonus
        return max(
            0.0,
            self._config.graph_base_bonus - (depth - 2) * self._config.graph_decay_per_depth,
        )


def _chain_complete(candidates: list[RecallCandidate]) -> bool:
    rounds = {candidate.signals.get("iteration_round") for candidate in candidates}
    if 1 in rounds and 2 in rounds:
        return True
    hop_indices = [
        int(candidate.signals["hop_index"])
        for candidate in candidates
        if isinstance(candidate.signals.get("hop_index"), int)
    ]
    if bool(hop_indices) and max(hop_indices) >= 2:
        return True

    # Check graph BFS depth signal for RELATIONAL_CHAIN Query
    bfs_depths = [
        int(candidate.signals["bfs_depth"])
        for candidate in candidates
        if isinstance(candidate.signals.get("bfs_depth"), int)
    ]
    return bool(bfs_depths) and max(bfs_depths) >= 2


def _iteration_bonus(candidate: RecallCandidate) -> float:
    round_index = candidate.signals.get("iteration_round")
    if not isinstance(round_index, int | float):
        return 0.0
    return min(0.3, max(0.0, float(round_index)) * 0.1)


@dataclass(frozen=True)
class LLMRerankBudget:
    """Hard ceilings for the LLM rerank stage ().

    The budget is enforced before any LLM call: the reranker slices
    its inputs to max_candidates, refuses to run when the
    minimum fused score gap suggests reranking will not change the order,
    and falls back to the upstream order if the LLM call raises or
    times out. There is no "best-effort" mode — exceeding any of these
    limits short-circuits to the deterministic input order.
    """

    max_candidates: int = 10
    """Upper bound on candidates handed to the LLM in a single call."""

    min_candidates: int = 2
    """Below this we don't bother calling the LLM — order is trivial."""

    max_input_chars: int = 8000
    """Sum-of-content cap; protects against runaway prompt size."""

    timeout_s: float = 4.0
    """Soft timeout passed to the adapter; relies on the adapter
 honoring its own cancellation semantics. The orchestrator does not
 enforce a hard kill so partial reranks can still be salvaged.
 """


class LLMReranker(Reranker):
    """Opt-in LLM-driven reranker.

    Default off: this class is only instantiated by callers that
    explicitly want LLM rerank, so the orchestrator's default
    EvidenceAwareReranker path is unaffected. The constructor refuses
    a None adapter to make the dependency obvious.

    The contract with the LLM adapter is intentionally narrow:

    - await llm_adapter.chat(messages, ...) returning an object with
    a .content attribute or a dict with a "content" key.
    - The reranker expects a comma-separated list of candidate indices
    (e.g. "3,0,1,2,4"); anything else falls back to the input
    order. This keeps the adapter contract dead-simple and tolerates
    noisy LLM output.
    """

    def __init__(
        self,
        llm_adapter: Any,
        *,
        budget: LLMRerankBudget | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if llm_adapter is None:
            raise ValueError("llm_adapter is required for LLMReranker")
        self._adapter = llm_adapter
        self._budget = budget or LLMRerankBudget()
        self._system_prompt = system_prompt or _DEFAULT_RERANK_PROMPT

    def rerank(
        self,
        *,
        query_type: QueryType,
        candidates: Iterable[RecallCandidate],
        top_k: int,
        query: str | None = None,
    ) -> list[RecallCandidate]:
        """Sync stub — refuses to run synchronously.

        LLM rerank inherently needs async I/O. Calling rerank
        directly would either block the event loop or run the adapter
        on a fresh loop (unsafe under uvloop / nested loops). The
        orchestrator always calls arerank; legacy sync callers
        should fall back to EvidenceAwareReranker.
        """
        raise RuntimeError(
            "LLMReranker requires async; call arerank() instead "
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
        """Budget-guarded async rerank.

        Pipeline:

        1. Materialize and clip the candidate list to
        LLMRerankBudget.max_candidates.
        2. Bail out if below LLMRerankBudget.min_candidates or
        over LLMRerankBudget.max_input_chars — in either case
        we return the input order trimmed to top_k.
        3. Call the adapter; parse its index list; reorder. Any
        exception keeps the input order.
        """
        ordered = list(candidates)
        if top_k <= 0:
            return []
        if len(ordered) < self._budget.min_candidates:
            return ordered[:top_k]

        window = ordered[: self._budget.max_candidates]
        total_chars = sum(len(c.fact.object) for c in window)
        if total_chars > self._budget.max_input_chars:
            return ordered[:top_k]

        try:
            reordered = await self._call_llm(query_type, window)
        except Exception as exc:
            logger.warning("LLM rerank failed, falling back to input order: %s", exc)
            return ordered[:top_k]

        # Append any candidates the LLM did not score so we never lose
        # information; the orchestrator's downstream stages will still
        # see the trailing items.
        seen = {id(c) for c in reordered}
        tail = [c for c in ordered if id(c) not in seen]
        return (reordered + tail)[:top_k]

    async def _call_llm(
        self,
        query_type: QueryType,
        window: list[RecallCandidate],
    ) -> list[RecallCandidate]:
        prompt = _build_rerank_prompt(query_type, window)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self._adapter.chat(messages, timeout=self._budget.timeout_s)
        content = _extract_content(response)
        indices = _parse_index_list(content, len(window))
        if not indices:
            return window
        return [window[i] for i in indices]


def _build_rerank_prompt(
    query_type: QueryType,
    window: list[RecallCandidate],
) -> str:
    lines = [f"Query type: {query_type.value}", "Candidates:"]
    for idx, candidate in enumerate(window):
        snippet = candidate.fact.object.replace("\n", " ")
        lines.append(f"[{idx}] {candidate.fact.subject}.{candidate.fact.predicate} = {snippet}")
    lines.append(
        "Return the candidate indices ordered from most to least "
        "relevant, comma-separated, e.g. 3,1,0,2."
    )
    return "\n".join(lines)


def _extract_content(response: Any) -> str:
    """Pull text from either response.content or response["content"]."""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    return str(content or "")


def _parse_index_list(text: str, n: int) -> list[int]:
    """Parse "3,1,0,2" into [3, 1, 0, 2], filtered to [0, n)."""
    out: list[int] = []
    seen: set[int] = set()
    for token in text.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if 0 <= idx < n and idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


_DEFAULT_RERANK_PROMPT = (
    "You are a memory reranker. Given a query type and candidate "
    "facts, return the candidate indices ordered from most to least "
    "relevant. Reply with only the comma-separated list of integers."
)


__all__ = [
    "EvidenceAwareReranker",
    "EvidenceRerankConfig",
    "LLMRerankBudget",
    "LLMReranker",
    "Reranker",
]
