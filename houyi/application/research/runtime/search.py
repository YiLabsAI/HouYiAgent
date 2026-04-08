"""SearchExecutor — multi-round search for a single sub-question.

Each research sub-question gets its own SearchExecutor instance. It
generates search queries via the LLM, executes them via WebSearchService
(``include_content=True`` browse mode), evaluates sufficiency, and repeats
until the information collected is sufficient or ``max_search_rounds`` is
reached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import (
    SearchContext,
    SearchHit,
    SearchResult,
    SearchRound,
    SourceReference,
    SubQuestion,
)
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import WebSearchResult

logger = logging.getLogger(__name__)

_MAX_QUERY_LENGTH = 380

SearchEventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
CollaborationSnapshotCallback = Callable[[int], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class _QueryExecution:
    query: str
    hits: list[SearchHit]
    elapsed_ms: float
    provider: str


@dataclass(slots=True)
class _RoundExecution:
    hits: list[SearchHit]
    skipped_queries: int
    cancelled_queries: int


_QUERY_GEN_PROMPT = """\
You are an expert research assistant generating web search queries for \
an academic-grade research task.

Sub-question: {question}
User's overall query: {user_query}
Prior findings: {prior}
Round: {round} of max {max_rounds}
Peer findings: {peer_findings}
Peer queries already attempted: {peer_queries}
Peer gaps still open: {peer_gaps}
Preferred providers from collaboration: {preferred_providers}
Shared source count so far: {shared_source_count}

Generate 2-3 DISTINCT search queries. Strategy:
- Round 1: Start with precise, specific queries using domain terminology, \
author names, paper titles, or technical terms when applicable.
- Later rounds: DIVERSIFY — rephrase, use synonyms, try alternative \
angles, or broaden/narrow scope based on what prior rounds found.
- Avoid duplicating peer queries unless you are intentionally deepening a still-open gap.
- Include at least one query with temporal qualifiers (e.g., "2024", \
"recent", "latest") when the topic benefits from recency.
- For non-English topics, generate queries in BOTH the original language \
and English to maximize source coverage.
- Avoid vague or overly broad queries. Each query should target a specific \
aspect of the sub-question.

Respond ONLY with a JSON array of query strings, e.g. ["query 1", "query 2"].
"""

_SUFFICIENCY_PROMPT = """\
You are a research quality assessor evaluating whether collected sources \
are sufficient for an academic-grade analysis.

Sub-question: {question}
Sources found so far: {source_count}
Latest results summary: {summary}
Collaboration summary: {collaboration_summary}
Peer gaps still open: {peer_gaps}

Evaluate sufficiency on three criteria:
1. **Breadth**: Do sources cover multiple perspectives or data points?
2. **Depth**: Are there authoritative or primary sources (not just summaries)?
3. **Diversity**: Are sources from different authors/publishers/years?

Mark sufficient=true when at least 2 of 3 criteria are met, OR when \
{source_count} >= 6 (diminishing returns beyond this point). Balance \
thoroughness against efficiency — if sources already cover the core \
aspects of the question, stop searching.

Respond ONLY with JSON: {{"sufficient": true/false, "rationale": "..."}}
"""


class SearchExecutor:
    """Coordinates multi-round search for a single sub-question.

    Uses existing ``WebSearchService`` with ``include_content=True`` (browse
    mode) to get both URLs and full content in a single call.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        max_search_rounds: int = 3,
        max_results_per_query: int = 8,
        max_query_parallelism: int = 2,
        on_event: SearchEventCallback | None = None,
        claim_query: Callable[[str], Awaitable[bool]] | None = None,
        claim_url: Callable[[str], Awaitable[bool]] | None = None,
        check_cancelled: Callable[[], None] | None = None,
        get_collaboration_snapshot: CollaborationSnapshotCallback | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._web_search = web_search
        self._max_rounds = max_search_rounds
        self._max_results_per_query = max_results_per_query
        self._max_query_parallelism = max(1, max_query_parallelism)
        self._on_event = on_event
        self._claim_query = claim_query
        self._claim_url = claim_url
        self._check_cancelled = check_cancelled
        self._get_collaboration_snapshot = get_collaboration_snapshot
        self._llm_kwargs = llm_kwargs

    async def search(
        self,
        sub_question: SubQuestion,
        context: SearchContext,
    ) -> SearchResult:
        """Execute multi-round search until information is sufficient.

        Returns a ``SearchResult`` containing all rounds, deduplicated
        sources, and a coverage score.
        """
        rounds: list[SearchRound] = []
        all_sources: dict[str, SourceReference] = {}
        seen_urls: set[str] = set(context.excluded_urls)
        prior = list(context.prior_findings)
        max_results_per_query = self._resolve_max_results_per_query(context)
        total_source_target = self._resolve_total_source_target(sub_question, context)

        for round_idx in range(self._max_rounds):
            self._run_cancel_check()
            round_number = round_idx + 1
            collaboration = await self._read_collaboration_snapshot(round_number)
            stop_reason = _collaboration_stop_reason(collaboration)
            if stop_reason:
                rounds.append(
                    SearchRound(
                        round_index=round_idx,
                        queries=[],
                        hits=[],
                        sufficient=True,
                        rationale=stop_reason,
                        elapsed_ms=0.0,
                    )
                )
                await self._notify(
                    "search.round_timing",
                    {
                        "question_id": sub_question.question_id,
                        "round": round_number,
                        "elapsed_ms": 0.0,
                        "query_count": 0,
                        "skipped_queries": 0,
                        "cancelled_queries": 0,
                        "hit_count": 0,
                        "source_count": len(all_sources),
                        "sufficient": True,
                        "rationale": stop_reason,
                    },
                )
                break
            raw_queries = await self._generate_queries(
                sub_question.question,
                context.user_query,
                prior,
                round_idx,
                collaboration,
            )
            queries, skipped_queries = await self._claim_queries(raw_queries, set())

            await self._notify(
                "search.queries_generated",
                {
                    "question_id": sub_question.question_id,
                    "round": round_number,
                    "queries": queries,
                },
            )

            round_started = time.perf_counter()
            round_result = await self._run_round_queries(
                question_id=sub_question.question_id,
                round_index=round_number,
                queries=queries,
                seen_urls=seen_urls,
                all_sources=all_sources,
                max_results_per_query=max_results_per_query,
                query_parallelism=self._resolve_query_parallelism(context, len(queries)),
                target_total_sources=total_source_target,
            )
            hits = round_result.hits

            summary = "; ".join(h.title for h in hits[:5])
            sufficient, rationale = await self._evaluate_sufficiency(
                sub_question.question,
                len(all_sources),
                summary,
                collaboration,
            )
            round_elapsed_ms = (time.perf_counter() - round_started) * 1000.0

            rounds.append(
                SearchRound(
                    round_index=round_idx,
                    queries=queries,
                    hits=hits,
                    sufficient=sufficient,
                    rationale=rationale,
                    elapsed_ms=round(round_elapsed_ms, 1),
                    skipped_queries=skipped_queries + round_result.skipped_queries,
                    cancelled_queries=round_result.cancelled_queries,
                )
            )

            await self._notify(
                "search.round_timing",
                {
                    "question_id": sub_question.question_id,
                    "round": round_number,
                    "elapsed_ms": round(round_elapsed_ms, 1),
                    "query_count": len(queries),
                    "skipped_queries": skipped_queries + round_result.skipped_queries,
                    "cancelled_queries": round_result.cancelled_queries,
                    "hit_count": len(hits),
                    "source_count": len(all_sources),
                    "sufficient": sufficient,
                    "rationale": rationale,
                },
            )

            if hits:
                prior.append(summary)

            if sufficient or not queries:
                break

        sources = list(all_sources.values())
        coverage = min(1.0, len(sources) / max(sub_question.expected_sources, 1))

        return SearchResult(
            question_id=sub_question.question_id,
            rounds=rounds,
            sources=sources,
            summary=prior[-1] if prior else "",
            coverage_score=coverage,
            exhausted=len(rounds) >= self._max_rounds and not rounds[-1].sufficient,
        )

    async def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event:
            try:
                await self._on_event(event_type, data)
            except Exception:
                logger.debug("Search event callback failed for %s", event_type, exc_info=True)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _generate_queries(
        self,
        question: str,
        user_query: str,
        prior: list[str],
        round_idx: int,
        collaboration: dict[str, Any],
    ) -> list[str]:
        prompt = _QUERY_GEN_PROMPT.format(
            question=question,
            user_query=user_query,
            prior="; ".join(prior[-3:]) if prior else "(none)",
            round=round_idx + 1,
            max_rounds=self._max_rounds,
            peer_findings=_format_collaboration_items(collaboration.get("peer_findings")),
            peer_queries=_format_collaboration_items(collaboration.get("peer_queries")),
            peer_gaps=_format_collaboration_items(collaboration.get("peer_gaps")),
            preferred_providers=_format_collaboration_items(
                collaboration.get("preferred_providers")
            ),
            shared_source_count=collaboration.get("shared_source_count", 0),
        )
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            **self._llm_kwargs,
        )
        return _parse_query_list(resp.content)

    async def _evaluate_sufficiency(
        self,
        question: str,
        source_count: int,
        summary: str,
        collaboration: dict[str, Any],
    ) -> tuple[bool, str]:
        prompt = _SUFFICIENCY_PROMPT.format(
            question=question,
            source_count=source_count,
            summary=summary or "(no results yet)",
            collaboration_summary=_format_collaboration_summary(collaboration),
            peer_gaps=_format_collaboration_items(collaboration.get("peer_gaps")),
        )
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            **self._llm_kwargs,
        )
        return _parse_sufficiency(resp.content)

    async def _read_collaboration_snapshot(self, round_number: int) -> dict[str, Any]:
        if self._get_collaboration_snapshot is None:
            return {}
        snapshot = await self._get_collaboration_snapshot(round_number)
        return snapshot if isinstance(snapshot, dict) else {}

    def _resolve_max_results_per_query(self, context: SearchContext) -> int:
        if context.max_results_per_query > 0:
            return context.max_results_per_query
        return self._max_results_per_query

    def _resolve_query_parallelism(self, context: SearchContext, query_count: int) -> int:
        if query_count <= 0:
            return 1
        configured = context.max_query_parallelism or self._max_query_parallelism
        return max(1, min(configured, self._max_query_parallelism, query_count))

    def _resolve_total_source_target(
        self, sub_question: SubQuestion, context: SearchContext
    ) -> int:
        configured_cap = context.max_total_sources if context.max_total_sources > 0 else 100
        expected = max(sub_question.expected_sources, 1)
        return max(1, min(configured_cap, expected))

    def _run_cancel_check(self) -> None:
        if self._check_cancelled is not None:
            self._check_cancelled()

    async def _claim_queries(
        self,
        queries: list[str],
        seen_queries: set[str],
    ) -> tuple[list[str], int]:
        claimed: list[str] = []
        skipped = 0
        for query in queries:
            normalized = _canonical_query(query)
            if not normalized or normalized in seen_queries:
                skipped += 1
                continue
            if self._claim_query is not None and not await self._claim_query(normalized):
                skipped += 1
                continue
            seen_queries.add(normalized)
            claimed.append(query)
        return claimed, skipped

    async def _run_round_queries(
        self,
        *,
        question_id: str,
        round_index: int,
        queries: list[str],
        seen_urls: set[str],
        all_sources: dict[str, SourceReference],
        max_results_per_query: int,
        query_parallelism: int,
        target_total_sources: int,
    ) -> _RoundExecution:
        if not queries:
            return _RoundExecution(hits=[], skipped_queries=0, cancelled_queries=0)

        if query_parallelism <= 1 or len(queries) <= 1:
            return await self._run_round_queries_serial(
                question_id=question_id,
                round_index=round_index,
                queries=queries,
                seen_urls=seen_urls,
                all_sources=all_sources,
                max_results_per_query=max_results_per_query,
                target_total_sources=target_total_sources,
            )

        return await self._run_round_queries_parallel(
            question_id=question_id,
            round_index=round_index,
            queries=queries,
            seen_urls=seen_urls,
            all_sources=all_sources,
            max_results_per_query=max_results_per_query,
            query_parallelism=query_parallelism,
            target_total_sources=target_total_sources,
        )

    async def _run_round_queries_serial(
        self,
        *,
        question_id: str,
        round_index: int,
        queries: list[str],
        seen_urls: set[str],
        all_sources: dict[str, SourceReference],
        max_results_per_query: int,
        target_total_sources: int,
    ) -> _RoundExecution:
        hits: list[SearchHit] = []
        cancelled_queries = 0
        for index, query in enumerate(queries):
            execution = await self._search_query(query, max_results_per_query)
            hits.extend(
                await self._merge_query_hits(
                    question_id,
                    execution.query,
                    execution.hits,
                    seen_urls,
                    all_sources,
                )
            )
            await self._notify_query_timing(question_id, round_index, execution)
            if len(all_sources) < target_total_sources:
                continue
            remaining = queries[index + 1 :]
            cancelled_queries = len(remaining)
            await self._notify_cancelled_queries(question_id, round_index, remaining)
            break

        return _RoundExecution(
            hits=hits,
            skipped_queries=0,
            cancelled_queries=cancelled_queries,
        )

    async def _run_round_queries_parallel(
        self,
        *,
        question_id: str,
        round_index: int,
        queries: list[str],
        seen_urls: set[str],
        all_sources: dict[str, SourceReference],
        max_results_per_query: int,
        query_parallelism: int,
        target_total_sources: int,
    ) -> _RoundExecution:
        semaphore = asyncio.Semaphore(query_parallelism)

        async def _run_one(query: str) -> _QueryExecution:
            async with semaphore:
                return await self._search_query(query, max_results_per_query)

        tasks = {asyncio.create_task(_run_one(query)): query for query in queries}
        hits: list[SearchHit] = []
        cancelled_queries = 0

        for task in asyncio.as_completed(tasks):
            self._run_cancel_check()
            execution = await task
            hits.extend(
                await self._merge_query_hits(
                    question_id,
                    execution.query,
                    execution.hits,
                    seen_urls,
                    all_sources,
                )
            )
            await self._notify_query_timing(question_id, round_index, execution)
            if len(all_sources) < target_total_sources:
                continue
            cancelled_queries = await self._cancel_pending_query_tasks(
                question_id,
                round_index,
                tasks,
            )
            break

        if cancelled_queries:
            await asyncio.gather(*tasks, return_exceptions=True)

        return _RoundExecution(
            hits=hits,
            skipped_queries=0,
            cancelled_queries=cancelled_queries,
        )

    async def _notify_query_timing(
        self,
        question_id: str,
        round_index: int,
        execution: _QueryExecution,
    ) -> None:
        await self._notify(
            "search.query_timing",
            {
                "question_id": question_id,
                "round": round_index,
                "query": execution.query,
                "elapsed_ms": round(execution.elapsed_ms, 1),
                "hit_count": len(execution.hits),
                "cancelled": False,
                "provider": execution.provider,
            },
        )

    async def _notify_cancelled_queries(
        self,
        question_id: str,
        round_index: int,
        queries: list[str],
    ) -> None:
        for pending_query in queries:
            await self._notify(
                "search.query_cancelled",
                {
                    "question_id": question_id,
                    "round": round_index,
                    "query": pending_query,
                    "reason": "source_target_reached",
                },
            )

    async def _cancel_pending_query_tasks(
        self,
        question_id: str,
        round_index: int,
        tasks: dict[asyncio.Task[_QueryExecution], str],
    ) -> int:
        pending_queries: list[str] = []
        for pending, pending_query in tasks.items():
            if pending.done():
                continue
            pending.cancel()
            pending_queries.append(pending_query)
        await self._notify_cancelled_queries(question_id, round_index, pending_queries)
        return len(pending_queries)

    async def _merge_query_hits(
        self,
        question_id: str,
        query: str,
        items: list[SearchHit],
        seen_urls: set[str],
        all_sources: dict[str, SourceReference],
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for hit in items:
            if hit.url in seen_urls:
                continue
            if self._claim_url is not None and not await self._claim_url(hit.url):
                seen_urls.add(hit.url)
                continue
            seen_urls.add(hit.url)
            hits.append(hit)
            src = _to_source_ref(hit)
            all_sources[src.reference_id] = src
            await self._notify(
                "search.source_discovered",
                {
                    "question_id": question_id,
                    "title": hit.title,
                    "url": hit.url,
                    "snippet": (hit.snippet or "")[:200],
                    "query": query,
                },
            )
        return hits

    async def _search_query(
        self,
        query: str,
        max_results_per_query: int,
    ) -> _QueryExecution:
        started = time.perf_counter()
        try:
            resp = await self._web_search.search(
                query,
                max_results=max_results_per_query,
                include_content=True,
            )
        except Exception:
            logger.warning("Search query failed: %s", query, exc_info=True)
            return _QueryExecution(
                query=query,
                hits=[],
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                provider="",
            )
        return _QueryExecution(
            query=query,
            hits=[_to_search_hit(result, resp.provider) for result in resp.results],
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            provider=resp.provider,
        )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _to_search_hit(r: WebSearchResult, provider: str) -> SearchHit:
    return SearchHit(
        url=r.url,
        title=r.title,
        snippet=r.snippet,
        content=r.content,
        provider=provider,
        published_at=None,
        rank=0,
    )


def _to_source_ref(hit: SearchHit) -> SourceReference:
    return SourceReference(
        url=hit.url,
        title=hit.title,
        snippet=hit.snippet,
        source_type="web",
        reliability_score=0.5,
    )


def _parse_query_list(content: str) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _normalize_queries([str(q) for q in parsed[:5]])
    except json.JSONDecodeError:
        pass

    numbered = _extract_numbered_queries(text)
    if numbered:
        return _normalize_queries(numbered[:5])

    return _normalize_queries([text]) if text else []


def _extract_numbered_queries(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    queries: list[str] = []
    for line in lines:
        line = line.lstrip("-• ")
        match = re.match(
            r"(?:\*\*)?query\s*\d+\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(.+)",
            line,
            re.I,
        )
        if match:
            candidate = match.group(1).strip()
            if candidate:
                queries.append(candidate)
    return queries


def _normalize_queries(queries: list[str]) -> list[str]:
    normalized: list[str] = []
    for query in queries:
        candidate = " ".join(query.split()).strip()
        if not candidate:
            continue
        candidate = candidate[:_MAX_QUERY_LENGTH].rstrip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _canonical_query(query: str) -> str:
    normalized = _normalize_queries([query])
    if not normalized:
        return ""
    return normalized[0].lower()


def _format_collaboration_items(items: Any) -> str:
    if not items:
        return "(none)"
    if isinstance(items, str):
        return items.strip() or "(none)"
    if not isinstance(items, list):
        return str(items)
    rendered = [str(item).strip() for item in items if str(item).strip()]
    if not rendered:
        return "(none)"
    return "; ".join(rendered[:6])


def _format_collaboration_summary(collaboration: dict[str, Any]) -> str:
    if not collaboration:
        return "(none)"
    parts: list[str] = []
    shared_source_count = collaboration.get("shared_source_count")
    if isinstance(shared_source_count, int) and shared_source_count > 0:
        parts.append(f"shared_sources={shared_source_count}")
    preferred_providers = collaboration.get("preferred_providers")
    if preferred_providers:
        parts.append(f"preferred_providers={_format_collaboration_items(preferred_providers)}")
    peer_gaps = collaboration.get("peer_gaps")
    if peer_gaps:
        parts.append(f"peer_gaps={_format_collaboration_items(peer_gaps)}")
    return "; ".join(parts) if parts else "(none)"


def _collaboration_stop_reason(collaboration: dict[str, Any]) -> str | None:
    stop_reason = collaboration.get("stop_reason")
    if not stop_reason:
        return None
    text = str(stop_reason).strip()
    return text or None


def _parse_sufficiency(content: str) -> tuple[bool, str]:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return False, "Failed to parse sufficiency evaluation"
        return bool(data.get("sufficient", False)), str(data.get("rationale", ""))
    except json.JSONDecodeError:
        return False, "Failed to parse sufficiency evaluation"


_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "for",
        "nor",
        "so",
        "yet",
        "at",
        "by",
        "in",
        "of",
        "on",
        "to",
        "up",
        "is",
        "it",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "shall",
        "can",
        "could",
        "may",
        "might",
        "would",
        "should",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "how",
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "into",
        "about",
    ]
)


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, filtering stop words."""
    if not text:
        return set()
    words = [w.strip(".,!?;:\"'()[]{}") for w in text.lower().split()]
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


def _filter_relevant(
    sources: list[SourceReference],
    question: str,
    user_query: str,
    min_overlap: int = 1,
) -> list[SourceReference]:
    """Keep sources whose title/snippet share keywords with the query."""
    if not sources:
        return []
    keywords = _extract_keywords(question) | _extract_keywords(user_query)
    if not keywords:
        return sources
    kept = []
    for src in sources:
        text = f"{src.title} {src.snippet}".lower()
        if not text.strip():
            kept.append(src)
            continue
        overlap = sum(1 for kw in keywords if kw in text)
        if overlap >= min_overlap:
            kept.append(src)
    return kept if kept else sources
