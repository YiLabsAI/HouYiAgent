"""SearchCoordinator — multi-round search for a single sub-question.

Each research sub-question gets its own SearchCoordinator instance. It
generates search queries via the LLM, executes them via WebSearchService
(``include_content=True`` browse mode), evaluates sufficiency, and repeats
until the information collected is sufficient or ``max_search_rounds`` is
reached.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
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

_QUERY_GEN_PROMPT = """\
You are an expert research assistant generating web search queries for \
an academic-grade research task.

Sub-question: {question}
User's overall query: {user_query}
Prior findings: {prior}
Round: {round} of max {max_rounds}

Generate 2-3 DISTINCT search queries. Strategy:
- Round 1: Start with precise, specific queries using domain terminology, \
author names, paper titles, or technical terms when applicable.
- Later rounds: DIVERSIFY — rephrase, use synonyms, try alternative \
angles, or broaden/narrow scope based on what prior rounds found.
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


class SearchCoordinator:
    """Coordinates multi-round search for a single sub-question.

    Uses existing ``WebSearchService`` with ``include_content=True`` (browse
    mode) to get both URLs and full content in a single call.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        max_search_rounds: int = 10,
        max_results_per_round: int = 8,
        on_event: SearchEventCallback | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._web_search = web_search
        self._max_rounds = max_search_rounds
        self._max_per_round = max_results_per_round
        self._on_event = on_event
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

        for round_idx in range(self._max_rounds):
            queries = await self._generate_queries(
                sub_question.question,
                context.user_query,
                prior,
                round_idx,
            )

            await self._notify(
                "search.queries_generated",
                {
                    "question_id": sub_question.question_id,
                    "round": round_idx + 1,
                    "queries": queries,
                },
            )

            hits: list[SearchHit] = []
            for query in queries:
                try:
                    resp = await self._web_search.search(
                        query,
                        max_results=self._max_per_round,
                        include_content=True,
                    )
                    for r in resp.results:
                        hit = _to_search_hit(r, resp.provider)
                        if hit.url not in seen_urls:
                            seen_urls.add(hit.url)
                            hits.append(hit)
                            src = _to_source_ref(hit)
                            all_sources[src.reference_id] = src
                            await self._notify(
                                "search.source_discovered",
                                {
                                    "question_id": sub_question.question_id,
                                    "title": hit.title,
                                    "url": hit.url,
                                    "snippet": (hit.snippet or "")[:200],
                                    "query": query,
                                },
                            )
                except Exception:
                    logger.warning("Search query failed: %s", query, exc_info=True)

            summary = "; ".join(h.title for h in hits[:5])
            sufficient, rationale = await self._evaluate_sufficiency(
                sub_question.question,
                len(all_sources),
                summary,
            )

            rounds.append(
                SearchRound(
                    round_index=round_idx,
                    queries=queries,
                    hits=hits,
                    sufficient=sufficient,
                    rationale=rationale,
                )
            )

            if hits:
                prior.append(summary)

            if sufficient:
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
    ) -> list[str]:
        prompt = _QUERY_GEN_PROMPT.format(
            question=question,
            user_query=user_query,
            prior="; ".join(prior[-3:]) if prior else "(none)",
            round=round_idx + 1,
            max_rounds=self._max_rounds,
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
    ) -> tuple[bool, str]:
        prompt = _SUFFICIENCY_PROMPT.format(
            question=question,
            source_count=source_count,
            summary=summary or "(no results yet)",
        )
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            **self._llm_kwargs,
        )
        return _parse_sufficiency(resp.content)


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


def _parse_sufficiency(content: str) -> tuple[bool, str]:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        data = json.loads(text)
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
