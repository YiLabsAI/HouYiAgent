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

_QUERY_GEN_PROMPT = """\
You are a research assistant. Generate 2-3 distinct web search queries to \
investigate the following sub-question. Consider the prior findings and \
excluded URLs.

Sub-question: {question}
User's overall query: {user_query}
Prior findings: {prior}
Round: {round} of max {max_rounds}

Respond ONLY with a JSON array of query strings, e.g. ["query 1", "query 2"].
"""

_SUFFICIENCY_PROMPT = """\
You are evaluating search results for a research sub-question.

Sub-question: {question}
Sources found so far: {source_count}
Latest results summary: {summary}

Is the information collected SUFFICIENT to comprehensively answer this \
sub-question? Consider breadth, depth, and source diversity.

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
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._web_search = web_search
        self._max_rounds = max_search_rounds
        self._max_per_round = max_results_per_round
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
            temperature=0.4,
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
            return [str(q) for q in parsed[:5]]
    except json.JSONDecodeError:
        pass
    return [text] if text else []


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
