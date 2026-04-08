from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.runtime.search_parsing import _canonical_query, _parse_query_list

CollaborationSnapshotCallback = Callable[[int], Awaitable[dict[str, Any]]]

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


@dataclass(slots=True)
class QueryPlanner:
    llm: LLMAdapter
    max_rounds: int
    llm_kwargs: dict[str, Any]
    claim_query: Callable[[str], Awaitable[bool]] | None = None
    get_collaboration_snapshot: CollaborationSnapshotCallback | None = None

    async def read_collaboration_snapshot(self, round_number: int) -> dict[str, Any]:
        if self.get_collaboration_snapshot is None:
            return {}
        snapshot = await self.get_collaboration_snapshot(round_number)
        return snapshot if isinstance(snapshot, dict) else {}

    async def generate_queries(
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
            max_rounds=self.max_rounds,
            peer_findings=_format_collaboration_items(collaboration.get("peer_findings")),
            peer_queries=_format_collaboration_items(collaboration.get("peer_queries")),
            peer_gaps=_format_collaboration_items(collaboration.get("peer_gaps")),
            preferred_providers=_format_collaboration_items(
                collaboration.get("preferred_providers")
            ),
            shared_source_count=collaboration.get("shared_source_count", 0),
        )
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            **self.llm_kwargs,
        )
        return _parse_query_list(resp.content)

    async def claim_queries(
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
            if self.claim_query is not None and not await self.claim_query(normalized):
                skipped += 1
                continue
            seen_queries.add(normalized)
            claimed.append(query)
        return claimed, skipped


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


def _collaboration_stop_reason(collaboration: dict[str, Any]) -> str | None:
    stop_reason = collaboration.get("stop_reason")
    if not stop_reason:
        return None
    text = str(stop_reason).strip()
    return text or None
