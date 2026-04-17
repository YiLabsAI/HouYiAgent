from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.runtime.search_parsing import _canonical_query, _parse_query_list
from houyi.application.research.taxonomy import (
    ANALYTIC_TOPIC_CJK_HINTS as _ANALYTIC_TOPIC_CJK_HINTS,
)
from houyi.application.research.taxonomy import (
    ANALYTIC_TOPIC_EN_HINTS as _ANALYTIC_TOPIC_EN_HINTS,
)
from houyi.application.research.taxonomy import (
    ENGLISH_OFFICIAL_MARKERS as _ENGLISH_OFFICIAL_MARKERS,
)
from houyi.application.research.taxonomy import (
    IDENTITY_SOURCE_MARKERS as _IDENTITY_SOURCE_MARKERS,
)
from houyi.application.research.types import AnswerCoverageContract

CollaborationSnapshotCallback = Callable[[int], Awaitable[dict[str, Any]]]

# CJK entity anchor length bounds for name-like segments.
#
# Rationale:
# - <2 chars is usually too short and ambiguous for Chinese names in web search.
# - >6 chars tends to capture full clauses or descriptors, not entity anchors.
# These bounds are intentionally conservative and can be tuned with benchmark data.
_CJK_ANCHOR_MIN_LEN = 2
_CJK_ANCHOR_MAX_LEN = 6

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
Coverage facets: {coverage_facets}
Required caveats: {required_caveats}
Evidence expectations: {evidence_expectations}
Time scope: {time_scope}
Geo scope: {geo_scope}

- Round 1: Start with precise, specific queries using domain terminology, \
author names, paper titles, or technical terms when applicable.
- Later rounds: DIVERSIFY — rephrase, use synonyms, try alternative \
angles, or broaden/narrow scope based on what prior rounds found.
- Avoid duplicating peer queries unless you are intentionally deepening a still-open gap.
- Include at least one query with temporal qualifiers (e.g., "2026", \
"recent", "latest") when the topic benefits from recency.
- For non-English topics, generate queries in BOTH the original language \
and English to maximize source coverage.
- For person, organization, project, or repository queries, include at least \
one identity-anchored query that targets official profiles, employer or org pages, \
repositories, documentation, or papers instead of generic same-name pages.
- When the query contains CJK text and the subject likely has an English-facing \
footprint, prefer both native-language identity queries and English or romanized \
identity queries.
- Avoid same-name dump lists or broad disambiguation pages unless the task is \
explicitly about resolving ambiguity.
- Avoid vague or overly broad queries. Each query should target a specific \
aspect of the sub-question.
- CRITICAL: Every query MUST be self-contained — include the subject/entity name \
together with the topic in the SAME query string. Never generate a bare entity name \
as a standalone query, and never generate a topic-only query without the subject. \
Bad: ["current employer", "John Doe"]. Good: ["John Google current employer"].

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
        coverage_contract: AnswerCoverageContract | None = None,
        *,
        query_type: str = "factual",
        disambiguation_needed: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        coverage_contract = coverage_contract or AnswerCoverageContract()
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
            coverage_facets=_format_coverage_facets(coverage_contract),
            required_caveats=_format_collaboration_items(coverage_contract.required_caveats),
            evidence_expectations=_format_collaboration_items(
                coverage_contract.evidence_expectations
            ),
            time_scope=coverage_contract.time_scope or "(none)",
            geo_scope=coverage_contract.geo_scope or "(none)",
        )
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            **self.llm_kwargs,
        )
        parsed = _parse_query_list(resp.content)
        finalized, metadata = _ensure_bilingual_queries(
            parsed,
            question,
            user_query,
            coverage_contract=coverage_contract,
            query_type=query_type,
            disambiguation_needed=disambiguation_needed,
        )
        metadata["coverage_facets"] = [facet.name for facet in coverage_contract.must_cover_facets]
        metadata["query_type"] = query_type
        metadata["disambiguation_needed"] = disambiguation_needed
        return finalized, metadata

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


def _format_coverage_facets(contract: AnswerCoverageContract) -> str:
    if not contract.must_cover_facets:
        return "(none)"
    rendered: list[str] = []
    for facet in contract.must_cover_facets[:6]:
        parts = [facet.name]
        if facet.intent:
            parts.append(f"intent={facet.intent}")
        if facet.evidence_hint:
            parts.append(f"evidence={facet.evidence_hint}")
        if facet.bilingual_terms:
            parts.append(f"terms={', '.join(facet.bilingual_terms[:4])}")
        rendered.append(" | ".join(parts))
    return "; ".join(rendered)


def _looks_entity_query(
    question: str,
    user_query: str,
    coverage_contract: AnswerCoverageContract,
) -> bool:
    if _contract_requests_identity_surface(coverage_contract):
        return True
    return any(_looks_named_entity_surface(text) for text in (question, user_query) if text)


def _contract_requests_identity_surface(contract: AnswerCoverageContract) -> bool:
    for facet in contract.must_cover_facets:
        joined = " ".join(
            part
            for part in (
                facet.name,
                facet.intent,
                facet.evidence_hint,
                " ".join(facet.bilingual_terms),
            )
            if part
        ).strip()
        if not joined:
            continue
        lowered = joined.lower()
        if any(marker in lowered for marker in _IDENTITY_SOURCE_MARKERS):
            return True
        if any(_looks_named_entity_surface(term) for term in facet.bilingual_terms if term.strip()):
            return True
    return False


def _looks_named_entity_surface(text: str) -> bool:
    normalized = _normalize_query_text(text)
    if not normalized:
        return False
    compact = re.sub(r"[\s\-_/|:：,，。、“”‘’()（）\[\]{}]+", "", normalized)
    lowered = normalized.lower()
    if _contains_cjk(compact):
        if any(marker in compact for marker in _ANALYTIC_TOPIC_CJK_HINTS):
            return False
        cjk_chars = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
        ascii_terms = _extract_ascii_terms(normalized)
        return (
            cjk_chars > 0
            and cjk_chars <= 6
            and len(compact) <= 12
            and (bool(ascii_terms) or cjk_chars <= 4)
        )
    tokens = [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.&'_-]*", normalized) if token]
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    if not alpha_tokens:
        return False
    if any(marker in lowered for marker in _ANALYTIC_TOPIC_EN_HINTS):
        return False
    title_like = sum(
        1 for token in alpha_tokens if token[:1].isupper() or token.isupper() or "." in token
    )
    return len(alpha_tokens) <= 4 and title_like >= max(1, len(alpha_tokens) - 1)


def _collect_english_entity_seeds(
    question: str,
    user_query: str,
    coverage_contract: AnswerCoverageContract,
) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()
    candidates: list[str] = []
    for facet in coverage_contract.must_cover_facets[:3]:
        candidates.extend(facet.bilingual_terms[:4])
        candidates.extend(part for part in (facet.name, facet.intent, facet.evidence_hint) if part)
    candidates.extend([question, user_query])
    for candidate in candidates:
        seed = _extract_ascii_terms(candidate)
        normalized = _canonical_query(seed)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        seeds.append(seed)
    return seeds[:4]


def _expand_english_entity_queries(seed: str) -> list[str]:
    queries = [_strengthen_english_query(seed, prefer_profile=True)]
    lowered = seed.lower()
    if all(marker not in lowered for marker in ("github", "linkedin", "profile")):
        queries.append(f"{seed} github profile")
    return queries


def _query_metadata(
    queries: list[str],
    *,
    bilingual_expected: bool,
    bilingual_fallback_applied: bool,
    entity_query_expected: bool,
    missing_english_entity_seed: bool,
) -> dict[str, Any]:
    return {
        "bilingual_expected": bilingual_expected,
        "language_mix": [_query_language(query) for query in queries],
        "bilingual_fallback_applied": bilingual_fallback_applied,
        "query_role_mix": [_query_role(query) for query in queries],
        "entity_query_expected": entity_query_expected,
        "missing_english_entity_seed": missing_english_entity_seed,
    }


def _english_fallback_queries(
    *,
    question: str,
    user_query: str,
    coverage_contract: AnswerCoverageContract,
    entity_query_expected: bool,
) -> tuple[list[str], bool]:
    english_seeds = (
        _collect_english_entity_seeds(question, user_query, coverage_contract)
        if entity_query_expected
        else []
    )
    if not english_seeds:
        english_seed = _extract_ascii_terms(question) or _extract_ascii_terms(user_query)
        if english_seed:
            english_seeds = [english_seed]
    if not english_seeds:
        return [], entity_query_expected
    expanded: list[str] = []
    for seed in english_seeds[:2]:
        expanded.extend(
            _expand_english_entity_queries(seed)
            if entity_query_expected
            else [_strengthen_english_query(seed)]
        )
    return expanded, False


def _has_english_official_query(queries: list[str]) -> bool:
    return any(_query_role(query) == "english_official" for query in queries)


def _ensure_entity_query_mix(
    queries: list[str],
    *,
    question: str,
    user_query: str,
    coverage_contract: AnswerCoverageContract,
    entity_query_expected: bool,
) -> tuple[list[str], bool]:
    if not entity_query_expected:
        return queries, False
    if _has_english_official_query(queries):
        return queries, False
    english_fallbacks, missing_english_entity_seed = _english_fallback_queries(
        question=question,
        user_query=user_query,
        coverage_contract=coverage_contract,
        entity_query_expected=True,
    )
    if not english_fallbacks:
        return queries, missing_english_entity_seed
    return queries + english_fallbacks, False


def _enforce_entity_composition(
    queries: list[str],
    *,
    question: str,
    user_query: str,
    entity_query_expected: bool,
) -> list[str]:
    """Ensure every query contains the entity anchor for entity-type questions.

    If a query is a bare topic without the entity name, prepend the entity
    anchor so the search engine can disambiguate.  Bare entity-name-only
    queries get the sub-question topic appended.
    """
    if not entity_query_expected:
        return queries
    anchor = _extract_entity_anchor(question, user_query)
    if not anchor:
        return queries
    anchor_lower = anchor.lower()
    result: list[str] = []
    for query in queries:
        query_lower = query.lower()
        # Already contains the entity anchor — keep as-is.
        if anchor_lower in query_lower or anchor in query:
            result.append(query)
            continue
        # Bare topic query — prepend entity anchor.
        composed = f"{anchor} {query}".strip()
        result.append(composed)
    return result


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = _canonical_query(query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(query)
    return deduped


def _ensure_bilingual_queries(
    queries: list[str],
    question: str,
    user_query: str,
    coverage_contract: AnswerCoverageContract | None = None,
    *,
    query_type: str = "factual",
    disambiguation_needed: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    coverage_contract = coverage_contract or AnswerCoverageContract()
    # Planner metadata takes precedence over keyword heuristic.
    entity_query_expected = (
        query_type == "entity"
        or disambiguation_needed
        or _looks_entity_query(question, user_query, coverage_contract)
    )
    original = [_normalize_query_text(query) for query in queries if _normalize_query_text(query)]
    original = _prepend_facet_queries(
        original,
        coverage_contract,
        question=question,
        user_query=user_query,
        entity_query_expected=entity_query_expected,
    )
    if not original:
        return [], _query_metadata(
            [],
            bilingual_expected=False,
            bilingual_fallback_applied=False,
            entity_query_expected=entity_query_expected,
            missing_english_entity_seed=False,
        )
    expects_bilingual = _contains_cjk(question) or _contains_cjk(user_query)
    if not expects_bilingual:
        return original, _query_metadata(
            original,
            bilingual_expected=False,
            bilingual_fallback_applied=False,
            entity_query_expected=entity_query_expected,
            missing_english_entity_seed=False,
        )
    has_english = any(_query_language(query) == "en" for query in original)
    has_original = any(_query_language(query) == "non_en" for query in original)
    finalized = list(original)
    fallback_applied = False
    missing_english_entity_seed = False
    if not has_original:
        finalized.insert(0, question.strip() or user_query.strip())
        fallback_applied = True
    if not has_english:
        english_fallbacks, missing_english_entity_seed = _english_fallback_queries(
            question=question,
            user_query=user_query,
            coverage_contract=coverage_contract,
            entity_query_expected=entity_query_expected,
        )
        if english_fallbacks:
            finalized.extend(english_fallbacks)
            fallback_applied = True
    finalized, entity_mix_missing = _ensure_entity_query_mix(
        finalized,
        question=question,
        user_query=user_query,
        coverage_contract=coverage_contract,
        entity_query_expected=entity_query_expected,
    )
    missing_english_entity_seed = missing_english_entity_seed or entity_mix_missing
    finalized = _enforce_entity_composition(
        finalized,
        question=question,
        user_query=user_query,
        entity_query_expected=entity_query_expected,
    )
    finalized = [
        _strengthen_query_roles(query, entity_query_expected=entity_query_expected)
        for query in finalized
    ]
    deduped = _dedupe_queries(finalized)
    return deduped, _query_metadata(
        deduped,
        bilingual_expected=True,
        bilingual_fallback_applied=fallback_applied,
        entity_query_expected=entity_query_expected,
        missing_english_entity_seed=missing_english_entity_seed,
    )


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _query_language(query: str) -> str:
    return "non_en" if _contains_cjk(query) else "en"


def _normalize_query_text(query: str) -> str:
    return " ".join(query.strip().split())


def _strengthen_query_roles(query: str, *, entity_query_expected: bool = False) -> str:
    if _query_language(query) != "en":
        return query
    return _strengthen_english_query(query, prefer_profile=entity_query_expected)


def _strengthen_english_query(query: str, *, prefer_profile: bool = False) -> str:
    lowered = query.lower()
    if any(marker in lowered for marker in _ENGLISH_OFFICIAL_MARKERS):
        return query
    suffix = "official profile" if prefer_profile else "official report"
    return f"{query} {suffix}"


def _query_role(query: str) -> str:
    if _query_language(query) != "en":
        return "native_local"
    lowered = query.lower()
    if "benchmark" in lowered or "dataset" in lowered:
        return "english_benchmark"
    if any(marker in lowered for marker in _ENGLISH_OFFICIAL_MARKERS):
        return "english_official"
    return "english_general"


def _extract_ascii_terms(text: str) -> str:
    tokens = [
        token for token in text.split() if token.isascii() and any(ch.isalpha() for ch in token)
    ]
    return " ".join(tokens[:8]).strip()


def _extract_entity_anchor(question: str, user_query: str) -> str:
    """Extract the short entity name from question or user_query for query composition.

    For CJK-heavy text, returns the shortest CJK-dominant segment (likely the
    person/org name).  For English, returns the capitalized proper-noun phrase.
    Returns empty string when no entity anchor is detected.
    """
    for text in (question, user_query):
        if not text:
            continue
        # CJK entity: look for short CJK-dominant segments
        cjk_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        if cjk_chars:
            # Split on any character that is NOT a CJK ideograph.
            # This is structural: we define what we keep (CJK runs) rather
            # than enumerating punctuation/quote characters to exclude.
            segments = re.split(r"[^\u4e00-\u9fff]+", text)
            cjk_segments = [
                seg
                for seg in segments
                if seg and _CJK_ANCHOR_MIN_LEN <= len(seg) <= _CJK_ANCHOR_MAX_LEN
            ]
            if cjk_segments:
                return min(cjk_segments, key=len)
    # English: extract capitalized proper-noun phrase
    tokens = question.split() if question else user_query.split()
    proper = [t for t in tokens if (t and t[0].isupper() and not t.isupper()) or len(t) <= 3]
    return " ".join(proper[:3]).strip()


def _prepend_facet_queries(
    queries: list[str],
    coverage_contract: AnswerCoverageContract,
    *,
    question: str = "",
    user_query: str = "",
    entity_query_expected: bool = False,
) -> list[str]:
    """Build search queries from facet metadata, optionally composed with entity anchor.

    For entity-type questions, composes the entity name with topic-relevant
    terms to produce queries a search engine can disambiguate.  For non-entity
    questions, uses concrete bilingual terms directly (no anchor composition).
    """
    if not coverage_contract.must_cover_facets:
        return queries

    entity_anchor = _extract_entity_anchor(question, user_query) if entity_query_expected else ""
    facet_queries: list[str] = []
    for facet in coverage_contract.must_cover_facets[:2]:
        # Skip the identity meta-facet — entity anchoring handles disambiguation.
        if facet.name.strip().lower() == "identity":
            continue
        # Collect only concrete, search-worthy terms from bilingual_terms.
        concrete_terms = [
            term
            for term in facet.bilingual_terms[:3]
            if term.strip()
            and len(term.strip()) >= 2
            and not any(
                meta in term.lower()
                for meta in (
                    "confirm",
                    "identify",
                    "distinguish",
                    "disambiguate",
                    "official",
                    "profile",
                    "organization",
                    "anchor",
                )
            )
        ]
        if concrete_terms and entity_anchor:
            # Entity query: compose anchor + topic term in a single query.
            for term in concrete_terms[:1]:
                composed = f"{entity_anchor} {term}".strip()
                facet_queries.append(composed)
        elif concrete_terms:
            # Non-entity query: use concrete terms directly.
            facet_queries.extend(concrete_terms[:1])
        elif entity_anchor and facet.name.strip():
            # Entity query with no concrete terms: compose anchor + facet name.
            name = facet.name.strip()
            if len(name) >= 3 and not any(
                meta in name.lower() for meta in ("identity", "confirm", "distinguish", "anchor")
            ):
                facet_queries.append(f"{entity_anchor} {name}".strip())
    return facet_queries + queries
