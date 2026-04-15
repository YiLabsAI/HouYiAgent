from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.runtime.search_parsing import _parse_sufficiency
from houyi.application.research.types import (
    AnswerCoverageContract,
    SourceReference,
    SufficiencyDecision,
    SufficiencyFeatures,
)

_QUERY_PROMPT = """\
You are a research quality assessor evaluating whether collected sources \
are sufficient for an academic-grade analysis.

Sub-question: {question}
Sources found so far: {source_count}
Latest results summary: {summary}
Collaboration summary: {collaboration_summary}
Peer gaps still open: {peer_gaps}
Structured evidence: {feature_summary}
Missing dimensions: {missing_dimensions}
Missing facets: {missing_facets}
Noisy-only facets: {noisy_only_facets}
Coverage facets: {coverage_facets}

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


@dataclass(slots=True)
class SufficiencyEvaluator:
    llm: LLMAdapter
    llm_kwargs: dict[str, Any]

    async def evaluate(
        self,
        *,
        question: str,
        user_query: str,
        summary: str,
        sources: list[SourceReference],
        collaboration: dict[str, Any],
        features: SufficiencyFeatures,
        expected_sources: int,
        coverage_contract: AnswerCoverageContract,
    ) -> SufficiencyDecision:
        guardrail = _guardrail_sufficiency_decision(
            question=question,
            user_query=user_query,
            features=features,
            expected_sources=expected_sources,
        )
        if guardrail is not None:
            return guardrail
        prompt = _QUERY_PROMPT.format(
            question=question,
            source_count=len(sources),
            summary=summary or "(no results yet)",
            collaboration_summary=_format_collaboration_summary(collaboration),
            peer_gaps=_format_collaboration_items(collaboration.get("peer_gaps")),
            feature_summary=_format_feature_summary(features),
            missing_dimensions=_format_collaboration_items(features.missing_dimensions),
            missing_facets=_format_collaboration_items(features.missing_facets),
            noisy_only_facets=_format_collaboration_items(features.noisy_only_facets),
            coverage_facets=_format_contract_facets(coverage_contract),
        )
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            **self.llm_kwargs,
        )
        sufficient, rationale = _parse_sufficiency(resp.content)
        return SufficiencyDecision(
            sufficient=sufficient,
            rationale=rationale,
            decision_by="llm",
            reason_code="llm_sufficient" if sufficient else "llm_continue",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )

    def build_features(
        self,
        sources: list[SourceReference],
        question: str,
        user_query: str,
        coverage_contract: AnswerCoverageContract,
    ) -> SufficiencyFeatures:
        source_count = len(sources)
        relevant = _filter_relevant(sources, question, user_query)
        relevant_count = len(relevant)
        domains = {_extract_domain(src.url) for src in sources if _extract_domain(src.url)}
        providers = {src.provider.strip().lower() for src in sources if src.provider.strip()}
        authority_count = sum(1 for src in sources if _is_authoritative_source(src))
        recent_count = sum(1 for src in sources if _looks_recent_source(src))
        relevance_score = relevant_count / max(source_count, 1)
        diversity_score = min(1.0, (len(domains) + len(providers)) / max(source_count, 1))
        authority_score = authority_count / max(source_count, 1)
        recency_score = recent_count / max(source_count, 1)
        covered_facets, noisy_only_facets, missing_facets = _classify_facet_coverage(
            sources,
            coverage_contract,
        )
        noisy_source_count = _count_noisy_sources(sources, question, user_query, coverage_contract)
        missing_dimensions: list[str] = []
        if source_count < 2 or relevance_score < 0.5:
            missing_dimensions.append("relevance")
        if len(domains) < 2:
            missing_dimensions.append("diversity")
        if authority_count == 0:
            missing_dimensions.append("authority")
        if _requires_recency(question, user_query) and recent_count == 0:
            missing_dimensions.append("recency")
        if missing_facets:
            missing_dimensions.append("facet_coverage")
        if noisy_only_facets or noisy_source_count > max(source_count // 2, 1):
            missing_dimensions.append("task_fit")
        return SufficiencyFeatures(
            source_count=source_count,
            relevant_source_count=relevant_count,
            domain_count=len(domains),
            provider_count=len(providers),
            authority_source_count=authority_count,
            recent_source_count=recent_count,
            relevance_score=relevance_score,
            diversity_score=diversity_score,
            authority_score=authority_score,
            recency_score=recency_score,
            has_primary_source=authority_count > 0,
            covered_facets=covered_facets,
            missing_facets=missing_facets,
            noisy_only_facets=noisy_only_facets,
            noisy_source_count=noisy_source_count,
            missing_dimensions=missing_dimensions,
        )


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


def _format_feature_summary(features: SufficiencyFeatures) -> str:
    return (
        f"relevance={features.relevance_score:.2f}; "
        f"diversity={features.diversity_score:.2f}; "
        f"authority={features.authority_score:.2f}; "
        f"recency={features.recency_score:.2f}; "
        f"covered_facets={len(features.covered_facets)}; "
        f"missing_facets={len(features.missing_facets)}; "
        f"noisy_sources={features.noisy_source_count}; "
        f"sources={features.source_count}; "
        f"relevant_sources={features.relevant_source_count}; "
        f"domains={features.domain_count}; "
        f"providers={features.provider_count}; "
        f"primary={features.has_primary_source}"
    )


def _format_contract_facets(contract: AnswerCoverageContract) -> str:
    if not contract.must_cover_facets:
        return "(none)"
    return "; ".join(facet.name for facet in contract.must_cover_facets[:6])


def _extract_keywords(text: str) -> set[str]:
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


def _classify_facet_coverage(
    sources: list[SourceReference],
    coverage_contract: AnswerCoverageContract,
) -> tuple[list[str], list[str], list[str]]:
    if not coverage_contract.must_cover_facets:
        return [], [], []

    covered: list[str] = []
    noisy_only: list[str] = []
    missing: list[str] = []
    for facet in coverage_contract.must_cover_facets:
        strong_hit = False
        weak_hit = False
        facet_terms = _facet_terms(
            facet.name, facet.intent, facet.evidence_hint, facet.bilingual_terms
        )
        for src in sources:
            text = f"{src.title} {src.snippet} {src.url or ''}".lower()
            overlap = sum(1 for term in facet_terms if term in text)
            if overlap >= 2:
                strong_hit = True
                break
            if overlap == 1:
                weak_hit = True
        if strong_hit:
            covered.append(facet.name)
        elif weak_hit:
            noisy_only.append(facet.name)
        else:
            missing.append(facet.name)
    return covered, noisy_only, missing


def _count_noisy_sources(
    sources: list[SourceReference],
    question: str,
    user_query: str,
    coverage_contract: AnswerCoverageContract,
) -> int:
    if not sources:
        return 0
    global_terms = _extract_keywords(question) | _extract_keywords(user_query)
    for facet in coverage_contract.must_cover_facets:
        global_terms |= _facet_terms(
            facet.name,
            facet.intent,
            facet.evidence_hint,
            facet.bilingual_terms,
        )
    noisy = 0
    for src in sources:
        text = f"{src.title} {src.snippet} {src.url or ''}".lower()
        overlap = sum(1 for term in global_terms if term in text)
        if overlap == 0:
            noisy += 1
    return noisy


def _facet_terms(*parts: Any) -> set[str]:
    terms: set[str] = set()
    for part in parts:
        if isinstance(part, list):
            for item in part:
                terms |= _extract_keywords(str(item))
            continue
        terms |= _extract_keywords(str(part))
    return terms


def _guardrail_sufficiency_decision(
    *,
    question: str,
    user_query: str,
    features: SufficiencyFeatures,
    expected_sources: int,
) -> SufficiencyDecision | None:
    minimum_sources = max(1, min(max(expected_sources, 1), 2))
    target_sources = max(minimum_sources, min(max(expected_sources, 1), 4))
    quality_gate_sources = max(2, target_sources)
    required_domains = 1 if target_sources == 1 else 2
    recency_required = _requires_recency(question, user_query)
    if features.source_count == 0:
        return SufficiencyDecision(
            sufficient=False,
            rationale="No sources collected yet",
            decision_by="guardrail",
            reason_code="no_sources",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if features.source_count < minimum_sources:
        return SufficiencyDecision(
            sufficient=False,
            rationale="Need more evidence before stopping",
            decision_by="guardrail",
            reason_code="insufficient_sources",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if features.relevance_score < 0.5:
        return SufficiencyDecision(
            sufficient=False,
            rationale="Collected sources are not relevant enough yet",
            decision_by="guardrail",
            reason_code="low_relevance",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if features.missing_facets:
        return SufficiencyDecision(
            sufficient=False,
            rationale="Collected evidence still leaves planned answer facets uncovered",
            decision_by="guardrail",
            reason_code="missing_facets",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if features.noisy_only_facets:
        return SufficiencyDecision(
            sufficient=False,
            rationale="Some answer facets are backed only by noisy or weakly aligned sources",
            decision_by="guardrail",
            reason_code="noisy_facets",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if (
        quality_gate_sources >= 2
        and features.source_count >= quality_gate_sources
        and features.domain_count < required_domains
    ):
        return SufficiencyDecision(
            sufficient=False,
            rationale="Collected sources are still too concentrated on one domain",
            decision_by="guardrail",
            reason_code="low_diversity",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if (
        quality_gate_sources >= 2
        and features.source_count >= quality_gate_sources
        and features.authority_source_count == 0
    ):
        return SufficiencyDecision(
            sufficient=False,
            rationale="Collected sources still lack authoritative or primary evidence",
            decision_by="guardrail",
            reason_code="low_authority",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if (
        recency_required
        and quality_gate_sources >= 2
        and features.source_count >= quality_gate_sources
        and features.recent_source_count == 0
    ):
        return SufficiencyDecision(
            sufficient=False,
            rationale="Collected sources still lack recent evidence for a recency-sensitive question",
            decision_by="guardrail",
            reason_code="low_recency",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    if (
        features.source_count >= target_sources
        and features.relevance_score >= 0.6
        and features.domain_count >= required_domains
        and features.authority_source_count >= 1
        and (not recency_required or features.recent_source_count >= 1)
    ):
        return SufficiencyDecision(
            sufficient=True,
            rationale="Structured evidence already covers relevance, diversity, and authority",
            decision_by="guardrail",
            reason_code="guardrail_sufficient",
            missing_dimensions=list(features.missing_dimensions),
            features=features,
        )
    return None


def _extract_domain(url: str | None) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower().strip()


def _is_authoritative_source(source: SourceReference) -> bool:
    domain = _extract_domain(source.url)
    if domain.endswith((".gov", ".edu")):
        return True
    if any(token in domain for token in ("arxiv.org", "acm.org", "ieee.org", "github.com")):
        return True
    text = f"{source.title} {source.snippet}".lower()
    return any(token in text for token in ("official", "documentation", "whitepaper", "paper"))


def _looks_recent_source(source: SourceReference) -> bool:
    text = f"{source.title} {source.snippet}"
    years = [int(match) for match in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return False
    return max(years) >= time.gmtime().tm_year - 2


def _requires_recency(question: str, user_query: str) -> bool:
    text = f"{question} {user_query}".lower()
    return any(token in text for token in ("current", "latest", "recent", "2024", "2025", "2026"))
