from __future__ import annotations

import json

from houyi.application.research.runtime.search_sufficiency import (
    SufficiencyEvaluator,
    _extract_domain,
    _is_authoritative_source,
    _looks_recent_source,
    _requires_recency,
)
from houyi.application.research.types import (
    AnswerCoverageContract,
    CoverageFacet,
    SourceReference,
    SufficiencyFeatures,
)

from ..conftest import MockLLM


def _src(url: str = "https://x.com", title: str = "", snippet: str = "") -> SourceReference:
    return SourceReference(
        url=url, title=title, snippet=snippet, source_type="web", provider="mock"
    )


def _empty_features() -> SufficiencyFeatures:
    return SufficiencyFeatures(
        source_count=0,
        relevant_source_count=0,
        domain_count=0,
        provider_count=0,
        authority_source_count=0,
        recent_source_count=0,
        relevance_score=0.0,
        diversity_score=0.0,
        authority_score=0.0,
        recency_score=0.0,
        has_primary_source=False,
        missing_dimensions=["relevance", "diversity", "authority"],
    )


def _rich_features() -> SufficiencyFeatures:
    return SufficiencyFeatures(
        source_count=2,
        relevant_source_count=2,
        domain_count=2,
        provider_count=1,
        authority_source_count=1,
        recent_source_count=0,
        relevance_score=1.0,
        diversity_score=1.0,
        authority_score=0.5,
        recency_score=0.0,
        has_primary_source=True,
        missing_dimensions=[],
    )


def _contract() -> AnswerCoverageContract:
    return AnswerCoverageContract(
        must_cover_facets=[
            CoverageFacet(
                name="current role",
                intent="identify employer",
                evidence_hint="official profile",
                bilingual_terms=["Messi official profile"],
            )
        ]
    )


class TestSufficiencyEvaluator:
    async def test_guardrail_blocks_llm(self):
        llm = MockLLM(responses=[])
        evaluator = SufficiencyEvaluator(llm=llm, llm_kwargs={})
        decision = await evaluator.evaluate(
            question="Q?",
            user_query="Q?",
            summary="",
            sources=[],
            collaboration={},
            features=_empty_features(),
            expected_sources=2,
            coverage_contract=AnswerCoverageContract(),
        )
        assert decision.reason_code == "no_sources"
        assert llm._call_count == 0

    async def test_calls_llm_on_guardrail(self):
        llm = MockLLM(responses=[json.dumps({"sufficient": True, "rationale": "enough"})])
        evaluator = SufficiencyEvaluator(llm=llm, llm_kwargs={})
        sources = [
            _src("https://a.com/1", "AI frameworks overview", "overview of AI frameworks"),
            _src("https://b.com/2", "AI frameworks comparison", "comparison across AI frameworks"),
        ]
        decision = await evaluator.evaluate(
            question="Q?",
            user_query="Q?",
            summary="summary",
            sources=sources,
            collaboration={},
            features=_rich_features(),
            expected_sources=3,
            coverage_contract=AnswerCoverageContract(),
        )
        assert decision.decision_by == "llm"
        assert decision.sufficient is True
        assert llm._call_count == 1

    def test_build_features_counts_domains(self):
        evaluator = SufficiencyEvaluator(llm=MockLLM(), llm_kwargs={})
        sources = [
            _src("https://a.com/page", "AI agent framework", "AI frameworks overview"),
            _src("https://b.edu/paper", "Official AI paper", "official documentation 2025"),
        ]
        features = evaluator.build_features(
            sources, "AI frameworks", "AI", AnswerCoverageContract()
        )
        assert features.domain_count == 2
        assert features.authority_source_count >= 1

    def test_build_features_tracks_facets(self):
        evaluator = SufficiencyEvaluator(llm=MockLLM(), llm_kwargs={})
        sources = [
            _src(
                "https://official.example.com/profile",
                "Current role at Example Corp",
                "official profile confirms current role and employer",
            ),
            _src(
                "https://news.example.com/story",
                "Biography overview",
                "general biography story",
            ),
        ]
        features = evaluator.build_features(sources, "Who is this person?", "person", _contract())
        assert features.covered_facets == ["current role"]
        assert features.missing_facets == []

    async def test_guardrail_blocks_missing_facets(self):
        evaluator = SufficiencyEvaluator(llm=MockLLM(responses=[]), llm_kwargs={})
        features = _rich_features()
        features.missing_facets = ["current role"]
        features.missing_dimensions = ["facet_coverage"]
        decision = await evaluator.evaluate(
            question="Who is this person?",
            user_query="Who is this person?",
            summary="summary",
            sources=[_src(title="Biography", snippet="general overview")],
            collaboration={},
            features=features,
            expected_sources=2,
            coverage_contract=_contract(),
        )
        assert decision.reason_code == "missing_facets"

    def test_build_features_marks_entity(self):
        evaluator = SufficiencyEvaluator(llm=MockLLM(), llm_kwargs={})
        sources = [
            _src("https://news.example.com/story", "Biography overview", "general biography story")
        ]
        features = evaluator.build_features(sources, "Messi", "Messi", _contract())
        assert "entity_identity" in features.missing_dimensions

    async def test_guardrail_blocks_entity_noise(self):
        evaluator = SufficiencyEvaluator(llm=MockLLM(responses=[]), llm_kwargs={})
        features = _rich_features()
        features.noisy_source_count = 2
        features.source_count = 3
        decision = await evaluator.evaluate(
            question="Messi",
            user_query="Messi",
            summary="summary",
            sources=[
                _src(title="General index", snippet="list of unrelated same-name pages"),
                _src(title="Biography", snippet="general overview"),
                _src(title="Forum thread", snippet="unverified discussion"),
            ],
            collaboration={},
            features=features,
            expected_sources=3,
            coverage_contract=_contract(),
        )
        assert decision.reason_code == "entity_noise"


class TestHelpers:
    def test_extract_domain(self):
        assert _extract_domain("https://arxiv.org/abs/123") == "arxiv.org"

    def test_extract_domain_empty(self):
        assert _extract_domain(None) == ""

    def test_authoritative_edu(self):
        src = _src("https://example.edu/paper")
        assert _is_authoritative_source(src) is True

    def test_authoritative_arxiv(self):
        src = _src("https://arxiv.org/abs/1234")
        assert _is_authoritative_source(src) is True

    def test_not_authoritative(self):
        src = _src("https://blog.example.com/post")
        assert _is_authoritative_source(src) is False

    def test_recent_source_year(self):
        import time

        year = time.gmtime().tm_year
        src = _src(title=f"AI paper {year}")
        assert _looks_recent_source(src) is True

    def test_requires_recency_keyword(self):
        assert _requires_recency("latest AI frameworks", "") is True

    def test_no_recency_required(self):
        assert _requires_recency("AI frameworks history", "") is False
