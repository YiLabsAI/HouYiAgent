from __future__ import annotations

import json
from unittest.mock import AsyncMock

from houyi.application.research.runtime.search_query_planner import (
    QueryPlanner,
    _enforce_entity_composition,
    _ensure_bilingual_queries,
    _extract_entity_anchor,
    _prepend_facet_queries,
)
from houyi.application.research.types import AnswerCoverageContract, CoverageFacet

from ..conftest import MockLLM


def _planner(**kwargs) -> QueryPlanner:
    return QueryPlanner(llm=MockLLM(), max_rounds=3, llm_kwargs={}, **kwargs)


def _non_english_query() -> str:
    return "".join(
        chr(codepoint) for codepoint in [0x667A, 0x80FD, 0x4F53, 0x6846, 0x67B6, 0x5BF9, 0x6BD4]
    )


def _non_english_entity() -> str:
    return "".join(chr(codepoint) for codepoint in [0x67D0, 0x7532])


def _non_english_lookup_suffix() -> str:
    return "".join(chr(codepoint) for codepoint in [0x8D44, 0x6599])


def _non_english_identity_question() -> str:
    return "".join(chr(codepoint) for codepoint in [0x662F, 0x8C01])


class TestQueryPlanner:
    async def test_queries_bilingual(self):
        non_english_query = _non_english_query()
        planner = QueryPlanner(
            llm=MockLLM(responses=[json.dumps([f"{non_english_query} 2026"], ensure_ascii=False)]),
            max_rounds=3,
            llm_kwargs={},
        )
        queries, metadata = await planner.generate_queries(
            f"{non_english_query} HouYi LangChain agent framework",
            f"{non_english_query} HouYi LangChain agent framework",
            [],
            0,
            {},
        )
        assert any(any(ch.isascii() and ch.isalpha() for ch in query) for query in queries)
        assert any(any("\u4e00" <= ch <= "\u9fff" for ch in query) for query in queries)
        assert metadata["bilingual_expected"] is True
        assert "query_role_mix" in metadata
        assert any(role == "english_official" for role in metadata["query_role_mix"])

    async def test_strengthens_english_seed(self):
        non_english_query = _non_english_query()

        queries, metadata = _ensure_bilingual_queries(
            [non_english_query],
            f"{non_english_query} HouYi LangChain",
            f"{non_english_query} HouYi LangChain",
        )

        assert any("official report" in query.lower() for query in queries if query.isascii())
        assert metadata["bilingual_fallback_applied"] is True
        assert "english_official" in metadata["query_role_mix"]

    async def test_uses_contract_terms(self):
        entity = _non_english_entity()
        lookup = _non_english_lookup_suffix()
        identity_question = _non_english_identity_question()
        queries, metadata = _ensure_bilingual_queries(
            [f"{entity} {lookup}"],
            f"{entity}{identity_question}",
            f"{entity}{identity_question}",
            coverage_contract=AnswerCoverageContract(
                must_cover_facets=[
                    CoverageFacet(
                        name="current role",
                        intent="identify employer",
                        evidence_hint="official profile",
                        bilingual_terms=["Sample Person official profile", "Sample Person GitHub"],
                    )
                ]
            ),
        )

        assert metadata["entity_query_expected"] is True
        assert metadata["missing_english_entity_seed"] is False
        assert any("sample person" in query.lower() for query in queries)
        # After entity composition, the composed query may be classified as
        # native_local (CJK+English mix) rather than english_official.
        assert any(
            role in ("english_official", "native_local") for role in metadata["query_role_mix"]
        )

    async def test_entity_seed_marks_missing(self):
        entity = _non_english_entity()
        lookup = _non_english_lookup_suffix()
        identity_question = _non_english_identity_question()
        queries, metadata = _ensure_bilingual_queries(
            [f"{entity} {lookup}"],
            f"{entity}{identity_question}",
            f"{entity}{identity_question}",
            coverage_contract=AnswerCoverageContract(),
        )

        # Entity composition may prepend the anchor extracted from the question.
        assert len(queries) == 1
        assert entity in queries[0]
        assert lookup in queries[0]
        assert metadata["entity_query_expected"] is True
        assert metadata["missing_english_entity_seed"] is True

    async def test_non_entity_cjk(self):
        non_english_query = _non_english_query()
        queries, metadata = _ensure_bilingual_queries(
            [f"{non_english_query} comparison"],
            non_english_query,
            non_english_query,
            coverage_contract=AnswerCoverageContract(
                must_cover_facets=[
                    CoverageFacet(
                        name="framework comparison",
                        intent="compare agent frameworks",
                        bilingual_terms=["agent framework comparison"],
                    )
                ]
            ),
        )

        assert metadata["entity_query_expected"] is False
        assert metadata["missing_english_entity_seed"] is False
        assert any(
            "agent framework comparison" in query.lower() for query in queries if query.isascii()
        )

    async def test_entity_query_adds_mix(self):
        entity = _non_english_entity()
        identity_question = _non_english_identity_question()
        queries, metadata = _ensure_bilingual_queries(
            [f"{entity}", "Sample Person biography"],
            f"{entity}{identity_question}",
            f"{entity}{identity_question}",
            coverage_contract=AnswerCoverageContract(
                must_cover_facets=[
                    CoverageFacet(
                        name="current role",
                        intent="identify employer",
                        evidence_hint="official profile",
                        bilingual_terms=["Sample Person official profile"],
                    )
                ]
            ),
        )

        assert metadata["entity_query_expected"] is True
        # After entity composition, queries may be CJK+English mix.
        assert any(
            role in ("english_official", "native_local") for role in metadata["query_role_mix"]
        )
        assert any(
            "sample person" in query.lower() or "official" in query.lower() for query in queries
        )

    async def test_claim_dedupes_normalized(self):
        planner = _planner(claim_query=AsyncMock(side_effect=[True, False]))
        claimed, skipped = await planner.claim_queries(["Q1", "q1", "Q2"], set())
        assert claimed == ["Q1"]
        assert skipped == 2

    async def test_snapshot_normalizes_non_dict(self):
        async def _bad(_round):
            return [_round]

        planner = _planner(get_collaboration_snapshot=_bad)
        result = await planner.read_collaboration_snapshot(1)
        assert result == {}

    async def test_snapshot_returns_empty(self):
        planner = _planner()
        result = await planner.read_collaboration_snapshot(1)
        assert result == {}

    async def test_prepends_facet_queries(self):
        contract = AnswerCoverageContract(
            must_cover_facets=[
                CoverageFacet(
                    name="current role",
                    intent="identify employer",
                    evidence_hint="engineering career",
                    bilingual_terms=["engineering leadership career"],
                )
            ]
        )
        queries, metadata = _ensure_bilingual_queries(
            ["person biography"],
            "person biography",
            "person biography",
            coverage_contract=contract,
        )
        assert queries[0] == "engineering leadership career"
        assert metadata["bilingual_expected"] is False

    async def test_emits_coverage_metadata(self):
        planner = QueryPlanner(
            llm=MockLLM(responses=[json.dumps(["person current role official profile"])]),
            max_rounds=3,
            llm_kwargs={},
        )
        contract = AnswerCoverageContract(
            must_cover_facets=[CoverageFacet(name="current role", intent="identify employer")]
        )
        queries, metadata = await planner.generate_queries(
            "Who is the person?",
            "Who is the person?",
            [],
            0,
            {},
            coverage_contract=contract,
        )
        assert queries
        assert metadata["coverage_facets"] == ["current role"]

    async def test_metadata_marks_entity(self):
        queries, metadata = _ensure_bilingual_queries(
            ["some generic query"],
            "What about X?",
            "general user query",
            query_type="entity",
            disambiguation_needed=False,
        )
        assert queries == ["some generic query"]
        assert metadata["entity_query_expected"] is True

    async def test_metadata_marks_disambiguation(self):
        _, metadata = _ensure_bilingual_queries(
            ["analytic topic analysis"],
            "Framework comparison",
            "compare frameworks",
            query_type="factual",
            disambiguation_needed=True,
        )
        assert metadata["entity_query_expected"] is True

    async def test_metadata_keeps_analytic(self):
        _, metadata = _ensure_bilingual_queries(
            ["framework comparison analysis"],
            "Framework comparison analysis",
            "compare frameworks",
            query_type="factual",
            disambiguation_needed=False,
        )
        assert metadata["entity_query_expected"] is False


class TestEntityAnchor:
    def test_cjk_anchor_extracted(self):
        # Build CJK string via codepoints to avoid raw Chinese in code.
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])  # two-char CJK name
        question = f"{name} Apache RocketMQ"
        assert _extract_entity_anchor(question, "") == name

    def test_shortest_cjk_segment(self):
        seg2 = "".join(chr(cp) for cp in [0x5F20, 0x4E09])  # 2-char
        seg4 = "".join(chr(cp) for cp in [0x6280, 0x672F, 0x6846, 0x67B6])  # 4-char
        question = f"{seg4} {seg2} open-source"
        assert _extract_entity_anchor(question, "") == seg2

    def test_english_anchor(self):
        result = _extract_entity_anchor("What is John Doe doing now", "")
        assert "John" in result or "Doe" in result

    def test_empty_input(self):
        assert _extract_entity_anchor("", "") == ""

    def test_strips_ascii_possessive(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        question = f"What is {name}'s current employer"
        anchor = _extract_entity_anchor(question, "")
        assert anchor == name
        assert "'" not in anchor

    def test_strips_curly_possessive(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        question = f"What is {name}\u2019s current role"
        anchor = _extract_entity_anchor(question, "")
        assert anchor == name


class TestPrependFacetQueries:
    def test_identity_facet_skipped(self):
        contract = AnswerCoverageContract(
            must_cover_facets=[
                CoverageFacet(name="identity", intent="confirm entity"),
            ]
        )
        result = _prepend_facet_queries(
            ["existing query"],
            contract,
            question="Who is X",
            user_query="Who is X",
        )
        assert result == ["existing query"]

    def test_facet_composed_with_anchor(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        contract = AnswerCoverageContract(
            must_cover_facets=[
                CoverageFacet(
                    name="current role",
                    bilingual_terms=["Apache RocketMQ"],
                ),
            ]
        )
        result = _prepend_facet_queries(
            ["base"],
            contract,
            question=f"{name} background",
            user_query=f"{name}",
            entity_query_expected=True,
        )
        composed = result[0]
        assert name in composed
        assert "Apache" in composed or "RocketMQ" in composed

    def test_meta_terms_filtered(self):
        contract = AnswerCoverageContract(
            must_cover_facets=[
                CoverageFacet(
                    name="background",
                    bilingual_terms=["confirm identity", "distinguish candidates"],
                ),
            ]
        )
        result = _prepend_facet_queries(
            ["q1"],
            contract,
            question="About X",
            user_query="About X",
        )
        # Meta terms should be filtered; only "q1" remains.
        assert result == ["q1"]


class TestEntityComposition:
    def test_bare_topic_gets_anchor(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        result = _enforce_entity_composition(
            ["current employer"],
            question=f"{name} background",
            user_query=f"{name}",
            entity_query_expected=True,
        )
        assert len(result) == 1
        assert name in result[0]
        assert "current employer" in result[0]

    def test_existing_anchor_unchanged(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        original = f"{name} Apache RocketMQ"
        result = _enforce_entity_composition(
            [original],
            question=f"{name} background",
            user_query=f"{name}",
            entity_query_expected=True,
        )
        assert result == [original]

    def test_non_entity_query_unchanged(self):
        result = _enforce_entity_composition(
            ["general analysis"],
            question="framework comparison",
            user_query="compare frameworks",
            entity_query_expected=False,
        )
        assert result == ["general analysis"]
