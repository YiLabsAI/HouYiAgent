from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.router import (
    CascadingRouter,
    QueryRouter,
    RouteDecision,
    Tier0RuleRouter,
    Tier1SemanticRouter,
    Tier2LLMRouter,
)
from houyi.adapters.memory.recall.types import QueryType, RecallQuery


@pytest.mark.asyncio
async def test_rule_router_classifies() -> None:
    router = Tier0RuleRouter()

    cases = [
        # Chinese yes/no preference question.
        ("\u4ed6\u559c\u6b22\u5496\u5561\u5417", QueryType.NEGATION_CHECK),
        ("phone of Martin", QueryType.FACTUAL_LOOKUP),
        ("when did we meet", QueryType.TEMPORAL_QUERY),
        ("email of manager of Martin", QueryType.RELATIONAL_CHAIN),
        ("how to fix cache", QueryType.PROCEDURAL_RECALL),
    ]

    for text, expected in cases:
        decision = await router.classify(RecallQuery(text=text))
        assert decision.query_type == expected


@pytest.mark.asyncio
async def test_rule_router_timeunit_question() -> None:
    """ "(in) which/what <time-unit>" routes to temporal, not factual/default.

    "In which month did X happen" asks for a point on the timeline; without
    the dedicated pattern the wh-stem fell to the generic factual rule (or the
    thematic default) and lost the timeline-priority route. Single-fact and
    aggregation wh-questions without a time-unit must stay factual_lookup.
    """
    router = Tier0RuleRouter()

    cases = [
        ("In which month's game did John achieve a career-high score?", QueryType.TEMPORAL_QUERY),
        ("Which year did Caroline graduate?", QueryType.TEMPORAL_QUERY),
        # No time-unit: stay factual_lookup.
        ("What might John's financial status be?", QueryType.FACTUAL_LOOKUP),
        ("What books has Tim read?", QueryType.FACTUAL_LOOKUP),
    ]

    for text, expected in cases:
        decision = await router.classify(RecallQuery(text=text))
        assert decision.query_type == expected, f"{text!r} -> {decision.query_type}"


@pytest.mark.asyncio
async def test_cascade_defaults() -> None:
    router = CascadingRouter(Tier0RuleRouter())

    decision = await router.classify(RecallQuery(text="tell me about recent work"))

    assert decision.query_type == QueryType.THEMATIC_SUMMARY
    assert decision.tier == "default"


class FixedRouter(QueryRouter):
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.calls = 0

    async def classify(self, query: RecallQuery) -> RouteDecision:
        self.calls += 1
        return self.decision


@pytest.mark.asyncio
async def test_cascade_uses_tier1() -> None:
    tier0 = FixedRouter(RouteDecision(query_type=QueryType.THEMATIC_SUMMARY, confidence=0.1))
    tier1 = FixedRouter(
        RouteDecision(query_type=QueryType.TEMPORAL_QUERY, confidence=0.9, tier="semantic")
    )
    router = CascadingRouter(tier0, tier1=tier1)

    decision = await router.classify(RecallQuery(text="latest address"))

    assert decision.query_type == QueryType.TEMPORAL_QUERY
    assert decision.tier == "semantic"
    assert tier0.calls == 1
    assert tier1.calls == 1


@pytest.mark.asyncio
async def test_cascade_skips_tier1() -> None:
    tier0 = FixedRouter(RouteDecision(query_type=QueryType.FACTUAL_LOOKUP, confidence=0.8))
    tier1 = FixedRouter(
        RouteDecision(query_type=QueryType.TEMPORAL_QUERY, confidence=1.0, tier="semantic")
    )
    router = CascadingRouter(tier0, tier1=tier1)

    decision = await router.classify(RecallQuery(text="phone of Martin"))

    assert decision.query_type == QueryType.FACTUAL_LOOKUP
    assert tier1.calls == 0


@pytest.mark.asyncio
async def test_cascade_uses_tier2() -> None:
    tier0 = FixedRouter(RouteDecision(query_type=QueryType.THEMATIC_SUMMARY, confidence=0.1))
    tier1 = FixedRouter(
        RouteDecision(query_type=QueryType.THEMATIC_SUMMARY, confidence=0.2, tier="semantic")
    )
    tier2 = FixedRouter(
        RouteDecision(query_type=QueryType.PROCEDURAL_RECALL, confidence=1.0, tier="llm")
    )
    router = CascadingRouter(tier0, tier1=tier1, tier2=tier2)

    decision = await router.classify(RecallQuery(text="ambiguous"))

    assert decision.query_type == QueryType.PROCEDURAL_RECALL
    assert decision.tier == "llm"


@pytest.mark.asyncio
async def test_semantic_defaults() -> None:
    decision = await Tier1SemanticRouter().classify(RecallQuery(text="anything"))

    assert decision.query_type == QueryType.THEMATIC_SUMMARY
    assert decision.confidence == 0.0
    assert decision.tier == "semantic"


@pytest.mark.asyncio
async def test_llm_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        await Tier2LLMRouter().classify(RecallQuery(text="anything"))


def test_cascade_requires_tier0() -> None:
    with pytest.raises(ValueError):
        CascadingRouter(None)  # type: ignore[arg-type]
