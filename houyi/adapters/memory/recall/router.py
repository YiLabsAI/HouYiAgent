"""Three-tier cascading query router.

The router classifies an incoming RecallQuery into exactly
one QueryType so dispatch can choose a single primary
retriever. Fan-out is reserved for explicit fusion, not routing
ambiguity.

Three tiers, in increasing cost order:

- Tier 0 (rule) — regex + keyword dictionaries. ~70% of LoCoMo
 + HaluMem-style queries hit cleanly here in ≤ 5ms. Implemented as
 Tier0RuleRouter.
- Tier 1 (semantic) — prototype-utterance embedding cosine. Lifts
 cumulative coverage to ~95% in 5-30ms. Wired as a placeholder until
 an embedding-backed implementation is enabled.
- Tier 2 (LLM) — small-model classifier (gpt-4o-mini / qwen-turbo)
 for the residual ~5% ambiguity. Disabled by default —
 CascadingRouter requires explicit tier2 injection so
 the cost path stays opt-in.

Default config:
CascadingRouter(tier0=Tier0RuleRouter()) falls through to the
thematic_summary default when T0 finds no high-confidence match.

This module deliberately holds no LLM dependency; the LLM tier is
plugged in through the abstract QueryRouter interface so unit
tests for T0 cannot accidentally hit a network call.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from houyi.adapters.memory.recall.types import QueryType, RecallQuery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision value object
# ---------------------------------------------------------------------------

RouteTier = Literal["rule", "semantic", "llm", "default"]
"""Which tier produced the decision; surfaced in trace logs."""


class RouteDecision(BaseModel):
    """One classification verdict.

    The router emits exactly one of these per query; the orchestrator
    uses query_type to look up the routing-table row,
    while confidence and tier flow into trace logs for
    post-hoc routing-quality analysis (e.g. how often does T0
    misclassify negation_check as factual_lookup).
    """

    model_config = ConfigDict(frozen=True)

    query_type: QueryType
    """The classified type; always non-null (THEMATIC_SUMMARY is the default)."""

    confidence: float = 0.0
    """0..1 — how strongly the producing tier believes in this label."""

    tier: RouteTier = "rule"
    """Which tier produced the decision; useful for cost / quality tracking."""

    matched_pattern: str | None = None
    """The regex source / prototype id that fired, when applicable."""

    reasoning: str = ""
    """Human-readable one-liner; never used for routing logic."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class QueryRouter(ABC):
    """Abstract base for any tier of the cascading router.

    All tiers share the same surface so CascadingRouter can
    chain them generically and tests can mix-and-match (e.g. real T0 +
    fake T1 returning a fixed type).
    """

    @abstractmethod
    async def classify(self, query: RecallQuery) -> RouteDecision:
        """Return the tier's verdict for query.

        Implementations MUST always return a RouteDecision;
        when they cannot classify with sufficient confidence they
        return one with confidence below their advertised
        threshold so the cascade can move to the next tier.
        """
        raise NotImplementedError  # pragma: no cover - abstract


# ---------------------------------------------------------------------------
# Tier 0 — regex rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pattern:
    """One regex rule with its target type and strength.

    Strength is a deliberate per-pattern weight rather than per-type,
    because some patterns are more diagnostic than others.
    """

    query_type: QueryType
    regex: re.Pattern[str]
    strength: float
    label: str  # human-readable id for trace


# Order of rule groups follows the "specific before generic"
# preference: NEGATION_CHECK before FACTUAL_LOOKUP because both can
# match on attribute words; TEMPORAL_QUERY before everything except
# negation because temporal phrasing wraps any other intent.
#
# Strengths are calibrated so a single high-strength match (>= 0.85)
# beats multiple medium matches; this keeps the cascade simple
# (highest-strength wins) without an explicit priority list.

_TIER_0_PATTERNS: list[_Pattern] = [
    # NEGATION_CHECK - yes/no presence questions.
    _Pattern(
        QueryType.NEGATION_CHECK,
        re.compile(
            "\u662f\u4e0d\u662f|\u6709\u6ca1\u6709|\u4e0d\u662f.*\u5417|"
            "\u662f\u5426|\u5bf9\u5417|\u5bf9\u4e0d\u5bf9|.+\u5417[??]?$",
            re.IGNORECASE,
        ),
        0.92,
        "zh_negation",
    ),
    _Pattern(
        QueryType.NEGATION_CHECK,
        # English yes/no inversion: starts with aux-verb, not a wh-word.
        # Anchored to the start (after optional whitespace) so factual
        # queries like "what is your name?" do not falsely trigger on
        # the embedded "is".
        re.compile(
            r"^\s*(?:do(?:es)?|did|is|are|was|were|has|have|can|could|will|would|should)\b.+\?",
            re.IGNORECASE,
        ),
        0.70,  # English yes/no is weaker because it overlaps factual queries
        "en_yesno",
    ),
    # TEMPORAL_QUERY — time words / order words / dates
    _Pattern(
        QueryType.TEMPORAL_QUERY,
        re.compile(
            "\u4ec0\u4e48\u65f6\u5019|\u4f55\u65f6|\u4e0a\u6b21|"
            "\u53bb\u5e74|\u4eca\u5e74|\u6628\u5929|\u524d\u5929|"
            "\u591a\u4e45(?:\u4e4b\u524d|\u524d)|\u6700\u8fd1|"
            "\u6700\u521d|\u4ece\u6b64",
            re.IGNORECASE,
        ),
        0.90,
        "zh_temporal",
    ),
    _Pattern(
        QueryType.TEMPORAL_QUERY,
        re.compile(
            r"\b(?:when|before|after|since|until|earliest|latest|last|first)\b",
            re.IGNORECASE,
        ),
        0.88,
        "en_temporal",
    ),
    _Pattern(
        QueryType.TEMPORAL_QUERY,
        re.compile(
            r"\d{4}\u5e74|\d{1,2}\u6708|\d+\s*years?\s*ago|\d+\s*\u5929\u524d",
            re.IGNORECASE,
        ),
        0.95,
        "date_literal",
    ),
    _Pattern(
        QueryType.TEMPORAL_QUERY,
        # "(in) which/what <time-unit>" — the answer is a point on the
        # timeline ("in which month did X happen"). Without this the wh-stem
        # falls to the generic factual pattern (or default) and loses the
        # timeline-priority route. Strength edges out en_wh_question (0.85).
        re.compile(
            r"\b(?:in\s+)?(?:which|what)\s+(?:month|year|day|week|date|season)\b",
            re.IGNORECASE,
        ),
        0.90,
        "en_which_timeunit",
    ),
    # RELATIONAL_CHAIN - multi-hop ownership/attribute chains.
    _Pattern(
        QueryType.RELATIONAL_CHAIN,
        re.compile(r"(\S+)\u7684(\S+)\u7684(\S+)"),
        0.95,
        "zh_chain_3hop",
    ),
    _Pattern(
        QueryType.RELATIONAL_CHAIN,
        re.compile("\u901a\u8fc7.+\u7136\u540e.+|\u5148.+\u518d.+|\u7ecf\u7531.+\u5230\u8fbe"),
        0.85,
        "zh_chain_via",
    ),
    _Pattern(
        QueryType.RELATIONAL_CHAIN,
        re.compile(
            r"\b(?:through|via|because|caused by|leads? to|depends? on|relate[sd]? to)\b",
            re.IGNORECASE,
        ),
        0.80,
        "en_chain",
    ),
    _Pattern(
        QueryType.RELATIONAL_CHAIN,
        re.compile(r"\b\w+\s+of\s+\w+\s+of\s+\w+\b", re.IGNORECASE),
        0.96,
        "en_of_chain",
    ),
    # PROCEDURAL_RECALL — "how to" / strategy reuse
    _Pattern(
        QueryType.PROCEDURAL_RECALL,
        re.compile(
            "^\u600e\u4e48|^\u5982\u4f55|^\u600e\u6837|"
            "\u600e\u4e48\u505a|\u5982\u4f55\u505a|\u8be5\u600e\u4e48"
        ),
        0.88,
        "zh_howto",
    ),
    _Pattern(
        QueryType.PROCEDURAL_RECALL,
        re.compile(r"\bhow\s+(?:to|do|did|can|should)\b", re.IGNORECASE),
        0.88,
        "en_howto",
    ),
    # FACTUAL_LOOKUP - single-entity attribute query.
    _Pattern(
        QueryType.FACTUAL_LOOKUP,
        re.compile(
            r"^(?:\u8c01\u662f|\u4ec0\u4e48\u662f|\u54ea\u91cc\u662f|"
            r"\u54ea\u4f4d\u662f)\S+[??]?$"
        ),
        0.92,
        "zh_who_what_where",
    ),
    _Pattern(
        QueryType.FACTUAL_LOOKUP,
        re.compile(
            r"\S+\s*\u7684\s*(?:\u7535\u8bdd|\u90ae\u7bb1|\u5730\u5740|"
            r"\u751f\u65e5|\u804c\u4e1a|\u540d\u5b57|\u5e74\u9f84|"
            r"\u8eab\u4efd|\u516c\u53f8)"
        ),
        0.95,
        "zh_attribute",
    ),
    _Pattern(
        QueryType.FACTUAL_LOOKUP,
        # Wh-question stems at the start of the query. We deliberately
        # do NOT require an immediate copula ("what is …") — natural
        # phrasings like "what time is it" / "whose phone is this" put
        # the noun phrase before the verb, and a stricter pattern was
        # observed (during T0 smoke testing) to drop them to default.
        # how is handled separately by the procedural pattern.
        re.compile(
            r"^(?:who|what|where|whose|which|why)\b[^?]*\??$",
            re.IGNORECASE,
        ),
        0.85,
        "en_wh_question",
    ),
    _Pattern(
        QueryType.FACTUAL_LOOKUP,
        re.compile(
            r"\b(?:phone|email|address|birthday|job|name|age|company)\s+of\b",
            re.IGNORECASE,
        ),
        0.92,
        "en_attribute",
    ),
]


class Tier0RuleRouter(QueryRouter):
    """Regex + keyword classifier; no LLM, ≤ 5ms per query.

    Behavior:

    - Iterates over _TIER_0_PATTERNS and collects every match.
    - Returns the single highest-strength match.
    - When nothing matches, returns THEMATIC_SUMMARY with
    confidence=0.0 and tier="default" so the cascade can
    decide whether to escalate.

    Custom patterns can be injected via __init__ for tests or
    for caller-specific vocabulary (e.g. a code-domain caller might
    add r"\\bfunction\\s+\\w+" to bias toward
    PROCEDURAL_RECALL); injected patterns are appended after
    defaults so test patterns do not silently mask production ones.
    """

    def __init__(self, extra_patterns: Iterable[_Pattern] | None = None) -> None:
        self._patterns: list[_Pattern] = list(_TIER_0_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    async def classify(self, query: RecallQuery) -> RouteDecision:
        text = query.text.strip()
        # Track the strongest hit; tie-break by appearance order
        # (deterministic for tests).
        best: tuple[_Pattern, re.Match[str]] | None = None
        for pat in self._patterns:
            m = pat.regex.search(text)
            if not m:
                continue
            if best is None or pat.strength > best[0].strength:
                best = (pat, m)

        if best is None:
            return RouteDecision(
                query_type=QueryType.THEMATIC_SUMMARY,
                confidence=0.0,
                tier="default",
                matched_pattern=None,
                reasoning="no T0 rule matched; default to thematic_summary",
            )

        pat, _match = best
        return RouteDecision(
            query_type=pat.query_type,
            confidence=pat.strength,
            tier="rule",
            matched_pattern=pat.label,
            reasoning=f"T0 rule {pat.label!r} matched",
        )


# ---------------------------------------------------------------------------
# Tier 1 — semantic prototype router (placeholder)
# ---------------------------------------------------------------------------


class Tier1SemanticRouter(QueryRouter):
    """Prototype-cosine classifier placeholder.

    The placeholder avoids bringing an embedding dependency into the
    baseline path. It always returns a low-confidence
    THEMATIC_SUMMARY so CascadingRouter naturally falls
    through to T2 (when present) or to the default.

    A full implementation can embed a small set of curated prototype
    utterances per QueryType once at construction; cosine
    against the query embedding yields the type with peak similarity.
    Until then, callers must not rely on T1 producing useful output.
    """

    async def classify(self, query: RecallQuery) -> RouteDecision:
        return RouteDecision(
            query_type=QueryType.THEMATIC_SUMMARY,
            confidence=0.0,
            tier="semantic",
            matched_pattern=None,
            reasoning="Tier1SemanticRouter placeholder",
        )


# ---------------------------------------------------------------------------
# Tier 2 — LLM classifier (placeholder, opt-in)
# ---------------------------------------------------------------------------


class Tier2LLMRouter(QueryRouter):
    """Small-LLM classifier for the residual ambiguous ~5%.

    This placeholder exists for callers that want to opt into a small
    model classifier after query-stream telemetry shows which patterns
    fall through T0 + T1.

    The stub raises NotImplementedError rather than returning
    a low-confidence default so that a caller who explicitly opted
    in to T2 (by passing it to CascadingRouter) gets a
    visible failure rather than a silent thematic-summary
    misclassification — being silent here would defeat the purpose of
    paying T2's cost.
    """

    def __init__(self, llm_adapter: object | None = None) -> None:
        self._llm = llm_adapter

    async def classify(self, query: RecallQuery) -> RouteDecision:
        raise NotImplementedError(
            "Tier2LLMRouter is a placeholder; wire a small-LLM "
            "classifier (gpt-4o-mini / qwen-turbo) before opting in."
        )


# ---------------------------------------------------------------------------
# Cascading router
# ---------------------------------------------------------------------------


class CascadingRouter(QueryRouter):
    """T0 → T1 → T2 cascade with explicit thresholds.

    A tier is consulted only if all earlier tiers produced a decision
    with confidence < threshold; the first tier whose confidence
    meets its threshold wins. When all tiers fall short the router
    returns the last decision (i.e. the most informed guess) but
    flagged with tier="default" so downstream knows to be skeptical.

    Defaults use tier0_threshold=0.7 and tier1_threshold=0.7.
    The LLM tier is opt-in (tier2=None
    by default) so the cost path cannot be entered by accident.
    """

    def __init__(
        self,
        tier0: QueryRouter,
        tier1: QueryRouter | None = None,
        tier2: QueryRouter | None = None,
        *,
        tier0_threshold: float = 0.7,
        tier1_threshold: float = 0.7,
    ) -> None:
        if tier0 is None:
            raise ValueError("CascadingRouter requires at least tier0")
        self._tier0 = tier0
        self._tier1 = tier1
        self._tier2 = tier2
        self._t0 = tier0_threshold
        self._t1 = tier1_threshold

    async def classify(self, query: RecallQuery) -> RouteDecision:
        decision = await self._tier0.classify(query)
        if decision.confidence >= self._t0:
            return decision

        if self._tier1 is not None:
            t1 = await self._tier1.classify(query)
            if t1.confidence >= self._t1:
                return t1
            decision = t1  # keep most-informed guess

        if self._tier2 is not None:
            t2 = await self._tier2.classify(query)
            # T2 has no threshold; whatever it says, take it.
            return t2

        # Nothing met the threshold — return the last guess but
        # clearly mark it as default so trace consumers can spot
        # "no tier was confident" runs.
        return RouteDecision(
            query_type=decision.query_type,
            confidence=decision.confidence,
            tier="default",
            matched_pattern=decision.matched_pattern,
            reasoning=(
                f"all tiers below threshold; falling back to "
                f"{decision.query_type.value!r} from {decision.tier!r}"
            ),
        )


__all__ = [
    "CascadingRouter",
    "QueryRouter",
    "RouteDecision",
    "RouteTier",
    "Tier0RuleRouter",
    "Tier1SemanticRouter",
    "Tier2LLMRouter",
]
