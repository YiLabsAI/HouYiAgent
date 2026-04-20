"""Shared taxonomy constants for research query and section classification.

Consolidates keyword-hint tuples that were previously scattered across
planner.py, search_query_planner.py, validation.py, and report.py.
All downstream modules import from here to avoid duplication and make
the hint vocabulary maintainable in one place.

Long-term these keyword hints should be replaced by planner-output
structured metadata (query_type, disambiguation_needed, section_archetype)
which are now the primary classification signal.  The hints below serve
as a zero-latency fallback when the planner does not populate those fields.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Query-type classification hints (fallback for planner.query_type)
# ---------------------------------------------------------------------------

# Keywords suggesting an entity-oriented question (person, org, project).
ENTITY_QUERY_HINTS: tuple[str, ...] = (
    "who is",
    "profile",
    "biography",
    "background",
    "current role",
    "employer",
)

# CJK tokens that indicate an analytic (non-entity) topic, used to
# suppress false-positive entity detection on Chinese queries.
ANALYTIC_TOPIC_CJK_HINTS: tuple[str, ...] = (
    "\u6846\u67b6",  # framework
    "\u5bf9\u6bd4",  # comparison
    "\u6bd4\u8f83",  # compare
    "\u8d8b\u52bf",  # trend
    "\u5206\u6790",  # analysis
    "\u65b9\u6cd5",  # method
    "\u7cfb\u7edf",  # system
    "\u6280\u672f",  # technology
    "\u5e02\u573a",  # market
    "\u5f71\u54cd",  # impact
    "\u7814\u7a76",  # research
    "\u6a21\u578b",  # model
)

# English equivalents.
ANALYTIC_TOPIC_EN_HINTS: tuple[str, ...] = (
    "framework",
    "comparison",
    "compare",
    "trend",
    "analysis",
    "method",
    "system",
    "technology",
    "market",
    "impact",
    "research",
    "model",
)

# ---------------------------------------------------------------------------
# Interrogative connector vocabulary (used to convert sub-questions into
# declarative section titles when the outline expansion path runs).
# ---------------------------------------------------------------------------

# CJK interrogative connectors that appear between a topical prefix and the
# object of the question, e.g. "<prefix><connector><object>".  Removing them
# yields a declarative phrase suitable for a section heading.
CJK_INTERROGATIVE_CONNECTORS: tuple[str, ...] = (
    "\u6709\u54ea\u4e9b",  # "what (kinds) are there"
    "\u662f\u4ec0\u4e48",  # "what is"
    "\u662f\u591a\u5c11",  # "how much / how many"
    "\u662f\u600e\u6837",  # "how is / what is like"
    "\u600e\u4e48\u6837",  # "how"
    "\u5728\u54ea\u91cc",  # "where"
    "\u5982\u4f55",  # "how"
    "\u600e\u6837",  # "how"
)

# English interrogative leads: "<lead> <rest>" -> "<rest>".  Matched at the
# start of the string only to avoid chopping substantive prefixes.
ENGLISH_INTERROGATIVE_LEADS: tuple[str, ...] = (
    # (lead_word, optional_follow) pairs.  empty follow means the lead
    # itself is stripped when it starts the phrase.
    "what",
    "who",
    "how",
    "why",
    "when",
    "where",
    "which",
)

# ---------------------------------------------------------------------------
# Identity / disambiguation hints (fallback for planner.disambiguation_needed)
# ---------------------------------------------------------------------------

# Markers that appear in coverage-contract text when identity evidence is needed.
IDENTITY_SOURCE_MARKERS: tuple[str, ...] = (
    "official",
    "profile",
    "bio",
    "homepage",
    "documentation",
    "github",
    "repository",
    "repo",
)

# Markers used to detect whether an English query already targets official sources.
ENGLISH_OFFICIAL_MARKERS: tuple[str, ...] = (
    "benchmark",
    "official",
    "report",
    "paper",
    "dataset",
    "documentation",
    "profile",
    "github",
    "linkedin",
)

# Hints for validation-layer identity context detection.
IDENTITY_CONTEXT_HINTS: tuple[str, ...] = (
    "identity",
    "same-name",
    "official profile",
    "employer",
    "current role",
)

# ---------------------------------------------------------------------------
# Section archetype classification hints (fallback for outline.section_archetype)
# ---------------------------------------------------------------------------

# Each tuple maps an archetype to keywords found in coverage-contract text.
ARCHETYPE_COMPARISON_HINTS: tuple[str, ...] = (
    "compare",
    "comparison",
    "versus",
    "vs",
    "trade-off",
)

ARCHETYPE_RISK_HINTS: tuple[str, ...] = (
    "risk",
    "limitation",
    "caveat",
    "uncertainty",
    "constraint",
)

ARCHETYPE_TREND_HINTS: tuple[str, ...] = (
    "trend",
    "timeline",
    "evolution",
    "history",
    "current",
)

# ---------------------------------------------------------------------------
# Counter-evidence detection hints
# ---------------------------------------------------------------------------

COUNTER_EVIDENCE_MARKERS: tuple[str, ...] = (
    # English markers
    "however",
    "contradict",
    "dispute",
    "critic",
    "limitation",
    "risk",
    "challenge",
    "concern",
    "debate",
    "controversy",
    "drawback",
    "downside",
    "flaw",
    "weakness",
    "skeptic",
    "caveat",
    # CJK equivalents (unicode escapes to satisfy no-Chinese-in-code rule)
    "\u4e89\u8bae",  # zhengyi  - controversy / dispute
    "\u98ce\u9669",  # fengxian - risk
    "\u8d28\u7591",  # zhiyi    - doubt / skepticism
    "\u6279\u8bc4",  # pipan    - criticism
    "\u5c40\u9650",  # juxian   - limitation
    "\u7f3a\u9677",  # quexian  - flaw / defect
    "\u6311\u6218",  # tiaozhan - challenge
    "\u4e0d\u8db3",  # buzu     - shortcoming
    "\u53cd\u5bf9",  # fandui   - opposition
    "\u62c5\u5fe7",  # danyou   - concern
)


# ---------------------------------------------------------------------------
# Sidecar compliance keywords (used by report._compute_section_sidecar_metrics)
# ---------------------------------------------------------------------------
#
# Distinct from ARCHETYPE_*_HINTS above: those detect archetype from coverage
# contract text at planning time; the tables below detect whether the final
# section content uses phrasing consistent with the assigned archetype. They
# are consumed only by the sidecar diagnostic signal and are not part of the
# main scoring path.
ARCHETYPE_COMPLIANCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "comparison": (
        "compared",
        "versus",
        "vs",
        "whereas",
        "unlike",
        "trade-off",
        "advantage",
        "disadvantage",
        "outperform",
    ),
    "risk_and_caveat": (
        "risk",
        "limitation",
        "caveat",
        "however",
        "despite",
        "uncertainty",
        "contested",
        "debated",
    ),
    "trend_and_state": (
        "trend",
        "grew",
        "declined",
        "since",
        "over the past",
        "evolution",
        "trajectory",
        "inflection",
    ),
}


# ---------------------------------------------------------------------------
# Query hygiene vocabularies (consumed by search-executor query cleanup)
# ---------------------------------------------------------------------------
#
# These lexicons let the executor drop noise emitted by the planner, such as:
# * Filler fragments from user prompts that accidentally become keywords
#   ("financial-power etc." was observed five times in a single case-1 run).
# * Pure interrogative/connector tokens that carry no retrieval signal when
#   they form the entire query content.
#
# Membership is matched case-insensitively and after whitespace trimming.
# The hygiene logic must depend only on this file so the vocabulary stays
# maintainable in a single place (consistent with ARCHETYPE_COMPLIANCE_KEYWORDS).

# Tokens that are clearly user-prompt detritus. Any query whose tokens are
# entirely drawn from this set is dropped. Tokens here should be strings the
# planner has no reason to emit as a standalone keyword under any topic.
QUERY_HYGIENE_FILLER_TOKENS: tuple[str, ...] = (
    # "caili dengdeng" - verbatim tail from a user prompt meaning
    # "financial-power etc.", not a search keyword.
    "\u8d22\u529b\u7b49\u7b49",
    # "dengdeng" - lone "etc." marker.
    "\u7b49\u7b49",
    # "shoujizhengli" - verbatim "collect and organize" prompt phrasing.
    "\u6536\u96c6\u6574\u7406",
)

# High-frequency CJK tokens that are interrogatives or generic connectors.
# A query built entirely from these tokens is unfit for retrieval.
QUERY_HYGIENE_CJK_STOPWORDS: tuple[str, ...] = (
    "\u5982\u4f55",  # ruhe - how
    "\u662f\u4ec0\u4e48",  # shi shenme - what is
    "\u4e3a\u4ec0\u4e48",  # weishenme - why
    "\u600e\u6837",  # zenyang - how
    "\u54ea\u4e9b",  # naxie - which ones
    "\u6709\u54ea\u4e9b",  # you naxie - what are there
    "\u76ee\u524d",  # muqian - currently
    "\u5f53\u524d",  # dangqian - currently
    "\u6982\u8ff0",  # gaishu - overview
)

# English equivalents for the same filter.
QUERY_HYGIENE_EN_STOPWORDS: tuple[str, ...] = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "overview",
    "introduction",
    "etc",
    "etc.",
    "currently",
)


# ---------------------------------------------------------------------------
# Section structural contract vocabularies
# ---------------------------------------------------------------------------
#
# These lexicons support a topic-agnostic structural contract applied to each
# generated report section. The goal is to let any research topic benefit from
# the same "critical analysis + data visualisation" discipline without
# hard-coding any specific domain.
#
# Three vocabularies power the contract:
#
# * ``SECTION_CRITICAL_ANALYSIS_KEYWORDS`` - presence of any of these tokens
#   in a section body signals that the author has acknowledged limitations,
#   caveats, or competing interpretations. Postprocess guards attach a debug
#   hint (HTML comment) when none of the tokens are found.
# * ``SECTION_VISUAL_TRIGGER_KEYWORDS`` - hierarchy/sequence cues that justify
#   a mermaid diagram. Postprocess guards attach a visualization hint when the
#   cue is present but no mermaid fence is detected.
# * ``UNIVERSAL_BACKBONE_FACETS`` - the minimal topic-agnostic outline
#   contract. Only two facets are included: concept-framework and
#   controversies-and-caveats. Any further facet would drift into
#   topic-specific territory and is deliberately excluded.
#
# Matching is case-insensitive after lowercase normalisation for ASCII and
# raw-substring for CJK.

SECTION_CRITICAL_ANALYSIS_KEYWORDS: tuple[str, ...] = (
    # English markers
    "limitation",
    "limitations",
    "caveat",
    "caveats",
    "methodological",
    "methodology",
    "competing",
    "dispute",
    "contested",
    "debated",
    "however",
    "uncertainty",
    "trade-off",
    "tradeoff",
    # CJK markers
    "\u5c40\u9650",  # juxian - limitation
    "\u9650\u5236",  # xianzhi - restriction
    "\u53e3\u5f84",  # koujing - caliber/scope
    "\u4e89\u8bae",  # zhengyi - controversy
    "\u5206\u6b67",  # fenqi - divergence
    "\u5dee\u5f02",  # chayi - difference
    "\u4e0d\u786e\u5b9a",  # buqueding - uncertainty
    "\u65b9\u6cd5\u8bba",  # fangfa-lun - methodology
    "\u6279\u8bc4",  # piping - critique
    "\u6743\u8861",  # quanheng - trade-off
)


SECTION_VISUAL_TRIGGER_KEYWORDS: tuple[str, ...] = (
    # English cues for hierarchy / sequence / flow
    "hierarchy",
    "framework",
    "pipeline",
    "workflow",
    "sequence",
    "stages",
    "levels",
    "tiers",
    "taxonomy",
    "architecture",
    # CJK cues
    "\u5c42\u7ea7",  # cengji - hierarchy/level
    "\u9636\u5c42",  # jieceng - stratum
    "\u6846\u67b6",  # kuangjia - framework
    "\u6d41\u7a0b",  # liucheng - flow
    "\u6b65\u9aa4",  # buzhou - step/stage
    "\u9636\u6bb5",  # jieduan - stage/phase
    "\u7ed3\u6784",  # jiegou - structure
    "\u5206\u7c7b",  # fenlei - classification
)


# Minimal topic-agnostic outline contract. Facet names use semantic English
# snake_case; descriptions are intentionally topic-neutral so the contract
# adapts to any research question without case-specific alignment.
#
# Two facets are the maximum that stay unambiguously universal:
# * framework_and_definition - any rigorous report defines its terms and the
#   framework it uses; this hits the Insight rigor dimension.
# * controversies_and_caveats - any rigorous report surfaces dissenting views
#   and known limitations; this hits the Insight critical-analysis dimension.
#
# Additional facets that look "universal" at first glance (stakeholders,
# metrics, timeline...) quickly drift into domain-specific shapes and are
# excluded to keep the contract purely structural.
UNIVERSAL_BACKBONE_FACETS: tuple[dict[str, str], ...] = (
    {
        "name": "framework_and_definition",
        "description": (
            "Introduce the conceptual framework and define the core terms "
            "used across the report, including any competing definitions "
            "that matter for the research question."
        ),
    },
    {
        "name": "controversies_and_caveats",
        "description": (
            "Surface the main controversies, limitations, data caveats, "
            "and competing interpretations relevant to the research "
            "question, so the analysis does not read as one-sided."
        ),
    },
)
