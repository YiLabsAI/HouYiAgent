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
