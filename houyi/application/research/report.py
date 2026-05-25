"""ReportGenerator — structured Markdown report with citations.

Generates reports section-by-section via LLM, annotating inline citations
that link back to SourceReference entries. Supports both batch and
streaming (AsyncIterator[ReportChunk]) output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.report_postprocess import (
    SectionPostProcessContext,
    _rewrite_noisy_paragraph_impl,
    clean_noise_step,
    postprocess_section,
)
from houyi.application.research.taxonomy import (
    ARCHETYPE_COMPLIANCE_KEYWORDS,
    COUNTER_EVIDENCE_MARKERS,
    SECTION_CRITICAL_ANALYSIS_KEYWORDS,
    SECTION_VISUAL_TRIGGER_KEYWORDS,
)
from houyi.application.research.types import (
    AggregatedSources,
    AnswerCoverageContract,
    Citation,
    OutlineSection,
    ReportChunk,
    ReportChunkType,
    ReportMetadata,
    ReportSection,
    ReportStyle,
    ResearchPlan,
    ResearchReport,
    SourceReference,
)

logger = logging.getLogger(__name__)

_SECTION_INPUT_METRICS_KEY = "section_input_metrics"

# ---------------------------------------------------------------------------
# Default limits for section source injection and intermediate context.
# Kept as module-level constants so existing callers without explicit
# config get the same behaviour.  ReportGenerator accepts overrides via
# constructor parameters.
# ---------------------------------------------------------------------------

# Maximum sources injected into a single section's LLM prompt.  Ranked by
# relevance; only the top-K are included to control prompt size and cost.
_DEFAULT_MAX_SECTION_SOURCES = 8

# Hard cap on sources formatted into the prompt text (defence-in-depth
# against unexpectedly large ranked lists).
_DEFAULT_MAX_SOURCE_DISPLAY = 20

# Maximum characters kept per source snippet in the section prompt.
_DEFAULT_SNIPPET_MAX_CHARS = 200

# Total / per-question character caps for intermediate-report context
# injected alongside sources.  Keeps prompt inflation predictable.
_DEFAULT_INTERMEDIATE_CONTEXT_MAX_CHARS = 2400
_DEFAULT_INTERMEDIATE_PER_QUESTION_MAX_CHARS = 800

# Minimum question-aligned candidates before the global source pool is
# mixed in.  Set equal to _DEFAULT_MAX_SECTION_SOURCES so every section
# has at least a full selection window of candidates; when fewer are
# available from the question mapping the global pool fills the gap.
# This prevents "citation deserts" where sections only see a handful of
# low-relevance sources from their aligned questions.
_SECTION_SOURCE_FLOOR = _DEFAULT_MAX_SECTION_SOURCES

# Default max output tokens for a single section generation LLM call.
# Covers both the Markdown body and the JSON citation envelope. At roughly
# 1.5 CJK chars or 0.75 English words per token, 2000 tokens comfortably
# fits a section of ~400-800 words plus the citation overhead.
_DEFAULT_SECTION_MAX_TOKENS = 2000

# ---------------------------------------------------------------------------
# Post-generation noise detection.  Regex patterns that flag paragraphs
# containing search-process narration, same-name dump lists, or prose
# without any inline citation.  Detection is zero-LLM-cost; only flagged
# paragraphs trigger a targeted micro-rewrite.
# ---------------------------------------------------------------------------

# Patterns indicating retrieval-process narration that should not appear
# in the final report prose.
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    # Retrieval-process narration.
    re.compile(r"search(ed|ing)?\s+(for|the web|online|results)", re.IGNORECASE),
    re.compile(r"(no|few|limited)\s+results?\s+(were\s+)?found", re.IGNORECASE),
    re.compile(r"(query|queries|retrieval|crawl|scrape)", re.IGNORECASE),
    re.compile(
        r"multiple\s+(people|individuals|entities)\s+(share|with)\s+(the\s+)?(same|this)\s+name",
        re.IGNORECASE,
    ),
    re.compile(r"disambiguation\s+page", re.IGNORECASE),
    # Thinking-trajectory / chain-of-thought leaks.
    re.compile(r"^(let me|I need to|I should|I will|I'll)\s", re.IGNORECASE),
    re.compile(r"^(first,?\s+I|next,?\s+I|now,?\s+I)", re.IGNORECASE),
    re.compile(r"(based on my analysis|upon (my )?reflection|after review)", re.IGNORECASE),
    re.compile(
        r"(the (key )?takeaway (here )?is|to summarize my (thought|finding))", re.IGNORECASE
    ),
    re.compile(r"(as (an|the) (AI|assistant|researcher),?\s+I)", re.IGNORECASE),
    re.compile(r"(I (can |will )?(observe|note|notice) that)", re.IGNORECASE),
]

# Minimum paragraph length (chars) to consider for citation-absence detection.
# Short paragraphs (headings, transitions) are exempt.
_MIN_PARAGRAPH_CHARS_FOR_CITATION = 80

# ---------------------------------------------------------------------------
# Archetype-specific analysis hints injected into the soft checklist to
# steer the LLM toward archetype-appropriate analytical depth.
# Only non-default archetypes carry hints; overview_and_synthesis relies
# on the base prompt rules.
# ---------------------------------------------------------------------------

_ARCHETYPE_ANALYSIS_HINTS: dict[str, str] = {
    "comparison": (
        "Organize analysis around explicit comparison dimensions "
        "(e.g. cost, performance, adoption, limitations). Identify trade-offs "
        "and conditions under which one approach is preferred over another."
    ),
    "risk_and_caveat": (
        "Identify tensions or contradictions between sources. "
        "Distinguish confirmed risks from speculative concerns. "
        "Note where evidence is limited, contested, or evolving."
    ),
    "trend_and_state": (
        "Structure analysis along a temporal axis where applicable. "
        "Distinguish established trends from emerging signals. "
        "Note inflection points, drivers of change, and current trajectory."
    ),
}

# ---------------------------------------------------------------------------
# Sidecar verification constants.  Used by _compute_section_sidecar_metrics
# to produce deterministic prompt-compliance signals at zero LLM cost.
# Metrics are attached to section_input_metrics for offline analysis.
# ---------------------------------------------------------------------------

# CJK escapes: U+3002 ideographic period "ju-hao",
# U+FF01 fullwidth exclamation "jing-tan-hao",
# U+FF1F fullwidth question "wen-hao".
_SENTENCE_TERMINATORS = re.compile(r"[.!?]\s|[.!?]$|" + "[\u3002\uff01\uff1f]")


_NOISE_REWRITE_PROMPT = """\
The following paragraph from a research report section contains noise \
(search-process narration, same-name dumps, or uncited factual claims). \
Rewrite it to be clean, factual, and well-cited. Keep only claims that \
can be supported by the available sources. If nothing salvageable remains, \
respond with an empty string.

Section context: {title} — {objective}
Available reference IDs: {available_refs}

Noisy paragraph:
{paragraph}

Respond with ONLY the rewritten paragraph text (no JSON, no explanation). \
If the paragraph is entirely noise, respond with exactly: (empty)
"""


@dataclass(frozen=True, slots=True)
class SectionEvidencePolicy:
    """Policy for assembling a compact but more coverage-aware section evidence set."""

    candidate_pool_size: int = 12
    min_domain_diversity: int = 3
    require_content_usable: bool = True
    min_cross_question_sources: int = 2
    min_authority_sources: int = 1
    min_counter_evidence_sources: int = 1


if TYPE_CHECKING:
    from houyi.application.research.runtime.intermediate import IntermediateReport

_SECTION_PROMPT = """\
You are writing a section of an academic-grade research report.

MANDATORY OUTPUT LANGUAGE
-------------------------
Match the language of the report title / query EXACTLY.  If the query \
is in English, the ENTIRE section body — every paragraph, heading, \
list item, table cell, and caption — MUST be in English.  If some \
search evidence is in Chinese, translate factual content into English \
rather than copying Chinese text.  Mixing languages inside a section \
is a hard failure.

Report title context: {query}
Section: {title}
Section objective: {objective}
Section archetype: {section_archetype}
Section position in report: {section_position}
Previous section: {previous_section}
Next section: {next_section}
Related sub-question focus:
{related_questions}
Available sources (reference_id | title | snippet):
{sources_text}
Primary evidence:
{primary_evidence}
Counter-evidence or tension points:
{counter_evidence}

Write the section in Markdown. Rules:
- Do NOT include a heading for this section (the heading is added externally).
- CITATION DISCIPLINE: Every factual claim, statistic, date, or attribution MUST \
have an inline citation as [ref_id]. A paragraph without citations is unacceptable. \
Use multiple citations when claims are supported by multiple sources.
- NARRATIVE CONTROL: Lead each major paragraph or sub-section with the claim or finding first, \
then support it with evidence, and place caveats / uncertainty / boundary conditions after the evidence.
- Keep retrieval noise out of the main narrative: do not narrate search-process artifacts, same-name dump lists, \
or generic source-hunting commentary unless they are directly relevant evidence.
- When entities share the same or similar names, keep the prose anchored to the intended entity only and compress \
irrelevant disambiguation into a short caveat instead of letting it dominate the section.
- Only cite sources from the provided list. Do NOT fabricate reference IDs.
- ANALYSIS DEPTH: Go beyond summarizing — synthesize across sources, identify \
patterns, note contradictions, and provide analytical commentary.
- STRUCTURE: Use **bold** sub-headings (### level) to separate major topics. \
Produce multiple distinct analytical paragraphs; target 6 to 10 substantive \
paragraphs per section and do NOT fold unrelated claims into a single oversized \
paragraph. Use bullet points or tables only when listing distinct items \
(e.g., comparison dimensions, timelines).
- Maintain continuity with the surrounding report structure: do not repeat the previous
  section, and end in a way that keeps the transition to the next section natural.
- Write in the SAME language as the report title / query above. \
If the query is in Chinese, write in Chinese. If English, write in English.
- Use clear, professional, scholarly prose. Aim for 400-800 words per section.
- CRITICAL ANALYSIS: After presenting the main evidence, include a concise \
critical passage that acknowledges data source limitations, methodological \
caveats, or competing interpretations where relevant to this section. Keep it \
proportionate to the evidence above; skip entirely if no meaningful caveat applies.
- DATA PRESENTATION: When quantitative data (figures, ranges, percentages, \
comparisons) is central to this section, prefer a compact markdown table for \
side-by-side comparison over long prose enumerations.
- STRUCTURAL DIAGRAMS: When the section describes a hierarchy, taxonomy, \
pipeline, or sequence of stages, consider a ```mermaid flowchart or graph \
to visualize the relationships.
{soft_checklist}

Respond ONLY with JSON:
{{
  "content": "Markdown content with [ref_id] citations (NO heading)...",
  "citations": [
    {{"reference_id": "ref_xxx", "text_span": "the cited claim", "context": "source excerpt"}}
  ]
}}
"""

_SUMMARY_PROMPT = """\
Summarize the following research report in 200-300 words. \
Highlight key findings and conclusions. \
Write in the same language as the report content below.

Report sections:
{sections_text}

Respond with plain text (no JSON).
"""


# Minimum ASCII-to-CJK letter ratio for the query to be treated as English.
# A few CJK chars in an otherwise English query (e.g. a brand name) should
# not flip the classifier.  Gates the EN-only translate post-pass.
_EN_QUERY_ASCII_RATIO = 0.8

# Sub-agent retries and tool-layer plumbing occasionally leak raw
# runtime artifacts into the final section body: orphan ref_<hex>
# tokens, 30s retry markers, sync flags, and broken mermaid
# fences can survive upstream cleanup and tank readability.  These
# regexes are applied as a deterministic last-mile scrubber after the
# section parse/consolidate pipeline.
#
# _VALID_REF_CITATION keeps legitimate [ref_<hex>] citations
# intact during the orphan-token scrub.  Valid citations always appear
# inside square brackets; bare hex refs in prose are garbage.
_VALID_REF_CITATION = re.compile(r"\[ref_[a-fA-F0-9]{6,}\]")
# Fenced code blocks captured with their body so we can inspect and
# drop blocks whose contents devolved into junk tokens.  The closing
# fence is optional so a truncated tail still matches.
_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)(?:\n```|\Z)", re.DOTALL)
# Orphan ref_<hex> fragments that slipped past the citation
# renumberer.  Only matches the token itself plus an optional leading
# [ or trailing ] so the regex cannot swallow neighbouring
# prose.  Valid [ref_<hex>] citations are masked out before this
# regex runs so only truly-orphan fragments remain.
_ORPHAN_REF_RE = re.compile(r"\[?ref_[a-fA-F0-9]{6,}\]?")
# Multiline {"content": "..."} envelope that survived the per-
# paragraph unwrap because it spans paragraph boundaries.  The two
# patterns below locate the opener and (best-effort) tail so the
# scrubber can UNWRAP the envelope — extracting the inner prose and
# un-escaping JSON string escapes — rather than deleting.  Pure
# deletion would strip legitimate prose when the writer emits a
# malformed, never-closed envelope that still carries valuable body.
_ENVELOPE_OPENER_RE = re.compile(
    r"(?ms)^\{\s*\n?\s*\"content\"\s*:\s*\"",
)
# Matching tail: closing quote, optional citations array, optional
# whitespace/newline, closing brace.  _ENVELOPE_TAIL_RE.search
# is anchored after the opener so stray closing braces elsewhere in
# the body (e.g. inside mermaid diagrams) do not get confused with
# the envelope tail.
_ENVELOPE_TAIL_RE = re.compile(
    r"\"\s*(?:,\s*\"citations\"\s*:\s*\[[^\]]*\])?\s*\n?\s*\}",
)


# Language-consistency gate for EN queries.  The soft prompt rule is
# insufficient when search evidence is Chinese-dominant: the model
# follows the evidence language and can emit substantially CJK output
# on English queries.  When the post-gen CJK ratio exceeds this
# threshold we trigger a
# deterministic translation pass to force the section back into
# English without waiting for a regeneration cycle.
_EN_SECTION_CJK_RATIO_MAX = 0.15

_TRANSLATE_PROMPT = """\
You are translating a research section from Chinese (or mixed Chinese/\
English) into fluent academic English.  The original query was in \
English, so the reader expects an English-only article.

Rules:
- Translate EVERY Chinese character into English; the output must \
contain no CJK characters at all, in BOTH the title and the body.
- Preserve Markdown structure EXACTLY: keep every heading level, \
bullet, table, code block, and inline [ref_id] citation in its \
original position.
- Preserve numeric values, dates, named entities (person names, \
company names, place names) verbatim; supply an English rendering in \
parentheses for named entities on first mention if helpful.
- Do NOT add new facts, claims, or citations beyond what the source \
body contains.
- Do NOT drop any paragraphs or bullet points.
- Use clear, professional, scholarly English.

Section title (translate into English): {title}

Section body:
---
{body}
---

Respond ONLY with JSON:
{{
  "title": "Translated section title (no CJK characters)",
  "content": "Translated Markdown body (NO heading)",
  "citations": []
}}
"""


@dataclass(frozen=True, slots=True)
class SectionEvidenceBundle:
    """Evidence packet passed to section generation.

    The bundle keeps the writing stage grounded in answer obligations rather than
    handing the model an undifferentiated list of sources.
    """

    selected_sources: list[SourceReference]
    primary_evidence: list[SourceReference]
    counter_evidence: list[SourceReference]
    reserve_evidence: list[SourceReference]
    unresolved_gaps: list[str]
    caveat_obligations: list[str]
    coverage_facets: list[str]
    comparison_axes: list[str]
    evidence_expectations: list[str]
    time_scope: str
    geo_scope: str
    section_archetype: str = "overview_and_synthesis"


class ReportGenerator:
    """Generates structured Markdown research reports with citations.

    Phase 1-3: Markdown only. Phase 4 adds PDF/PPTX/DOCX export.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        *,
        max_section_sources: int = _DEFAULT_MAX_SECTION_SOURCES,
        max_source_display: int = _DEFAULT_MAX_SOURCE_DISPLAY,
        snippet_max_chars: int = _DEFAULT_SNIPPET_MAX_CHARS,
        intermediate_context_max_chars: int = _DEFAULT_INTERMEDIATE_CONTEXT_MAX_CHARS,
        intermediate_per_question_max_chars: int = _DEFAULT_INTERMEDIATE_PER_QUESTION_MAX_CHARS,
        section_max_tokens: int = _DEFAULT_SECTION_MAX_TOKENS,
        section_evidence_policy: SectionEvidencePolicy | None = None,
        # Max concurrent section generation tasks.  Typical reports have
        # 5-7 sections; default 8 fires them all in one batch.  Lower this
        # if the LLM API has concurrency limits.
        section_concurrency: int = 8,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._llm_kwargs = llm_kwargs
        self._max_section_sources = max_section_sources
        self._max_source_display = max_source_display
        self._snippet_max_chars = snippet_max_chars
        self._intermediate_context_max_chars = intermediate_context_max_chars
        self._intermediate_per_question_max_chars = intermediate_per_question_max_chars
        self._section_max_tokens = section_max_tokens
        self._section_evidence_policy = section_evidence_policy or SectionEvidencePolicy(
            candidate_pool_size=max(max_section_sources + 2, max_section_sources * 2)
        )
        self._section_concurrency = section_concurrency

    async def generate(
        self,
        plan: ResearchPlan,
        sources: AggregatedSources,
        style: ReportStyle = ReportStyle.DETAILED,
        intermediate_reports: list[IntermediateReport] | None = None,
        defer_summary: bool = False,
    ) -> tuple[ResearchReport, dict[str, Any]]:
        """Generate a complete report (non-streaming).

        Iterates over plan outline sections, calling the LLM once per section
        to produce content with inline citations.  When *intermediate_reports*
        are provided (standard/deep depth), the pre-analysed findings are
        injected as additional context to improve citation fidelity and reduce
        hallucinations.

        When *defer_summary* is True, the report is returned with an empty
        summary.  Call complete_summary later to fill it in.  This
        allows the caller to overlap summary generation with other work
        (e.g. URL validation) since that stage does not read the summary
        field.

        Returns:
            Tuple of (ResearchReport, metrics_dict) where metrics_dict contains
            timing fields and section input observability.
        """
        if not plan.outline:
            raise ValueError("Cannot generate report without outline sections")

        ir_by_qid: dict[str, IntermediateReport] = {}
        if intermediate_reports:
            for ir in intermediate_reports:
                ir_by_qid[ir.question_id] = ir

        start = time.monotonic()
        report_id = f"rpt_{uuid.uuid4().hex[:8]}"

        sem = asyncio.Semaphore(self._section_concurrency)

        async def _gen(outline_sec: OutlineSection) -> tuple[ReportSection, dict[str, Any]]:
            async with sem:
                section_index = plan.outline.index(outline_sec)
                bundle, relevant_total, evidence_metrics = self.build_section_evidence_bundle(
                    outline_sec.related_question_ids,
                    sources,
                    section_title=outline_sec.title,
                    objective=outline_sec.objective,
                    coverage_contract=outline_sec.coverage_contract,
                    section_archetype=outline_sec.section_archetype,
                )
                ir_context = _intermediate_context(
                    outline_sec.related_question_ids,
                    ir_by_qid,
                    per_question_cap=self._intermediate_per_question_max_chars,
                    total_cap=self._intermediate_context_max_chars,
                )
                section = await self._generate_section(
                    plan.query,
                    outline_sec.title,
                    outline_sec.objective,
                    bundle.selected_sources,
                    intermediate_context=ir_context,
                    evidence_bundle=bundle,
                    section_context=_build_section_prompt_context(
                        plan=plan,
                        outline=plan.outline,
                        section_index=section_index,
                    ),
                )
                section.section_id = outline_sec.section_id
                # Sidecar verification: deterministic prompt-compliance
                # metrics computed on final content.  Zero LLM cost.
                sidecar = _compute_section_sidecar_metrics(
                    section.content,
                    bundle.section_archetype,
                )
                return section, {
                    "section_id": outline_sec.section_id,
                    "title": outline_sec.title,
                    "relevant_source_count": relevant_total,
                    "selected_source_count": len(bundle.selected_sources),
                    **evidence_metrics,
                    "intermediate_context_chars": len(ir_context),
                    **sidecar,
                }

        t_sections = time.monotonic()
        section_results = list(await asyncio.gather(*[_gen(o) for o in plan.outline]))
        sections = [section for section, _ in section_results]
        section_input_metrics = [metrics for _, metrics in section_results]
        sections_ms = (time.monotonic() - t_sections) * 1000.0

        summary = ""
        summary_ms = 0.0
        if not defer_summary:
            t_summary = time.monotonic()
            summary = await self._generate_summary(plan.query, sections)
            summary_ms = (time.monotonic() - t_summary) * 1000.0

        duration = time.monotonic() - start

        report = ResearchReport(
            report_id=report_id,
            title=plan.query,
            summary=summary,
            sections=sections,
            references=sources.sources,
            metadata=ReportMetadata(
                style=style,
                source_count=len(sources.sources),
                section_count=len(sections),
                generated_by_mode=plan.settings.orchestration_mode,
                duration_seconds=round(duration, 2),
                section_input_metrics=section_input_metrics,
            ),
        )
        timings = {
            "report_sections_ms": round(sections_ms, 1),
            "report_summary_ms": round(summary_ms, 1),
            _SECTION_INPUT_METRICS_KEY: section_input_metrics,
        }
        return report, timings

    async def complete_summary(self, report: ResearchReport) -> float:
        """Generate and attach the summary for a report with deferred summary.

        Returns the wall-clock milliseconds spent generating the summary.
        """
        t = time.monotonic()
        report.summary = await self._generate_summary(report.title, report.sections)
        return round((time.monotonic() - t) * 1000.0, 1)

    async def generate_stream(
        self,
        plan: ResearchPlan,
        sources: AggregatedSources,
        style: ReportStyle = ReportStyle.DETAILED,
    ) -> AsyncIterator[ReportChunk]:
        """Stream report generation section-by-section."""
        report_id = f"rpt_{uuid.uuid4().hex[:8]}"
        seq = 0

        for outline_sec in plan.outline:
            seq += 1
            yield ReportChunk(
                report_id=report_id,
                sequence=seq,
                chunk_type=ReportChunkType.SECTION_START,
                section_id=outline_sec.section_id,
                section_title=outline_sec.title,
            )

            relevant, _, _ = self.select_section_sources(
                outline_sec.related_question_ids,
                sources,
                section_title=outline_sec.title,
                objective=outline_sec.objective,
                coverage_contract=outline_sec.coverage_contract,
            )
            section_index = plan.outline.index(outline_sec)
            section = await self._generate_section(
                plan.query,
                outline_sec.title,
                outline_sec.objective,
                relevant,
                section_context=_build_section_prompt_context(
                    plan=plan,
                    outline=plan.outline,
                    section_index=section_index,
                ),
            )

            seq += 1
            yield ReportChunk(
                report_id=report_id,
                sequence=seq,
                chunk_type=ReportChunkType.SECTION_COMPLETE,
                section_id=outline_sec.section_id,
                section_title=outline_sec.title,
                content_delta=section.content,
                citations_added=section.citations,
            )

        seq += 1
        yield ReportChunk(
            report_id=report_id,
            sequence=seq,
            chunk_type=ReportChunkType.COMPLETE,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def select_section_sources(
        self,
        question_ids: list[str],
        sources: AggregatedSources,
        *,
        section_title: str,
        objective: str,
        coverage_contract: AnswerCoverageContract | None = None,
    ) -> tuple[list[SourceReference], int, dict[str, int]]:
        """Select section sources via ranked recall, candidate-pool build, and final assembly."""

        bundle, relevant_total, metrics = self.build_section_evidence_bundle(
            question_ids,
            sources,
            section_title=section_title,
            objective=objective,
            coverage_contract=coverage_contract,
        )
        return bundle.selected_sources, relevant_total, metrics

    def build_section_evidence_bundle(
        self,
        question_ids: list[str],
        sources: AggregatedSources,
        *,
        section_title: str,
        objective: str,
        coverage_contract: AnswerCoverageContract | None = None,
        section_archetype: str = "",
    ) -> tuple[SectionEvidenceBundle, int, dict[str, int]]:
        """Build the evidence packet consumed by section writing.

        This keeps retrieval/ranking compatible with the existing source selection
        logic while making the writing contract explicit.

        When *section_archetype* is provided (from the planner), it takes
        precedence over the keyword-based _classify_section_archetype
        fallback, giving the planner full control over evidence policy.
        """

        coverage_contract = coverage_contract or AnswerCoverageContract()
        # Use planner-assigned archetype; fall back to keyword classification.
        resolved_archetype = section_archetype or _classify_section_archetype(coverage_contract)

        ranked = _relevant_sources(
            question_ids,
            sources,
            section_title=section_title,
            objective=objective,
            coverage_contract=coverage_contract,
        )
        source_question_counts = _build_source_question_counts(sources)
        candidate_pool = _build_section_candidate_pool(
            ranked,
            max_sources=self._max_section_sources,
            policy=self._section_evidence_policy,
        )
        selected = _assemble_section_sources(
            candidate_pool,
            ranked,
            max_sources=self._max_section_sources,
            policy=self._section_evidence_policy,
            source_question_counts=source_question_counts,
            coverage_contract=coverage_contract,
            section_archetype=resolved_archetype,
        )
        bundle = _build_section_evidence_bundle(
            selected,
            coverage_contract,
            ranked,
            section_archetype=resolved_archetype,
        )
        return (
            bundle,
            len(ranked),
            {
                "selected_domain_count": len(
                    {_source_domain(src) for src in bundle.selected_sources if _source_domain(src)}
                ),
                "authority_source_count": sum(
                    1 for src in bundle.selected_sources if _looks_authoritative_source(src)
                ),
                "cross_question_source_count": sum(
                    1
                    for src in bundle.selected_sources
                    if source_question_counts.get(src.reference_id, 0) > 1
                ),
                "content_usable_source_count": sum(
                    1 for src in bundle.selected_sources if _is_content_usable(src)
                ),
                "primary_evidence_count": len(bundle.primary_evidence),
                "counter_evidence_count": len(bundle.counter_evidence),
                "reserve_evidence_count": len(bundle.reserve_evidence),
                "unresolved_gap_count": len(bundle.unresolved_gaps),
                "caveat_count": len(bundle.caveat_obligations),
            },
        )

    async def _generate_section(
        self,
        query: str,
        title: str,
        objective: str,
        sources: list[SourceReference],
        intermediate_context: str = "",
        evidence_bundle: SectionEvidenceBundle | None = None,
        section_context: dict[str, str] | None = None,
    ) -> ReportSection:
        snip = self._snippet_max_chars
        bundle = evidence_bundle or _build_section_evidence_bundle(
            sources,
            AnswerCoverageContract(),
            sources,
        )
        context = section_context or _default_section_prompt_context()
        sources_text = "\n".join(
            f"  {s.reference_id} | {s.title} | {s.snippet[:snip]}"
            for s in sources[: self._max_source_display]
        )
        soft_checklist = _build_soft_checklist(bundle)
        prompt = _SECTION_PROMPT.format(
            query=query,
            title=title,
            objective=objective,
            section_archetype=bundle.section_archetype,
            section_position=context["section_position"],
            previous_section=context["previous_section"],
            next_section=context["next_section"],
            related_questions=context["related_questions"],
            sources_text=sources_text or "(no sources)",
            primary_evidence=_format_bundle_sources(bundle.primary_evidence, snip),
            counter_evidence=_format_bundle_sources(bundle.counter_evidence, snip),
            soft_checklist=soft_checklist,
        )
        if intermediate_context:
            prompt += (
                "\n\nPre-analysed findings from research agents "
                "(use to improve accuracy and citation fidelity):\n" + intermediate_context
            )
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=self._section_max_tokens,
            **self._llm_kwargs,
        )
        section = _parse_section(title, resp.content)
        # Post-generation fix-up chain lives in report_postprocess:
        # a single flat pipeline covers noise rewrite, paragraph layout,
        # duplicate subheading / paragraph dedup, writer-leak scrubbing,
        # and the query-language gate.  The order there is load-bearing;
        # see the module docstring for the rationale.
        ctx = SectionPostProcessContext(
            query=query,
            title=title,
            objective=objective,
            available_refs=[s.reference_id for s in sources],
            llm=self._llm,
            llm_kwargs=self._llm_kwargs,
            section_max_tokens=self._section_max_tokens,
        )
        return await postprocess_section(section, ctx)

    async def _clean_section_noise(
        self,
        content: str,
        *,
        title: str,
        objective: str,
        available_refs: list[str],
    ) -> str:
        """Test-compat shim delegating to the post-processing pipeline.

        The real orchestration lives in
        houyi.application.research.report_postprocess.clean_noise_step.
        A ReportSection is constructed purely to satisfy the step
        signature; callers keep passing raw content strings.
        """

        section = ReportSection(title=title, content=content, citations=[])
        ctx = SectionPostProcessContext(
            query="",
            title=title,
            objective=objective,
            available_refs=list(available_refs),
            llm=self._llm,
            llm_kwargs=self._llm_kwargs,
            section_max_tokens=self._section_max_tokens,
        )
        result = await clean_noise_step(section, ctx)
        return result.content

    async def _rewrite_noisy_paragraph(
        self,
        paragraph: str,
        *,
        title: str,
        objective: str,
        available_refs: str,
    ) -> str:
        """Test-compat shim delegating to the post-processing pipeline."""

        return await _rewrite_noisy_paragraph_impl(
            paragraph,
            title=title,
            objective=objective,
            available_refs=available_refs,
            llm=self._llm,
            llm_kwargs=self._llm_kwargs,
        )

    async def _generate_summary(
        self,
        query: str,
        sections: list[ReportSection],
    ) -> str:
        non_empty_sections = [s for s in sections if s.content.strip()]
        if not non_empty_sections:
            raise ValueError("Cannot generate report summary without section content")
        sections_text = "\n\n".join(f"## {s.title}\n{s.content[:500]}" for s in non_empty_sections)
        prompt = _SUMMARY_PROMPT.format(sections_text=sections_text)
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
            **self._llm_kwargs,
        )
        return resp.content.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_noisy_paragraphs(paragraphs: list[str]) -> list[int]:
    """Return indices of paragraphs that contain retrieval noise or lack citations.

    Detection is pure regex/heuristic — zero LLM cost.
    """
    noisy: list[int] = []
    for idx, para in enumerate(paragraphs):
        stripped = para.strip()
        if not stripped:
            continue
        # Skip headings and very short transition lines.
        if stripped.startswith("#") or len(stripped) < _MIN_PARAGRAPH_CHARS_FOR_CITATION:
            continue
        # Check for retrieval-process narration patterns.
        if any(pattern.search(stripped) for pattern in _NOISE_PATTERNS):
            noisy.append(idx)
            continue
        # Flag substantial paragraphs that contain no inline citation [ref_xxx].
        if not re.search(r"\[ref_\w+\]", stripped):
            noisy.append(idx)
    return noisy


# ---------------------------------------------------------------------------
# Section structural contract postprocess guards.
#
# These helpers run after the LLM section generation and the noise rewrite
# pass. They are deliberately zero-LLM-cost, topic-agnostic, and idempotent
# so they can also run in isolation against any Markdown body.
#
# * _consolidate_short_paragraphs merges adjacent short paragraphs so the
#   reader sees cohesive multi-sentence blocks instead of fragmented lines.
#   This is the only helper that mutates section.content — it performs a
#   human-visible layout repair.
# * _analyse_critical_analysis checks whether any critical-analysis
#   keyword is present in the body and returns a boolean flag.
# * _analyse_visualization_gaps checks whether the body lacks an expected
#   table (numeric-dense prose) or mermaid fence (hierarchy/sequence prose)
#   and returns the gap flags.
#
# The _analyse_* helpers **never mutate the content**. Their signals are
# collected by _compute_section_sidecar_metrics and surface through
# section_input_metrics for offline bench analysis. Writing the signals
# into section.content as HTML comments would leak them into the scored
# article text when upstream RACE cleaning falls back to the raw body.
# ---------------------------------------------------------------------------


_MERGEABLE_MIN_SENTENCES = 3
_MERGEABLE_TARGET_SENTENCES = 5
# Sentence-count based split thresholds (language-agnostic by design):
# paragraphs with more than _SPLITTABLE_MAX_SENTENCES registered sentences
# are broken at sentence boundaries into chunks of at most
# _SPLITTABLE_CHUNK_SENTENCES. Counts come from _SENTENCE_TERMINATORS so
# ASCII (.!?) and CJK (\u3002\uff01\uff1f) participate symmetrically.
_SPLITTABLE_MAX_SENTENCES = 8
_SPLITTABLE_CHUNK_SENTENCES = 4
_VISUALIZATION_NUMERIC_THRESHOLD = 3

# Numeric density heuristic: counts runs of digits with optional thousands
# separators or decimals, matching both Arabic numerals and percent signs.
_NUMERIC_TOKEN_PATTERN = re.compile(r"\d[\d,\.]*%?")

_MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_MERMAID_FENCE = re.compile(r"```\s*mermaid", re.IGNORECASE)


def _is_structural_paragraph(text: str) -> bool:
    """Return True when a paragraph must not be merged with its neighbours."""
    stripped = text.strip()
    if not stripped:
        return True
    first_line = stripped.splitlines()[0].lstrip()
    # Headings (ATX markdown).
    if first_line.startswith("#"):
        return True
    # List items (ordered or unordered).
    if re.match(r"^[-*+]\s", first_line) or re.match(r"^\d+[\.)]\s", first_line):
        return True
    # Table fragments.
    if first_line.startswith("|"):
        return True
    # Code / mermaid fences anywhere in the paragraph.
    if "```" in stripped:
        return True
    # Standalone HTML comment markers (already-attached hints or block quotes).
    return first_line.startswith("<!--") or first_line.startswith(">")


def _count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    # Count sentence terminators plus one for any trailing non-terminated
    # fragment, so a single-sentence paragraph without a terminator still
    # registers as one sentence.
    terminators = len(_SENTENCE_TERMINATORS.findall(stripped))
    if terminators == 0:
        return 1
    return terminators


def _split_long_paragraph(text: str) -> list[str]:
    """Break an over-long prose paragraph into chunks at sentence boundaries.

    Operates on the same _SENTENCE_TERMINATORS regex used for the merge
    threshold, so the behaviour is symmetric across ASCII and CJK prose.
    Never splits structural blocks; callers must pre-filter those via
    _is_structural_paragraph.

    Behaviour contract:

    * Paragraphs with <= _SPLITTABLE_MAX_SENTENCES sentences are
      returned unchanged (single-element list). This keeps moderate
      8-sentence paragraphs untouched.
    * Above the threshold, text is sliced at every
      _SPLITTABLE_CHUNK_SENTENCES-th terminator.
    * A trailing fragment shorter than 2 sentences is folded back into the
      previous chunk to avoid orphan single-sentence paragraphs.
    * No word or character is dropped; only whitespace collapses on the
      seam, preserving information while exposing paragraph structure.
    """
    terminator_ends = [m.end() for m in _SENTENCE_TERMINATORS.finditer(text)]
    sentence_count = len(terminator_ends) or 1
    if sentence_count <= _SPLITTABLE_MAX_SENTENCES:
        return [text]
    chunks: list[str] = []
    prev = 0
    for idx, end in enumerate(terminator_ends, start=1):
        if idx % _SPLITTABLE_CHUNK_SENTENCES == 0 and idx < sentence_count:
            piece = text[prev:end].strip()
            if piece:
                chunks.append(piece)
            prev = end
    tail = text[prev:].strip()
    if tail:
        tail_sentences = len(_SENTENCE_TERMINATORS.findall(tail)) or 1
        if tail_sentences < 2 and chunks:
            # Re-attach orphan tail to prior chunk so every resulting
            # paragraph carries >= 2 sentences.
            chunks[-1] = (chunks[-1] + " " + tail).strip()
        else:
            chunks.append(tail)
    return chunks or [text]


def _expand_long_paragraphs(paragraphs: Iterable[str]) -> list[str]:
    """Return paragraphs with non-structural giants split into chunks.

    Structural blocks (headings, lists, tables, code fences, HTML
    comments / blockquotes) pass through untouched. Prose paragraphs
    above _SPLITTABLE_MAX_SENTENCES are delegated to
    _split_long_paragraph; shorter ones are returned as-is. Factored
    out of _consolidate_short_paragraphs to keep that function's
    control flow under the C901 complexity gate.
    """
    expanded: list[str] = []
    for para in paragraphs:
        if _is_structural_paragraph(para):
            expanded.append(para)
            continue
        expanded.extend(_split_long_paragraph(para))
    return expanded


def _consolidate_short_paragraphs(content: str) -> str:
    """Normalise paragraph structure: split giants, merge shorts.

    A single post-generation pass handles both sides of the paragraph
    layout contract:

    * _expand_long_paragraphs calls _split_long_paragraph on
      non-structural paragraphs whose sentence count exceeds
      _SPLITTABLE_MAX_SENTENCES, exposing reading structure that EN
      LLM output collapses into 15-20 sentence monoliths.
    * The merge loop then rejoins adjacent short paragraphs up to
      _MERGEABLE_TARGET_SENTENCES sentences.

    Both stages ignore structural blocks (headings, lists, tables, code
    fences, HTML comments / blockquotes). All thresholds are expressed in
    sentences so the behaviour is language-agnostic by design.
    """
    if not content:
        return content
    # Step 0: unwrap per-paragraph {"content": "..."} envelopes.
    # Section writers occasionally emit multiple JSON envelopes in a
    # single response, which _parse_section only strips on the first
    # one; subsequent envelopes survive as raw text paragraphs. Stripping
    # at the paragraph level here catches every such leak regardless of
    # upstream source (main writer, noise rewrite, repair rewrite). The
    # helper is a no-op for plain prose so this adds zero overhead on
    # the common path.
    raw_paragraphs = [_strip_content_envelope(p) for p in content.split("\n\n")]
    paragraphs = _expand_long_paragraphs(raw_paragraphs)
    if len(paragraphs) < 2:
        return "\n\n".join(paragraphs)
    merged: list[str] = []
    buffer: list[str] = []
    buffer_sentences = 0

    def _flush() -> None:
        nonlocal buffer, buffer_sentences
        if buffer:
            merged.append(" ".join(part.strip() for part in buffer))
            buffer = []
            buffer_sentences = 0

    for para in paragraphs:
        if _is_structural_paragraph(para):
            _flush()
            merged.append(para)
            continue
        sentences = _count_sentences(para)
        if sentences >= _MERGEABLE_TARGET_SENTENCES:
            _flush()
            merged.append(para)
            continue
        if sentences >= _MERGEABLE_MIN_SENTENCES and not buffer:
            # Standalone paragraph already meets the minimum; leave as-is so
            # well-formed sections stay untouched.
            merged.append(para)
            continue
        buffer.append(para)
        buffer_sentences += sentences
        if buffer_sentences >= _MERGEABLE_TARGET_SENTENCES:
            _flush()
    _flush()
    return "\n\n".join(merged)


def _deduplicate_subheadings(content: str) -> str:
    """Collapse duplicate ### Subheading blocks in a single section body.

    The section writer occasionally emits the full ### Subheading tree
    twice inside one LLM response.  Empirical inspection of the
    bench2-en5-20260422b-head articles across five EN queries shows
    pass-2 bodies are either byte-for-byte identical to pass-1 (similarity
    1.000), strictly truncated prefixes of pass-1 (similarity 0.2-0.7 but
    prefix-contained), or overlapping second-drafts that share opening
    sentences.  In every observed pair, pass-2 contributed no unique facts
    beyond pass-1 (or vice-versa).

    Dedup rule: whenever two ### Subheading chunks in the same
    section body share the same heading text, keep the chunk with the
    longest body and drop the others.  This is the minimum information
    loss policy: empty / prefix-truncated / near-copy pass-2 bodies all
    lose to the richer pass-1; pass-2 that genuinely extends the
    subsection wins over an empty / short pass-1 (the Q51 "empty-pass-1
    full-pass-2" pattern).  When both chunks have identical body length,
    the earlier one wins deterministically.

    Operates on section-level content only — callers pass a single
    section body, so there is no cross-section leakage.
    Language-agnostic: works on both EN and ZH subheading titles.
    Returns the original string unchanged whenever no duplicates are
    detected, so this pass is a guaranteed no-op on clean bodies (ZH
    case1 baseline is bit-for-bit unchanged).
    """

    if "### " not in content:
        return content
    # Split on ###  at the start of a line.  Keep the prefix preceding
    # the first subheading intact; each later chunk is ### title\n<body>.
    parts = re.split(r"(?m)(?=^### )", content)
    if len(parts) < 2:
        return content
    head = parts[0]
    # Inventory every chunk keyed by normalised title.  For duplicates,
    # track the index of the chunk with the longest body so the final
    # pass can keep exactly that one.
    chunks = parts[1:]
    bodies: list[str] = []
    titles: list[str] = []
    for chunk in chunks:
        newline = chunk.find("\n")
        title = chunk if newline < 0 else chunk[:newline]
        body = "" if newline < 0 else chunk[newline + 1 :].strip()
        titles.append(title.strip())
        bodies.append(body)
    # Map each title to the index of its longest-body chunk.  Earlier
    # duplicates win on ties so the output is deterministic across runs.
    best_idx_by_title: dict[str, int] = {}
    for idx, title_key in enumerate(titles):
        if title_key not in best_idx_by_title:
            best_idx_by_title[title_key] = idx
            continue
        if len(bodies[idx]) > len(bodies[best_idx_by_title[title_key]]):
            best_idx_by_title[title_key] = idx
    # Emit the surviving chunks in the order each title FIRST appeared.
    # This preserves the writer's narrative ordering even when the
    # longest-body winner lives in pass 2.  Without this anchor the
    # observed Q51 "Food / Clothing / Comparative / Critical" sequence
    # got reordered to "Comparative / Critical / Food / Clothing"
    # because pass-2 Food/Clothing copies were one character longer
    # and their late positions leaked into the final order.
    first_seen_order: list[str] = []
    first_seen_set: set[str] = set()
    for title_key in titles:
        if title_key in first_seen_set:
            continue
        first_seen_set.add(title_key)
        first_seen_order.append(title_key)
    if len(first_seen_order) == len(chunks):
        return content
    kept = [chunks[best_idx_by_title[t]] for t in first_seen_order]
    return head + "".join(kept)


# Minimum paragraph length (characters) considered for duplicate removal.
# Short paragraphs (bullet fragments, one-line captions, table rows) can
# legitimately recur in prose and are skipped so the dedup pass does not
# collapse intentionally repeated topic sentences or short list items.
_PARAGRAPH_DUP_MIN_CHARS = 150


# ---------------------------------------------------------------------------
# Empty / dangling ### Subheading prune.
#
# The writer occasionally emits a ### Subheading whose body is either
# completely empty or a single "here comes a list" intro sentence ending
# in a colon, then yields the section without producing the promised
# content.  Observed live on user session rr_d79ddb66a58c section 4
# (two empty ### blocks) and section 5 (two empty plus one 29-char
# dangling-colon block whose body summarises the analysis and ends in
# a fullwidth CJK colon U+FF1A promising an unwritten list).
#
# Such orphan headings damage six of the sixteen RACE criteria at once
# (information depth, data support, analysis depth, logical reasoning,
# clear structure, complete coverage), so the prune is both cheap and
# high-leverage.  Logic is deterministic, language-agnostic, and a
# no-op on clean bodies.
# ---------------------------------------------------------------------------
# Prose below this threshold counts as "near-empty" when paired with a
# dangling-colon ending.  Chosen so the real short but substantive bodies
# observed in the ZH 52.55 baseline (smallest healthy body was 171 chars)
# never cross into the drop path.
_PRUNE_MIN_SUBSTANTIVE_CHARS = 40
# Absolute empty-body threshold.  Five chars lets "  \n  " style
# whitespace counts / stray punctuation count as empty without
# accidentally dropping a three-word sentence.
_PRUNE_EMPTY_BODY_THRESHOLD = 5
# Trailing colon (ASCII or CJK fullwidth) followed by optional trailing
# whitespace marks the dangling-intro shape: the body promises a list
# but the writer never emitted one.
_DANGLING_COLON_RE = re.compile(r"[\uff1a:]\s*\Z")
# [ref_<hex>] tokens are stripped before measuring prose length so
# a body that is nothing but citations counts as empty.
_PRUNE_CITATION_TOKEN_RE = re.compile(r"\[ref_[a-fA-F0-9]+\]")
# Body counts as structural (= substantive regardless of prose length)
# when any line looks like a table row, code fence, bullet / numbered
# list item, or blockquote.  Keeps bodies that are just a compact table
# from being mistaken for an empty heading.
_PRUNE_STRUCTURAL_HINT_RE = re.compile(r"(?m)^\s*(?:\||```|[-*+]\s|\d+\.\s|>\s)")


def _prune_empty_subheadings(content: str) -> str:
    """Drop ### Subheading blocks whose body is empty or a dangling intro.

    Operates on a single section body.  A block is pruned when:

    1. The body is empty or near-empty (< _PRUNE_EMPTY_BODY_THRESHOLD
       characters after stripping whitespace and [ref_xxx] tokens),
       or
    2. The body is short (< _PRUNE_MIN_SUBSTANTIVE_CHARS after the
       same strip) AND ends with a colon — the writer promised a list
       under this heading but never produced one.

    Structural bodies (tables, fenced code blocks, lists, blockquotes)
    are always kept because the structural content itself is substantive
    even when the surrounding prose is short.  Returns the input
    unchanged when nothing is pruned so this pass is a guaranteed no-op
    on clean ZH / EN bodies.
    """

    if "### " not in content:
        return content
    parts = re.split(r"(?m)(?=^### )", content)
    if len(parts) < 2:
        return content
    head = parts[0]
    kept: list[str] = []
    dropped_any = False
    for chunk in parts[1:]:
        newline = chunk.find("\n")
        body = "" if newline < 0 else chunk[newline + 1 :]
        if _PRUNE_STRUCTURAL_HINT_RE.search(body):
            kept.append(chunk)
            continue
        prose = _PRUNE_CITATION_TOKEN_RE.sub("", body).strip()
        if len(prose) < _PRUNE_EMPTY_BODY_THRESHOLD:
            dropped_any = True
            continue
        if len(prose) < _PRUNE_MIN_SUBSTANTIVE_CHARS and _DANGLING_COLON_RE.search(prose):
            dropped_any = True
            continue
        kept.append(chunk)
    if not dropped_any:
        return content
    return head + "".join(kept)


def _deduplicate_paragraphs(content: str) -> str:
    """Remove paragraphs that appear verbatim more than once in one section.

    After _deduplicate_subheadings drops pass-2 ### Subheading
    chunks, the transition paragraph that originally sat between pass 1
    and pass 2 remains attached to the tail of the last pass-1 chunk.
    In the bench2 20260422b-head articles this transition paragraph is
    identical to the section's opening paragraph (observed across Q51 /
    Q52 / Q53 / Q55), so the opener shows up twice in the final body.

    This pass is the paragraph-granularity companion to the
    subheading-level dedup: identify paragraphs that are at least
    _PARAGRAPH_DUP_MIN_CHARS characters long and appear verbatim
    more than once inside a single section body; keep only the first
    occurrence.  The minimum length filter avoids collapsing legitimate
    short repetitions (e.g. list item anchors, table cell fragments).

    Structural blocks (headings, fenced code, tables) are never dropped
    because _is_structural_paragraph reports them as structural and
    they skip the duplicate check.
    """

    if "\n\n" not in content:
        return content
    paragraphs = content.split("\n\n")
    seen: set[str] = set()
    kept: list[str] = []
    dropped_any = False
    for para in paragraphs:
        stripped = para.strip()
        # Structural blocks always pass through untouched so the output
        # keeps every heading, list, table, and fenced code block in
        # place.  Short prose fragments are exempt from the dup check
        # because short repeats are frequently legitimate.
        if _is_structural_paragraph(para) or len(stripped) < _PARAGRAPH_DUP_MIN_CHARS:
            kept.append(para)
            continue
        if stripped in seen:
            dropped_any = True
            continue
        seen.add(stripped)
        kept.append(para)
    if not dropped_any:
        return content
    return "\n\n".join(kept)


def _analyse_visualization_gaps(content: str) -> dict[str, bool]:
    """Return structural signals describing missing visualisations.

    The result carries two boolean flags, both observational only:

    * needs_table — numeric token density meets
      _VISUALIZATION_NUMERIC_THRESHOLD but the body contains no markdown
      table row, hinting the writer should prefer a compact table;
    * needs_mermaid — any hierarchy/sequence keyword from
      SECTION_VISUAL_TRIGGER_KEYWORDS is present but the body contains no
      mermaid fence, hinting a flowchart could clarify the relationships.

    The function never mutates content. Callers emit these flags as
    sidecar metrics for offline bench analysis.
    """
    if not content:
        return {"needs_table": False, "needs_mermaid": False}
    numeric_hits = len(_NUMERIC_TOKEN_PATTERN.findall(content))
    has_table = bool(_MARKDOWN_TABLE_ROW.search(content))
    needs_table = numeric_hits >= _VISUALIZATION_NUMERIC_THRESHOLD and not has_table
    lowered = content.lower()
    trigger_hit = any(keyword.lower() in lowered for keyword in SECTION_VISUAL_TRIGGER_KEYWORDS)
    has_mermaid = bool(_MERMAID_FENCE.search(content))
    needs_mermaid = trigger_hit and not has_mermaid
    return {"needs_table": needs_table, "needs_mermaid": needs_mermaid}


def _analyse_critical_analysis(content: str) -> bool:
    """Return True when at least one critical-analysis keyword is present.

    Keywords come from SECTION_CRITICAL_ANALYSIS_KEYWORDS (bilingual,
    topic-agnostic). The check is case-insensitive. Absence of any keyword
    is itself a useful signal — surfaced as a sidecar metric, not injected
    back into the narrative.
    """
    if not content:
        return False
    lowered = content.lower()
    for keyword in SECTION_CRITICAL_ANALYSIS_KEYWORDS:
        token = keyword.lower()
        if token and token in lowered:
            return True
    return False


def _compute_section_sidecar_metrics(
    content: str,
    section_archetype: str,
) -> dict[str, Any]:
    """Deterministic prompt-compliance metrics for sidecar verification.

    Zero LLM cost.  Computed on final section content after noise cleanup.
    Attached to section_input_metrics for offline analysis during
    benchmark runs — never enters the main scoring or generation path.

    Metrics produced:

    - sidecar_bullet_line_ratio: fraction of non-empty lines that are
      bullet/list items.  High values (>0.5) indicate the LLM ignored the
      "prefer dense analytical paragraphs" instruction.
    - sidecar_avg_paragraph_sentences: average sentence count per
      substantive paragraph.  Below 3 indicates shallow paragraphs.
    - sidecar_bold_heading_count: Markdown ### headings found.
    - sidecar_citation_count / sidecar_unique_citation_count:
      total and unique inline [ref_xxx] citations.
    - sidecar_uncited_paragraph_count: substantive paragraphs without
      any inline citation.
    - sidecar_word_count: total whitespace-delimited tokens.
    - sidecar_archetype: the resolved archetype label.
    - sidecar_archetype_compliant: whether archetype-specific keywords
      were detected (English-only; CJK content yields false, which is
      itself a useful diagnostic signal).
    - sidecar_critical_analysis_present: whether any bilingual
      critical-analysis keyword (SECTION_CRITICAL_ANALYSIS_KEYWORDS) is
      present in the body.
    - sidecar_visualization_needs_table /
      sidecar_visualization_needs_mermaid: whether the body is dense in
      numeric tokens without a table, or mentions hierarchy/sequence cues
      without a mermaid fence. Both are observability-only; the signals do
      not mutate the section body so nothing leaks into the scored article.
    """
    lines = [ln for ln in content.split("\n") if ln.strip()]
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # Bullet / list line ratio.
    bullet_lines = sum(
        1 for ln in lines if re.match(r"\s*[-*]\s", ln) or re.match(r"\s*\d+[.)]\s", ln)
    )
    bullet_line_ratio = round(bullet_lines / max(len(lines), 1), 2)

    # Average paragraph sentence count (skip headings / short transitions).
    sentence_counts: list[int] = []
    for para in paragraphs:
        if para.startswith("#") or len(para) < _MIN_PARAGRAPH_CHARS_FOR_CITATION:
            continue
        parts = _SENTENCE_TERMINATORS.split(para)
        sentence_counts.append(len([s for s in parts if s.strip()]))
    avg_paragraph_sentences = round(sum(sentence_counts) / max(len(sentence_counts), 1), 1)

    # Bold sub-headings.
    bold_heading_count = len(re.findall(r"^#{2,4}\s", content, re.MULTILINE))

    # Citation counts.
    citations = re.findall(r"\[ref_\w+\]", content)
    citation_count = len(citations)
    unique_citation_count = len(set(citations))

    # Uncited substantive paragraphs.
    uncited_paragraphs = 0
    for para in paragraphs:
        if para.startswith("#") or len(para) < _MIN_PARAGRAPH_CHARS_FOR_CITATION:
            continue
        if not re.search(r"\[ref_\w+\]", para):
            uncited_paragraphs += 1

    # Word count.
    word_count = len(content.split())

    # Archetype keyword compliance (English-only; CN extension deferred).
    archetype_keywords = ARCHETYPE_COMPLIANCE_KEYWORDS.get(section_archetype, ())
    content_lower = content.lower()
    matched_keywords = [kw for kw in archetype_keywords if kw in content_lower]

    # Structural contract signals (bilingual, topic-agnostic). Emitted as
    # observability only; never written back into section content.
    critical_present = _analyse_critical_analysis(content)
    visual_gaps = _analyse_visualization_gaps(content)

    return {
        "sidecar_bullet_line_ratio": bullet_line_ratio,
        "sidecar_avg_paragraph_sentences": avg_paragraph_sentences,
        "sidecar_bold_heading_count": bold_heading_count,
        "sidecar_citation_count": citation_count,
        "sidecar_unique_citation_count": unique_citation_count,
        "sidecar_uncited_paragraph_count": uncited_paragraphs,
        "sidecar_word_count": word_count,
        "sidecar_archetype": section_archetype,
        # Serialized as int/str to comply with section_input_metrics
        # type constraint: dict[str, int | str].
        "sidecar_archetype_compliant": 1 if matched_keywords else 0,
        "sidecar_archetype_keywords_matched": ",".join(matched_keywords[:5]),
        "sidecar_critical_analysis_present": 1 if critical_present else 0,
        "sidecar_visualization_needs_table": 1 if visual_gaps["needs_table"] else 0,
        "sidecar_visualization_needs_mermaid": 1 if visual_gaps["needs_mermaid"] else 0,
    }


def _intermediate_context(
    question_ids: list[str],
    ir_by_qid: dict[str, IntermediateReport],
    *,
    per_question_cap: int = _DEFAULT_INTERMEDIATE_PER_QUESTION_MAX_CHARS,
    total_cap: int = _DEFAULT_INTERMEDIATE_CONTEXT_MAX_CHARS,
) -> str:
    """Build a text block from intermediate reports for the given questions."""
    parts: list[str] = []
    total_chars = 0
    for qid in question_ids:
        ir = ir_by_qid.get(qid)
        if ir and ir.analysis:
            chunk = (
                f"[Sub-question: {ir.question}]\n"
                f"Confidence: {ir.confidence:.0%}\n"
                f"{ir.analysis[:per_question_cap]}"
            )
            projected = total_chars + len(chunk)
            if projected > total_cap:
                break
            parts.append(chunk)
            total_chars = projected
    return "\n\n---\n\n".join(parts)[:total_cap]


def _relevant_sources(
    question_ids: list[str],
    agg: AggregatedSources,
    *,
    section_title: str,
    objective: str,
    coverage_contract: AnswerCoverageContract,
) -> list[SourceReference]:
    """Pick and rank sources relevant to a report section.

    The ranking intentionally blends reliability, topical similarity, question
    coverage, authority, and freshness so that each section prompt sees a small
    but strong evidence set instead of an unbounded bag of sources.
    """
    ref_ids: set[str] = set()
    for qid in question_ids:
        ref_ids.update(agg.grouped_by_question.get(qid, []))
    lookup = {s.reference_id: s for s in agg.sources}
    question_candidates = [lookup[rid] for rid in ref_ids if rid in lookup]

    # When question-aligned candidates are below the floor, supplement
    # with the full source pool so the scoring function can surface
    # globally relevant sources.  Question-aligned sources retain a
    # natural advantage via the cross_question_coverage scoring bonus.
    if len(question_candidates) < _SECTION_SOURCE_FLOOR:
        seen = ref_ids.copy()
        global_extra = [s for s in agg.sources if s.reference_id not in seen]
        candidates = question_candidates + global_extra
    else:
        candidates = question_candidates

    source_question_counts: dict[str, int] = {}
    for grouped_ids in agg.grouped_by_question.values():
        for reference_id in grouped_ids:
            source_question_counts[reference_id] = source_question_counts.get(reference_id, 0) + 1

    ranked = sorted(
        candidates,
        key=lambda src: (
            _score_source_for_section(
                src,
                section_title=section_title,
                objective=objective,
                coverage_contract=coverage_contract,
                question_ids=question_ids,
                source_question_counts=source_question_counts,
            ),
            src.reliability_score,
            src.title,
        ),
        reverse=True,
    )
    return ranked


def _score_source_for_section(
    src: SourceReference,
    *,
    section_title: str,
    objective: str,
    coverage_contract: AnswerCoverageContract,
    question_ids: list[str],
    source_question_counts: dict[str, int],
) -> float:
    keywords = _extract_section_keywords(f"{section_title} {objective}")
    for facet in coverage_contract.must_cover_facets:
        keywords |= _extract_section_keywords(
            f"{facet.name} {facet.intent} {' '.join(facet.bilingual_terms)}"
        )
    searchable = f"{src.title} {src.snippet} {src.url or ''}"
    overlap = _keyword_overlap_score(searchable, keywords)
    authority = 1.5 if _looks_authoritative_source(src) else 0.0
    freshness = 0.6 if _looks_fresh_source(src) else 0.0
    cross_question_coverage = min(
        1.5,
        0.5 * min(source_question_counts.get(src.reference_id, 0), max(len(question_ids), 1)),
    )
    return (src.reliability_score * 4.0) + overlap + authority + freshness + cross_question_coverage


def _extract_section_keywords(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w\-]{3,}", text.lower()) if not token.isdigit()}


def _keyword_overlap_score(text: str, keywords: set[str]) -> float:
    if not keywords:
        return 0.0
    lowered = text.lower()
    overlap = sum(1 for keyword in keywords if keyword in lowered)
    return min(2.0, overlap * 0.35)


def _looks_authoritative_source(src: SourceReference) -> bool:
    url = (src.url or "").lower()
    title = src.title.lower()
    return any(
        marker in url for marker in (".gov", ".edu", "arxiv.org", "ssrn.com", "who.int")
    ) or any(marker in title for marker in ("report", "official", "white paper", "annual report"))


def _looks_fresh_source(src: SourceReference) -> bool:
    text = f"{src.title} {src.snippet} {src.url or ''}".lower()
    return any(marker in text for marker in ("2024", "2025", "2026", "current", "latest", "recent"))


def _is_content_usable(src: SourceReference) -> bool:
    snippet = (src.snippet or "").strip()
    title = (src.title or "").strip()
    return len(snippet) >= 24 or (len(title) >= 18 and len(snippet) >= 8)


def _source_domain(src: SourceReference) -> str:
    url = (src.url or "").lower().strip()
    if not url:
        return ""
    normalized = re.sub(r"^https?://", "", url)
    return normalized.split("/", 1)[0]


def _build_source_question_counts(agg: AggregatedSources) -> dict[str, int]:
    counts: dict[str, int] = {}
    for grouped_ids in agg.grouped_by_question.values():
        for reference_id in grouped_ids:
            counts[reference_id] = counts.get(reference_id, 0) + 1
    return counts


def _build_section_candidate_pool(
    ranked: list[SourceReference],
    *,
    max_sources: int,
    policy: SectionEvidencePolicy,
) -> list[SourceReference]:
    target_size = max(max_sources, policy.candidate_pool_size)
    pool = list(ranked[:target_size])
    if not policy.require_content_usable:
        return pool
    seen_ids = {src.reference_id for src in pool}
    for src in ranked[target_size:]:
        if src.reference_id in seen_ids or not _is_content_usable(src):
            continue
        pool.append(src)
        seen_ids.add(src.reference_id)
        if len(pool) >= target_size + max_sources:
            break
    return pool


def _assemble_section_sources(
    candidate_pool: list[SourceReference],
    ranked: list[SourceReference],
    *,
    max_sources: int,
    policy: SectionEvidencePolicy,
    source_question_counts: dict[str, int],
    coverage_contract: AnswerCoverageContract,
    section_archetype: str = "overview_and_synthesis",
) -> list[SourceReference]:
    selected: list[SourceReference] = []
    seen_ids: set[str] = set()
    seen_domains: set[str] = set()
    targets = _archetype_targets(policy, section_archetype)

    for src in candidate_pool:
        _try_add_section_source(
            src,
            selected=selected,
            seen_ids=seen_ids,
            seen_domains=seen_domains,
            max_sources=max_sources,
            policy=policy,
            prefer_new_domain=False,
        )
        if selected:
            break

    _fill_section_sources(
        candidate_pool,
        selected=selected,
        seen_ids=seen_ids,
        seen_domains=seen_domains,
        max_sources=max_sources,
        policy=policy,
        prefer_new_domain=True,
        stop_when_domain_count=policy.min_domain_diversity,
    )
    _fill_section_sources(
        candidate_pool,
        selected=selected,
        seen_ids=seen_ids,
        seen_domains=seen_domains,
        max_sources=max_sources,
        policy=policy,
        prefer_new_domain=False,
    )

    _ensure_cross_question_sources(
        ranked,
        selected=selected,
        seen_ids=seen_ids,
        seen_domains=seen_domains,
        max_sources=max_sources,
        policy=policy,
        source_question_counts=source_question_counts,
    )
    _ensure_authority_sources(
        ranked,
        selected=selected,
        seen_ids=seen_ids,
        seen_domains=seen_domains,
        max_sources=max_sources,
        policy=targets,
    )
    _ensure_facet_coverage_sources(
        ranked,
        selected=selected,
        seen_ids=seen_ids,
        seen_domains=seen_domains,
        max_sources=max_sources,
        policy=policy,
        coverage_contract=coverage_contract,
    )
    _ensure_counter_evidence_sources(
        ranked,
        selected=selected,
        seen_ids=seen_ids,
        seen_domains=seen_domains,
        max_sources=max_sources,
        policy=targets,
    )

    _fill_section_sources(
        candidate_pool,
        selected=selected,
        seen_ids=seen_ids,
        seen_domains=seen_domains,
        max_sources=max_sources,
        policy=policy,
        prefer_new_domain=False,
    )

    return selected[:max_sources]


def _build_section_evidence_bundle(
    selected_sources: list[SourceReference],
    coverage_contract: AnswerCoverageContract,
    ranked_sources: list[SourceReference],
    *,
    section_archetype: str = "",
) -> SectionEvidenceBundle:
    # Use caller-provided archetype; fall back to keyword classification.
    section_archetype = section_archetype or _classify_section_archetype(coverage_contract)
    primary: list[SourceReference] = []
    counter: list[SourceReference] = []
    for src in selected_sources:
        if _looks_authoritative_source(src) and len(primary) < 3:
            primary.append(src)
        elif _looks_counter_evidence(src) and len(counter) < 2:
            counter.append(src)
    if not primary:
        primary = selected_sources[: min(3, len(selected_sources))]
    unresolved_gaps = _missing_bundle_facets(selected_sources, coverage_contract)
    reserve = _build_reserve_evidence(
        selected_sources,
        ranked_sources,
        coverage_contract,
        unresolved_gaps,
    )
    return SectionEvidenceBundle(
        selected_sources=selected_sources,
        primary_evidence=primary,
        counter_evidence=counter,
        reserve_evidence=reserve,
        unresolved_gaps=unresolved_gaps,
        caveat_obligations=list(coverage_contract.required_caveats),
        coverage_facets=[facet.name for facet in coverage_contract.must_cover_facets],
        comparison_axes=list(coverage_contract.comparison_axes),
        evidence_expectations=list(coverage_contract.evidence_expectations),
        time_scope=coverage_contract.time_scope,
        geo_scope=coverage_contract.geo_scope,
        section_archetype=section_archetype,
    )


def _classify_section_archetype(coverage_contract: AnswerCoverageContract) -> str:
    contract_text = " ".join(
        [
            " ".join(coverage_contract.comparison_axes),
            " ".join(coverage_contract.required_caveats),
            " ".join(coverage_contract.evidence_expectations),
            " ".join(facet.name for facet in coverage_contract.must_cover_facets),
            " ".join(facet.intent for facet in coverage_contract.must_cover_facets),
        ]
    ).lower()
    if any(
        token in contract_text for token in ("compare", "comparison", "versus", "vs", "trade-off")
    ):
        return "comparison"
    if any(
        token in contract_text
        for token in ("risk", "limitation", "caveat", "uncertainty", "constraint")
    ):
        return "risk_and_caveat"
    if any(
        token in contract_text for token in ("trend", "timeline", "evolution", "history", "current")
    ):
        return "trend_and_state"
    return "overview_and_synthesis"


def _archetype_targets(
    policy: SectionEvidencePolicy,
    section_archetype: str,
) -> SectionEvidencePolicy:
    if section_archetype == "comparison":
        return SectionEvidencePolicy(
            candidate_pool_size=policy.candidate_pool_size,
            min_domain_diversity=policy.min_domain_diversity,
            require_content_usable=policy.require_content_usable,
            min_cross_question_sources=max(policy.min_cross_question_sources, 2),
            min_authority_sources=max(policy.min_authority_sources, 1),
            min_counter_evidence_sources=max(policy.min_counter_evidence_sources, 1),
        )
    if section_archetype == "risk_and_caveat":
        return SectionEvidencePolicy(
            candidate_pool_size=policy.candidate_pool_size,
            min_domain_diversity=policy.min_domain_diversity,
            require_content_usable=policy.require_content_usable,
            min_cross_question_sources=max(policy.min_cross_question_sources, 1),
            min_authority_sources=max(policy.min_authority_sources, 1),
            min_counter_evidence_sources=max(policy.min_counter_evidence_sources, 1),
        )
    return policy


def _looks_counter_evidence(src: SourceReference) -> bool:
    text = f"{src.title} {src.snippet}".lower()
    return any(marker in text for marker in COUNTER_EVIDENCE_MARKERS)


def _default_section_prompt_context() -> dict[str, str]:
    return {
        "section_position": "(unknown)",
        "previous_section": "(none)",
        "next_section": "(none)",
        "related_questions": "(none)",
    }


# Mermaid/diagram marker and repeating short-hash fragments that the
# sub-agent emits when its diagram rendering loop enters a retry
# storm.  A paragraph tripping both (a diagram marker *and* multiple
# junk signals) is functionally un-readable and always hurts the
# RACE read score.  See _looks_like_junk_paragraph for the
# combined heuristic.
_DIAGRAM_MARKER_RE = re.compile(
    r"\b(?:graph\s+(?:TD|LR|BT|RL)|flowchart|sequenceDiagram|stateDiagram)",
)
_HASH_FRAGMENT_RE = re.compile(r"\b[a-f]\d{2}-b\d{4}\b|\b[a-f]\d{2}b\d{4,}\b")


def _unwrap_envelope_leaks(text: str) -> str:
    """Unwrap multiline {"content": "..."} leaks, preserving prose.

    Locates each {\\n  "content": "... opener, pairs it with the
    nearest trailing "} (optionally followed by a citations array)
    or a fallback boundary (next Markdown heading / end of text), and
    emits the inner prose with JSON string escapes un-escaped.  Both
    closed and malformed envelopes are handled; only the JSON scaffold
    is removed, and no content is ever deleted.

    Idempotent on text that contains no envelopes.
    """

    result: list[str] = []
    pos = 0
    while True:
        opener = _ENVELOPE_OPENER_RE.search(text, pos)
        if opener is None:
            result.append(text[pos:])
            break
        result.append(text[pos : opener.start()])
        inner_start = opener.end()
        tail = _ENVELOPE_TAIL_RE.search(text, inner_start)
        if tail is not None:
            inner = text[inner_start : tail.start()]
            pos = tail.end()
        else:
            # No closing brace — envelope is truncated.  Recover the
            # body until the next Markdown heading or end of text so
            # the prose survives into the scored article.
            boundary = re.search(r"\n#{1,6}\s", text[inner_start:])
            if boundary is not None:
                inner = text[inner_start : inner_start + boundary.start()]
                pos = inner_start + boundary.start()
            else:
                inner = text[inner_start:]
                pos = len(text)
        result.append(_json_unescape(inner))
    return "".join(result)


def _json_unescape(text: str) -> str:
    """Un-escape a JSON string fragment without requiring a full parse."""

    try:
        return json.loads(f'"{text}"')
    except (json.JSONDecodeError, ValueError):
        # Fall back to common escape sequences when the fragment
        # contains unescaped control chars (which the writer often
        # emits and json.loads rejects).
        return (
            text.replace("\\\\", "\x00BS\x00")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\x00BS\x00", "\\")
        )


def _looks_like_junk_paragraph(paragraph: str) -> bool:
    """True when a paragraph is a diagram/sub-agent retry-storm residue.

    Requires BOTH a diagram/code marker AND at least two independent
    junk signals (ref_<hex> / sync / 30s / short-hash
    fragment).  The dual-signal requirement protects clean English
    prose that legitimately discusses "synchronous" systems or a
    "30s interval" from being swept up.
    """

    if len(paragraph) < 80:
        return False
    if not _DIAGRAM_MARKER_RE.search(paragraph):
        return False
    signals = 0
    if len(re.findall(r"ref_[a-fA-F0-9]{6,}", paragraph)) >= 1:
        signals += 1
    if len(re.findall(r"\bsync\b", paragraph)) >= 2:
        signals += 1
    if len(re.findall(r"\b30s\b", paragraph)) >= 2:
        signals += 1
    if len(_HASH_FRAGMENT_RE.findall(paragraph)) >= 2:
        signals += 1
    return signals >= 2


def _scrub_generation_artifacts(content: str) -> str:
    """Strip sub-agent / tool-layer junk tokens that outran upstream parsers.

    Three failure modes escape the per-paragraph
    _strip_content_envelope and the citation renumberer, and this
    scrubber handles them as a deterministic last line of defence
    before the section leaves the writer:

    1. Multiline {"content": "..."} envelope leaked between
       paragraphs.  The per-paragraph unwrap in
       _consolidate_short_paragraphs only catches envelopes that
       fit a single \\n\\n-delimited block; multiline envelopes
       survive and show up as literal JSON in the final Markdown.
    2. Fenced code blocks (typically mermaid) whose body devolved into
       repeated ref_<hex> / sync / 30s tokens.  These are
       runtime plumbing artifacts, never valid diagrams; the whole
       block is dropped.
    3. Orphan ref_<hex> fragments in prose outside a well-formed
       [ref_<hex>] citation.  These never round-trip through the
       citation renumberer and always surface as visible garbage.

    Idempotent: calling on already-clean prose returns it unchanged
    modulo blank-line collapse.
    """

    if not content:
        return content
    # Step 1: unwrap multiline JSON envelope leaks.  The envelope
    # scaffold ({\n  "content": "..." }) is removed but the
    # inner prose survives with JSON escapes un-escaped.  Pure
    # deletion would strip legitimate prose when the envelope is
    # malformed but still carries body content.
    content = _unwrap_envelope_leaks(content)

    # Step 2: wipe fenced blocks whose body is dominated by junk
    # tokens.  Thresholds chosen so a legitimate mermaid or code
    # sample with a single sync keyword or citation is retained.
    def _maybe_wipe_fence(match: re.Match[str]) -> str:
        body = match.group(1) or ""
        ref_hits = len(re.findall(r"ref_[a-fA-F0-9]{6,}", body))
        sync_hits = len(re.findall(r"\bsync\b", body))
        retry_hits = len(re.findall(r"\b30s\b", body))
        if ref_hits >= 2 or sync_hits >= 3 or retry_hits >= 2:
            return ""
        return match.group(0)

    content = _FENCED_BLOCK_RE.sub(_maybe_wipe_fence, content)

    # Step 2.5: drop *paragraph-level* junk that never got a code fence
    # around it.  Upstream fence loss in the sub-agent occasionally
    # leaves a "graph TD" stanza plus repeated ref_<hex> / 30s
    # / sync fragments sitting in prose form, so the fenced-block
    # scrub cannot reach them.  We require MULTIPLE independent
    # signals so clean prose that happens to mention "synchronous"
    # or "30 seconds" cannot trip this path.
    cleaned_paragraphs: list[str] = []
    for paragraph in content.split("\n\n"):
        if _looks_like_junk_paragraph(paragraph):
            continue
        cleaned_paragraphs.append(paragraph)
    content = "\n\n".join(cleaned_paragraphs)

    # Step 3: scrub orphan ref_<hex> tokens while preserving valid
    # bracketed citations.  Mask the valid tokens with a NUL-delimited
    # placeholder so the orphan regex cannot swallow them, then
    # restore after the scrub.
    masked: list[str] = []

    def _mask_valid(match: re.Match[str]) -> str:
        masked.append(match.group(0))
        return f"\x00VR{len(masked) - 1}\x00"

    scratch = _VALID_REF_CITATION.sub(_mask_valid, content)
    scratch = _ORPHAN_REF_RE.sub("", scratch)
    for idx, token in enumerate(masked):
        scratch = scratch.replace(f"\x00VR{idx}\x00", token)
    content = scratch

    # Step 4: collapse runs of blank lines created by the deletions
    # and strip trailing per-line whitespace to keep the output tidy.
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = "\n".join(line.rstrip() for line in content.splitlines())
    return content.strip("\n")


def _cjk_char_ratio(text: str) -> float:
    """Return the fraction of letters in text that are CJK ideographs.

    Counts both CJK characters and ASCII letters; digits, punctuation,
    and whitespace are ignored so structural markup does not bias the
    ratio.  Returns 0.0 for empty or letter-free text.
    """

    cjk = 0
    ascii_letters = 0
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            cjk += 1
        elif ch.isascii() and ch.isalpha():
            ascii_letters += 1
    total = cjk + ascii_letters
    if total == 0:
        return 0.0
    return cjk / total


def _query_is_english(query: str) -> bool:
    """Return True when query is predominantly English (i.e. not CJK).

    Used to gate English-only post-processing (the translate pass).
    Counts only letters and CJK ideographs so punctuation and digits do
    not bias the ratio.  A query with a handful of CJK chars (e.g. a
    brand name) still counts as English when ASCII letters dominate
    beyond _EN_QUERY_ASCII_RATIO.
    """

    ascii_letters = 0
    cjk_chars = 0
    for ch in query:
        if ch.isascii() and ch.isalpha():
            ascii_letters += 1
        elif (
            0x4E00 <= ord(ch) <= 0x9FFF
            or 0x3400 <= ord(ch) <= 0x4DBF
            or 0xF900 <= ord(ch) <= 0xFAFF
        ):
            cjk_chars += 1
    total = ascii_letters + cjk_chars
    if total == 0:
        return False
    return ascii_letters / total >= _EN_QUERY_ASCII_RATIO


def _build_section_prompt_context(
    *,
    plan: ResearchPlan,
    outline: list[OutlineSection],
    section_index: int,
) -> dict[str, str]:
    context = _default_section_prompt_context()
    total_sections = len(outline)
    if 0 <= section_index < total_sections:
        context["section_position"] = f"{section_index + 1} of {total_sections}"
        if section_index > 0:
            context["previous_section"] = outline[section_index - 1].title
        if section_index + 1 < total_sections:
            context["next_section"] = outline[section_index + 1].title
        related_ids = outline[section_index].related_question_ids
        question_lookup = {
            question.question_id: question.question for question in plan.sub_questions
        }
        related = [question_lookup[qid] for qid in related_ids if qid in question_lookup]
        if related:
            context["related_questions"] = _format_bundle_lines(related[:3])
    return context


def _build_reserve_evidence(
    selected_sources: list[SourceReference],
    ranked_sources: list[SourceReference],
    coverage_contract: AnswerCoverageContract,
    unresolved_gaps: list[str],
    *,
    cap: int = 2,
) -> list[SourceReference]:
    if not unresolved_gaps:
        return []
    selected_ids = {src.reference_id for src in selected_sources}
    unresolved = set(unresolved_gaps)
    reserve: list[SourceReference] = []
    for facet in coverage_contract.must_cover_facets:
        if facet.name not in unresolved:
            continue
        for src in ranked_sources:
            if src.reference_id in selected_ids or src in reserve:
                continue
            if _facet_match_count(src, facet) <= 0:
                continue
            reserve.append(src)
            break
        if len(reserve) >= cap:
            break
    return reserve


def _facet_match_count(src: SourceReference, facet: Any) -> int:
    terms = _extract_section_keywords(
        f"{facet.name} {facet.intent} {facet.evidence_hint} {' '.join(facet.bilingual_terms)}"
    )
    if not terms:
        return 0
    text = f"{src.title} {src.snippet} {src.url or ''}".lower()
    return sum(1 for term in terms if term in text)


def _missing_bundle_facets(
    selected_sources: list[SourceReference],
    coverage_contract: AnswerCoverageContract,
) -> list[str]:
    missing: list[str] = []
    for facet in coverage_contract.must_cover_facets:
        matched = False
        for src in selected_sources:
            if _facet_match_count(src, facet) >= 2:
                matched = True
                break
        if not matched:
            missing.append(facet.name)
    return missing


def _build_soft_checklist(bundle: SectionEvidenceBundle) -> str:
    """Build a compact reference checklist from the evidence bundle.

    Appended after the core writing rules as optional guidance.  The model
    treats these as analytical hints rather than mandatory acknowledgments,
    which preserves writing depth instead of triggering obligation-checking
    compression.
    """
    _MAX_CHECKLIST_ITEMS = 3
    parts: list[str] = []
    if bundle.coverage_facets and len(parts) < _MAX_CHECKLIST_ITEMS:
        items = ", ".join(bundle.coverage_facets[:3])
        parts.append(f"- Topics to address where relevant: {items}")
    if bundle.comparison_axes and len(parts) < _MAX_CHECKLIST_ITEMS:
        items = ", ".join(bundle.comparison_axes[:2])
        parts.append(f"- Comparison angles: {items}")
    if bundle.unresolved_gaps and len(parts) < _MAX_CHECKLIST_ITEMS:
        items = ", ".join(bundle.unresolved_gaps[:2])
        parts.append(f"- Evidence gaps: {items}")
    if bundle.caveat_obligations and len(parts) < _MAX_CHECKLIST_ITEMS:
        items = ", ".join(bundle.caveat_obligations[:2])
        parts.append(f"- Key caveats: {items}")
    # Archetype-specific analytical depth hint — not counted against the
    # topical item cap because it guides structure, not content topics.
    archetype_hint = _ARCHETYPE_ANALYSIS_HINTS.get(bundle.section_archetype, "")
    if archetype_hint:
        parts.append(f"- ARCHETYPE GUIDANCE ({bundle.section_archetype}): {archetype_hint}")
    if not parts:
        return ""
    return (
        "\n- REFERENCE GUIDANCE (interpret in the report's target language; "
        "use as analytical hints, not rigid obligations):\n" + "\n".join(parts)
    )


def _format_bundle_sources(sources: list[SourceReference], snippet_cap: int) -> str:
    if not sources:
        return "(none)"
    return "\n".join(
        f"- {src.reference_id} | {src.title} | {(src.snippet or '')[:snippet_cap]}"
        for src in sources
    )


def _format_bundle_lines(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _fill_section_sources(
    candidate_pool: list[SourceReference],
    *,
    selected: list[SourceReference],
    seen_ids: set[str],
    seen_domains: set[str],
    max_sources: int,
    policy: SectionEvidencePolicy,
    prefer_new_domain: bool,
    stop_when_domain_count: int | None = None,
) -> None:
    for src in candidate_pool:
        if stop_when_domain_count is not None and len(seen_domains) >= stop_when_domain_count:
            break
        _try_add_section_source(
            src,
            selected=selected,
            seen_ids=seen_ids,
            seen_domains=seen_domains,
            max_sources=max_sources,
            policy=policy,
            prefer_new_domain=prefer_new_domain,
        )


def _ensure_cross_question_sources(
    ranked: list[SourceReference],
    *,
    selected: list[SourceReference],
    seen_ids: set[str],
    seen_domains: set[str],
    max_sources: int,
    policy: SectionEvidencePolicy,
    source_question_counts: dict[str, int],
) -> None:
    current = sum(1 for src in selected if source_question_counts.get(src.reference_id, 0) > 1)
    if current >= policy.min_cross_question_sources:
        return
    for src in ranked:
        if source_question_counts.get(src.reference_id, 0) <= 1:
            continue
        before = len(selected)
        _try_add_section_source(
            src,
            selected=selected,
            seen_ids=seen_ids,
            seen_domains=seen_domains,
            max_sources=max_sources,
            policy=policy,
            prefer_new_domain=False,
        )
        if len(selected) == before:
            continue
        current += 1
        if current >= policy.min_cross_question_sources:
            break


def _ensure_authority_sources(
    ranked: list[SourceReference],
    *,
    selected: list[SourceReference],
    seen_ids: set[str],
    seen_domains: set[str],
    max_sources: int,
    policy: SectionEvidencePolicy,
) -> None:
    current = sum(1 for src in selected if _looks_authoritative_source(src))
    if current >= policy.min_authority_sources:
        return
    for src in ranked:
        if not _looks_authoritative_source(src):
            continue
        before = len(selected)
        _try_add_section_source(
            src,
            selected=selected,
            seen_ids=seen_ids,
            seen_domains=seen_domains,
            max_sources=max_sources,
            policy=policy,
            prefer_new_domain=False,
        )
        if len(selected) == before:
            continue
        current += 1
        if current >= policy.min_authority_sources:
            break


def _ensure_counter_evidence_sources(
    ranked: list[SourceReference],
    *,
    selected: list[SourceReference],
    seen_ids: set[str],
    seen_domains: set[str],
    max_sources: int,
    policy: SectionEvidencePolicy,
) -> None:
    current = sum(1 for src in selected if _looks_counter_evidence(src))
    if current >= policy.min_counter_evidence_sources:
        return
    for src in ranked:
        if not _looks_counter_evidence(src):
            continue
        before = len(selected)
        _try_add_section_source(
            src,
            selected=selected,
            seen_ids=seen_ids,
            seen_domains=seen_domains,
            max_sources=max_sources,
            policy=policy,
            prefer_new_domain=False,
        )
        if len(selected) == before:
            continue
        current += 1
        if current >= policy.min_counter_evidence_sources:
            break


def _ensure_facet_coverage_sources(
    ranked: list[SourceReference],
    *,
    selected: list[SourceReference],
    seen_ids: set[str],
    seen_domains: set[str],
    max_sources: int,
    policy: SectionEvidencePolicy,
    coverage_contract: AnswerCoverageContract,
) -> None:
    missing = set(_missing_bundle_facets(selected, coverage_contract))
    if not missing:
        return
    for facet in coverage_contract.must_cover_facets:
        if facet.name not in missing:
            continue
        for src in ranked:
            if _facet_match_count(src, facet) <= 0:
                continue
            before = len(selected)
            _try_add_section_source(
                src,
                selected=selected,
                seen_ids=seen_ids,
                seen_domains=seen_domains,
                max_sources=max_sources,
                policy=policy,
                prefer_new_domain=False,
            )
            if len(selected) > before:
                break


def _try_add_section_source(
    src: SourceReference,
    *,
    selected: list[SourceReference],
    seen_ids: set[str],
    seen_domains: set[str],
    max_sources: int,
    policy: SectionEvidencePolicy,
    prefer_new_domain: bool,
) -> None:
    if len(selected) >= max_sources or src.reference_id in seen_ids:
        return
    if policy.require_content_usable and not _is_content_usable(src):
        return
    domain = _source_domain(src)
    if prefer_new_domain and domain and domain in seen_domains:
        return
    selected.append(src)
    seen_ids.add(src.reference_id)
    if domain:
        seen_domains.add(domain)


def _parse_section(title: str, content: str) -> ReportSection:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()

    data = _try_parse_json(text)
    if data and "content" in data:
        citations = [
            Citation(
                reference_id=c.get("reference_id", ""),
                text_span=c.get("text_span", ""),
                context=c.get("context", ""),
            )
            for c in data.get("citations", [])
        ]
        body = _normalize_section_body(data["content"], title)
        return ReportSection(title=title, content=body, citations=citations)
    # Last-resort recovery: text looks like a {"content": ...} envelope but
    # json.loads could not parse it. Writers sometimes emit unescaped
    # newlines or quotes inside a long content string (typical of the final
    # caveats section), or they hit max_tokens mid-string leaving the
    # envelope unclosed. Repair by regex-extracting the content field so the
    # envelope tokens do not leak into the final report.
    repaired = _repair_content_envelope(text)
    if repaired is not None:
        return ReportSection(title=title, content=_normalize_section_body(repaired, title))
    return ReportSection(title=title, content=_normalize_section_body(text, title))


def _extract_translated_title(raw_response: str, *, fallback: str) -> str:
    """Return the translated section title from a translate LLM JSON response.

    The translate prompt asks the model to return {"title": "...",
    "content": "..."}.  _parse_section only consumes the
    content field, so we extract the optional title field here
    to feed back into the caller's ReportSection.title.

    Falls back to fallback whenever the response is unparseable, the
    title field is missing / blank, or the translated title still
    carries CJK characters (defensive guard against a misbehaving model).
    The fallback keeps the caller safe: a failed title translation never
    regresses the rendered heading below its pre-translation state.
    """

    text = raw_response.strip()
    if text.startswith("```"):
        # Strip fenced code block wrapper, mirroring _parse_section.
        first_nl = text.find("\n")
        last_fence = text.rfind("```")
        if first_nl >= 0 and last_fence > first_nl:
            text = text[first_nl + 1 : last_fence].strip()
    data = _try_parse_json(text)
    if not data:
        return fallback
    title = data.get("title")
    if not isinstance(title, str):
        return fallback
    title = title.strip()
    if not title:
        return fallback
    # Reject any title that still has CJK characters — the translation
    # gate exists precisely to drive CJK out of headings.  Keep the
    # original title rather than ship a regression.
    if _cjk_char_ratio(title) > 0.0:
        return fallback
    return title


def _normalize_section_body(body: str, title: str) -> str:
    """Normalize a writer-produced section body before it reaches the report.

    Applies stable, minimal text fixups for recurring writer output defects
    that otherwise leak into the rendered report and the bench article
    export. Kept as a single chokepoint so each class of defect has one
    well-tested repair site.
    """

    body = _strip_envelope_citations_trailer(body)
    body = _normalize_citation_groups(body)
    body = _balance_fence_markers(body)
    return _strip_leading_heading(title, body)


# Line-start triple-backtick fence marker. Allows a leading indent so
# fences nested inside list items still count, and captures any trailing
# language tag on the same line ( / python / ```mermaid / etc.).
_FENCE_MARKER_RE = re.compile(r"(?m)^[ \t]*```[^\n]*$")
# Four-space (or tab) leading indent — markdown's rule for an implicit
# "indented code block". Used to detect an orphan diagram body that
# precedes a dangling closing fence.
_INDENTED_CODE_LINE_RE = re.compile(r"^(?: {4,}|\t)")
# Headers and arrow operators unique to Mermaid. When an orphan code
# block carries any of these markers we can safely restore the opening
# fence with a mermaid language tag so the UI renders a real diagram
# instead of a nameless code stub.
_MERMAID_HEADER_RE = re.compile(
    r"\b(?:sequenceDiagram|flowchart|graph\s+(?:TD|LR|BT|RL)|stateDiagram"
    r"|classDiagram|erDiagram|gantt|pie|gitGraph|journey|timeline|participant)\b"
)
_MERMAID_ARROW_RE = re.compile(r"->>|-->|-\.->|==>")


def _infer_fence_language(block: str) -> str:
    """Best-effort language tag for an orphan code block.

    Returns "mermaid" when the block body contains the diagram
    keywords or arrow operators specific to Mermaid, otherwise an empty
    string to keep the restored fence generic.
    """

    if _MERMAID_HEADER_RE.search(block) or _MERMAID_ARROW_RE.search(block):
        return "mermaid"
    return ""


def _balance_fence_markers(text: str) -> str:
    """Pair an orphan trailing  fence with a restored opener.

    Writers occasionally emit a closing  without a matching opener
    (observed most often when the intended block was a Mermaid diagram
    and the opener was lost). Two failure modes then appear in the UI:
    the raw closer renders as the *opening* of a fresh code block that
    never closes, or, once the closer is stripped, the indented diagram
    body below it falls back to markdown's implicit indented-code-block
    rule and renders as a nameless code stub.

    When the orphan trailing fence is preceded by a contiguous indented
    block, restore an opening fence above that block (with a mermaid
    tag when the body looks like a Mermaid diagram) so the content
    renders as a proper labelled code block. Fall back to dropping the
    orphan fence when no indented block precedes it — that covers
    truncated \u0060\u0060\u0060json wrappers whose body was already
    cut by the citations-trailer stripper upstream.
    """

    matches = list(_FENCE_MARKER_RE.finditer(text))
    if len(matches) % 2 == 0:
        return text
    last = matches[-1]
    pre = text[: last.start()]
    post = text[last.end() :]

    lines = pre.splitlines(keepends=True)
    end_idx = len(lines) - 1
    while end_idx >= 0 and lines[end_idx].strip() == "":
        end_idx -= 1

    if end_idx < 0 or not _INDENTED_CODE_LINE_RE.match(lines[end_idx]):
        before = pre.rstrip("\n")
        after = post.lstrip("\n")
        if before and after:
            return f"{before}\n\n{after}"
        return before or after

    # Walk upward while the preceding line is still inside the block
    # (indented lines plus blanks sandwiched between them).
    start_idx = end_idx
    while start_idx > 0:
        prev = lines[start_idx - 1]
        if prev.strip() == "" or _INDENTED_CODE_LINE_RE.match(prev):
            start_idx -= 1
        else:
            break
    while start_idx <= end_idx and lines[start_idx].strip() == "":
        start_idx += 1

    head = "".join(lines[:start_idx]).rstrip("\n")
    block_lines = lines[start_idx : end_idx + 1]
    block = "".join(block_lines).rstrip("\n")
    tail = post.lstrip("\n")

    lang = _infer_fence_language(block)
    opener = f"```{lang}" if lang else "```"
    parts: list[str] = []
    if head:
        parts.append(head)
    parts.append(f"{opener}\n{block}\n```")
    if tail:
        parts.append(tail)
    return "\n\n".join(parts)


def _strip_envelope_citations_trailer(text: str) -> str:
    """Truncate an escaped citations-array trailer that leaked into content.

    Some writers produce a malformed envelope where the citations
    array is double-nested: once escaped inside the content string
    and once as a sibling JSON field. json.loads still succeeds on
    the outer envelope, but the content value carries the escaped
    trailer verbatim::

        ...final sentence.",\n  "citations": [\n    {\n      ...

    Cut the content at the first ","citations": boundary so the
    trailer does not render into the final section. Returns the input
    unchanged when no boundary is found.
    """

    match = _CITATIONS_SEPARATOR.search(text)
    if match is None:
        return text
    # match.start() points at the stray closing quote that belonged to
    # the envelope's content field, not the prose. Drop it along with
    # the trailer and any trailing whitespace to avoid a dangling newline
    # between the salvaged prose and whatever follows.
    return text[: match.start()].rstrip()


# Matches a single bracketed group of two or more comma-separated ref
# identifiers, e.g. [ref_a1b2c3d4, ref_e5f6a7b8]. Single-ref tokens are
# intentionally excluded so we leave the already-correct form alone.
_COMMA_GROUP_CITATION_RE = re.compile(r"\[(ref_[a-f0-9]+(?:\s*,\s*ref_[a-f0-9]+)+)\]")


def _normalize_citation_groups(text: str) -> str:
    """Expand comma-grouped ref citations into atomic bracketed tokens.

    Writers occasionally emit [ref_a, ref_b, ref_c] to cite several
    references at the same position. The bench exporter and the
    downstream citation resolver only match the single-ref form
    [ref_a], so comma groups previously survived into the final
    article as literal noise. Normalize them to [ref_a][ref_b][ref_c]
    so every citation is an atomic resolvable token.
    """

    def _expand(match: re.Match[str]) -> str:
        ids = [token.strip() for token in match.group(1).split(",")]
        return "".join(f"[{rid}]" for rid in ids if rid)

    return _COMMA_GROUP_CITATION_RE.sub(_expand, text)


def _strip_content_envelope(text: str) -> str:
    """Unwrap a {"content": "..."} envelope emitted by the rewrite LLM.

    Returns the extracted content when the input looks like a writer
    envelope (parsable JSON with a content string, or a malformed
    envelope recoverable by _repair_content_envelope). Falls back to
    the original text otherwise. Idempotent on already-plain prose.

    Centralises the parse → repair → fallback cascade so
    _parse_section and _rewrite_noisy_paragraph both strip the
    same envelope shape without code duplication.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    data = _try_parse_json(stripped)
    if data and isinstance(data.get("content"), str):
        return data["content"]
    repaired = _repair_content_envelope(stripped)
    if repaired is not None:
        return repaired
    return text


def _try_parse_json(text: str) -> dict | None:
    """Best-effort JSON extraction: try full text, then brace-delimited substring."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            data = json.loads(text[first : last + 1])
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# Matches the opening of a writer-output JSON envelope. Permissive on
# whitespace between the brace, the key, and the colon so pretty-printed
# objects still get repaired.
_CONTENT_ENVELOPE_OPEN = re.compile(r"^\s*\{\s*\"content\"\s*:\s*\"", re.DOTALL)
# Matches the tail of a fully-formed envelope: a quote plus optional
# whitespace plus the closing brace at end of string.
_CONTENT_ENVELOPE_TAIL = re.compile(r"\"\s*\}\s*$", re.DOTALL)
# Matches the boundary preceding a "citations" field when present.
_CITATIONS_SEPARATOR = re.compile(r"\"\s*,\s*\"citations\"\s*:", re.DOTALL)


def _repair_content_envelope(text: str) -> str | None:
    """Salvage the content field from an unparsable writer envelope.

    Writers are prompted to emit {"content": "...", "citations": [...]}.
    Two failure modes cause json.loads to reject the output:

    1. Unescaped double quotes or raw newlines inside a long prose body.
    2. The writer hit max_tokens mid-string, leaving no closing quote,
       comma, or brace — so neither the tail pattern nor the citations
       separator can be found.

    The repair path: detect the opener, find the best available right
    boundary (citations separator, tail brace, or end-of-text for the
    truncation case), and return the enclosed body with minimal unescaping.
    Returns None when the envelope shape is not detected so callers can
    fall back to returning the text unchanged.
    """

    open_match = _CONTENT_ENVELOPE_OPEN.match(text)
    if not open_match:
        return None
    body_start = open_match.end()
    citations = _CITATIONS_SEPARATOR.search(text, body_start)
    if citations is not None:
        body_end = citations.start()
    else:
        tail = _CONTENT_ENVELOPE_TAIL.search(text, body_start)
        if tail is not None:
            body_end = tail.start()
        else:
            # Truncation: no tail token exists. Salvage everything after the
            # opener and drop any trailing partial JSON escape (a lone
            # backslash) so unescaping below does not create a phantom char.
            body_end = len(text)
            while body_end > body_start and text[body_end - 1] == "\\":
                body_end -= 1
    if body_end <= body_start:
        return None
    body = text[body_start:body_end]
    # Undo the minimal escapes a writer might have produced without breaking
    # markdown. Leave \\ sequences alone because content is plain prose,
    # not regex, and over-unescaping risks corrupting URLs.
    body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    cleaned = body.strip()
    if not cleaned:
        return None
    return cleaned


def _strip_leading_heading(title: str, content: str) -> str:
    """Remove a leading markdown heading that duplicates the section title."""
    norm_title = title.strip().lower()
    pattern = re.compile(r"^#{1,6}\s+(.+)")
    lines = content.split("\n")
    stripped: list[str] = []
    removed_first = False
    for line in lines:
        if not removed_first:
            m = pattern.match(line.strip())
            if m and m.group(1).strip().lower() == norm_title:
                removed_first = True
                continue
            if line.strip() == "":
                stripped.append(line)
                continue
        stripped.append(line)
        removed_first = True
    return "\n".join(stripped).strip()
