"""ReportGenerator — structured Markdown report with citations.

Generates reports section-by-section via LLM, annotating inline citations
that link back to ``SourceReference`` entries. Supports both batch and
streaming (``AsyncIterator[ReportChunk]``) output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from houyi.adapters.llm.base import LLMAdapter
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

_SENTENCE_TERMINATORS = re.compile(r"[.!?]\s|[.!?]$")


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
        summary.  Call :meth:`complete_summary` later to fill it in.  This
        allows the caller to overlap summary generation with other work
        (e.g. validation) since neither validation nor repair reads the
        summary field.

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
        precedence over the keyword-based ``_classify_section_archetype``
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
        # Post-generation noise detection pass.  Rewrites only noisy paragraphs
        # to keep latency bounded (~1 LLM call per noisy paragraph, typically 0-1).
        available_refs = [s.reference_id for s in sources]
        section.content = await self._clean_section_noise(
            section.content,
            title=title,
            objective=objective,
            available_refs=available_refs,
        )
        # Structural postprocess: the only content-mutating guard is the
        # paragraph consolidator (human-visible layout repair). Critical and
        # visualization signals are computed later by
        # ``_compute_section_sidecar_metrics`` and emitted as observability
        # only, so they do not leak HTML markers into the scored article.
        section.content = _consolidate_short_paragraphs(section.content)
        return section

    async def _clean_section_noise(
        self,
        content: str,
        *,
        title: str,
        objective: str,
        available_refs: list[str],
    ) -> str:
        """Detect and micro-rewrite noisy paragraphs in generated section content.

        Detection uses regex/heuristic (zero LLM cost).  Only flagged paragraphs
        trigger a targeted LLM rewrite call, keeping the latency bounded.
        """
        paragraphs = content.split("\n\n")
        if not paragraphs:
            return content
        noisy_indices = _detect_noisy_paragraphs(paragraphs)
        if not noisy_indices:
            return content
        ref_str = ", ".join(available_refs[:20])
        for idx in noisy_indices:
            rewritten = await self._rewrite_noisy_paragraph(
                paragraphs[idx],
                title=title,
                objective=objective,
                available_refs=ref_str,
            )
            paragraphs[idx] = rewritten
        # Drop paragraphs that were entirely noise (rewritten to empty).
        joined = "\n\n".join(p for p in paragraphs if p.strip())
        # LLM rewrites occasionally re-emit comma-grouped ``[ref_a, ref_b]``
        # citations that bypass the write-path normalization pipeline.
        # Re-apply the citation-group normalization here so no call site can
        # skip it accidentally by wiring directly to _clean_section_noise.
        return _normalize_citation_groups(joined)

    async def _rewrite_noisy_paragraph(
        self,
        paragraph: str,
        *,
        title: str,
        objective: str,
        available_refs: str,
    ) -> str:
        """Targeted LLM rewrite for a single noisy paragraph."""
        prompt = _NOISE_REWRITE_PROMPT.format(
            title=title,
            objective=objective,
            available_refs=available_refs,
            paragraph=paragraph,
        )
        try:
            resp = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
                **self._llm_kwargs,
            )
            result = resp.content.strip()
            if result.lower() in ("(empty)", ""):
                return ""
            # Normalize comma-grouped citations at the earliest point so
            # every downstream consumer sees atomic ``[ref_x]`` tokens.
            return _normalize_citation_groups(result)
        except Exception:
            logger.warning("Noise rewrite failed for section '%s'", title, exc_info=True)
            return paragraph

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
# * ``_consolidate_short_paragraphs`` merges adjacent short paragraphs so the
#   reader sees cohesive multi-sentence blocks instead of fragmented lines.
#   This is the only helper that mutates ``section.content`` — it performs a
#   human-visible layout repair.
# * ``_analyse_critical_analysis`` checks whether any critical-analysis
#   keyword is present in the body and returns a boolean flag.
# * ``_analyse_visualization_gaps`` checks whether the body lacks an expected
#   table (numeric-dense prose) or mermaid fence (hierarchy/sequence prose)
#   and returns the gap flags.
#
# The ``_analyse_*`` helpers **never mutate the content**. Their signals are
# collected by ``_compute_section_sidecar_metrics`` and surface through
# ``section_input_metrics`` for offline bench analysis. Writing the signals
# into ``section.content`` as HTML comments would leak them into the scored
# article text when upstream RACE cleaning falls back to the raw body.
# ---------------------------------------------------------------------------


_MERGEABLE_MIN_SENTENCES = 3
_MERGEABLE_TARGET_SENTENCES = 5
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


def _consolidate_short_paragraphs(content: str) -> str:
    """Merge adjacent short paragraphs up to a readable target length.

    Structural paragraphs (headings, lists, tables, code fences, existing
    HTML comments) are never merged. The merge target is expressed in
    sentences so the heuristic remains language-agnostic.
    """
    if not content:
        return content
    paragraphs = content.split("\n\n")
    if len(paragraphs) < 2:
        return content
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


def _analyse_visualization_gaps(content: str) -> dict[str, bool]:
    """Return structural signals describing missing visualisations.

    The result carries two boolean flags, both observational only:

    * ``needs_table`` — numeric token density meets
      ``_VISUALIZATION_NUMERIC_THRESHOLD`` but the body contains no markdown
      table row, hinting the writer should prefer a compact table;
    * ``needs_mermaid`` — any hierarchy/sequence keyword from
      ``SECTION_VISUAL_TRIGGER_KEYWORDS`` is present but the body contains no
      ``mermaid`` fence, hinting a flowchart could clarify the relationships.

    The function never mutates ``content``. Callers emit these flags as
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

    Keywords come from ``SECTION_CRITICAL_ANALYSIS_KEYWORDS`` (bilingual,
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
    Attached to ``section_input_metrics`` for offline analysis during
    benchmark runs — never enters the main scoring or generation path.

    Metrics produced:

    - ``sidecar_bullet_line_ratio``: fraction of non-empty lines that are
      bullet/list items.  High values (>0.5) indicate the LLM ignored the
      "prefer dense analytical paragraphs" instruction.
    - ``sidecar_avg_paragraph_sentences``: average sentence count per
      substantive paragraph.  Below 3 indicates shallow paragraphs.
    - ``sidecar_bold_heading_count``: Markdown ``###`` headings found.
    - ``sidecar_citation_count`` / ``sidecar_unique_citation_count``:
      total and unique inline ``[ref_xxx]`` citations.
    - ``sidecar_uncited_paragraph_count``: substantive paragraphs without
      any inline citation.
    - ``sidecar_word_count``: total whitespace-delimited tokens.
    - ``sidecar_archetype``: the resolved archetype label.
    - ``sidecar_archetype_compliant``: whether archetype-specific keywords
      were detected (English-only; CJK content yields ``false``, which is
      itself a useful diagnostic signal).
    - ``sidecar_critical_analysis_present``: whether any bilingual
      critical-analysis keyword (``SECTION_CRITICAL_ANALYSIS_KEYWORDS``) is
      present in the body.
    - ``sidecar_visualization_needs_table`` /
      ``sidecar_visualization_needs_mermaid``: whether the body is dense in
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
# language tag on the same line (``` / ```python / ```mermaid / etc.).
_FENCE_MARKER_RE = re.compile(r"(?m)^[ \t]*```[^\n]*$")
# Four-space (or tab) leading indent — markdown's rule for an implicit
# "indented code block". Used to detect an orphan diagram body that
# precedes a dangling closing fence.
_INDENTED_CODE_LINE_RE = re.compile(r"^(?: {4,}|\t)")
# Headers and arrow operators unique to Mermaid. When an orphan code
# block carries any of these markers we can safely restore the opening
# fence with a ``mermaid`` language tag so the UI renders a real diagram
# instead of a nameless code stub.
_MERMAID_HEADER_RE = re.compile(
    r"\b(?:sequenceDiagram|flowchart|graph\s+(?:TD|LR|BT|RL)|stateDiagram"
    r"|classDiagram|erDiagram|gantt|pie|gitGraph|journey|timeline|participant)\b"
)
_MERMAID_ARROW_RE = re.compile(r"->>|-->|-\.->|==>")


def _infer_fence_language(block: str) -> str:
    """Best-effort language tag for an orphan code block.

    Returns ``"mermaid"`` when the block body contains the diagram
    keywords or arrow operators specific to Mermaid, otherwise an empty
    string to keep the restored fence generic.
    """

    if _MERMAID_HEADER_RE.search(block) or _MERMAID_ARROW_RE.search(block):
        return "mermaid"
    return ""


def _balance_fence_markers(text: str) -> str:
    """Pair an orphan trailing ``` fence with a restored opener.

    Writers occasionally emit a closing ``` without a matching opener
    (observed most often when the intended block was a Mermaid diagram
    and the opener was lost). Two failure modes then appear in the UI:
    the raw closer renders as the *opening* of a fresh code block that
    never closes, or, once the closer is stripped, the indented diagram
    body below it falls back to markdown's implicit indented-code-block
    rule and renders as a nameless ``code`` stub.

    When the orphan trailing fence is preceded by a contiguous indented
    block, restore an opening fence above that block (with a ``mermaid``
    tag when the body looks like a Mermaid diagram) so the content
    renders as a proper labelled code block. Fall back to dropping the
    orphan fence when no indented block precedes it — that covers
    truncated ``\u0060\u0060\u0060json`` wrappers whose body was already
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

    Some writers produce a malformed envelope where the ``citations``
    array is double-nested: once escaped inside the ``content`` string
    and once as a sibling JSON field. ``json.loads`` still succeeds on
    the outer envelope, but the ``content`` value carries the escaped
    trailer verbatim::

        ...final sentence.",\n  "citations": [\n    {\n      ...

    Cut the content at the first ``","citations":`` boundary so the
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
# identifiers, e.g. ``[ref_a1b2c3d4, ref_e5f6a7b8]``. Single-ref tokens are
# intentionally excluded so we leave the already-correct form alone.
_COMMA_GROUP_CITATION_RE = re.compile(r"\[(ref_[a-f0-9]+(?:\s*,\s*ref_[a-f0-9]+)+)\]")


def _normalize_citation_groups(text: str) -> str:
    """Expand comma-grouped ref citations into atomic bracketed tokens.

    Writers occasionally emit ``[ref_a, ref_b, ref_c]`` to cite several
    references at the same position. The bench exporter and the
    downstream citation resolver only match the single-ref form
    ``[ref_a]``, so comma groups previously survived into the final
    article as literal noise. Normalize them to ``[ref_a][ref_b][ref_c]``
    so every citation is an atomic resolvable token.
    """

    def _expand(match: re.Match[str]) -> str:
        ids = [token.strip() for token in match.group(1).split(",")]
        return "".join(f"[{rid}]" for rid in ids if rid)

    return _COMMA_GROUP_CITATION_RE.sub(_expand, text)


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
# Matches the boundary preceding a ``"citations"`` field when present.
_CITATIONS_SEPARATOR = re.compile(r"\"\s*,\s*\"citations\"\s*:", re.DOTALL)


def _repair_content_envelope(text: str) -> str | None:
    """Salvage the ``content`` field from an unparsable writer envelope.

    Writers are prompted to emit ``{"content": "...", "citations": [...]}``.
    Two failure modes cause ``json.loads`` to reject the output:

    1. Unescaped double quotes or raw newlines inside a long prose body.
    2. The writer hit ``max_tokens`` mid-string, leaving no closing quote,
       comma, or brace — so neither the tail pattern nor the citations
       separator can be found.

    The repair path: detect the opener, find the best available right
    boundary (citations separator, tail brace, or end-of-text for the
    truncation case), and return the enclosed body with minimal unescaping.
    Returns ``None`` when the envelope shape is not detected so callers can
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
    # markdown. Leave ``\\`` sequences alone because content is plain prose,
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
