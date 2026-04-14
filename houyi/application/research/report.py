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
from houyi.application.research.types import (
    AggregatedSources,
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


@dataclass(frozen=True, slots=True)
class SectionEvidencePolicy:
    """Policy for assembling a compact but more coverage-aware section evidence set."""

    candidate_pool_size: int = 12
    min_domain_diversity: int = 3
    require_content_usable: bool = True


if TYPE_CHECKING:
    from houyi.application.research.runtime.intermediate import IntermediateReport

_SECTION_PROMPT = """\
You are writing a section of an academic-grade research report.

Report title context: {query}
Section: {title}
Section objective: {objective}
Available sources (reference_id | title | snippet):
{sources_text}

Write the section in Markdown. Rules:
- Do NOT include a heading for this section (the heading is added externally).
- CITATION DISCIPLINE: Every factual claim, statistic, date, or attribution MUST \
have an inline citation as [ref_id]. A paragraph without citations is unacceptable. \
Use multiple citations when claims are supported by multiple sources.
- Only cite sources from the provided list. Do NOT fabricate reference IDs.
- ANALYSIS DEPTH: Go beyond summarizing — synthesize across sources, identify \
patterns, note contradictions, and provide analytical commentary.
- STRUCTURE: Use sub-headings (###), bullet points, or numbered lists to organize \
complex information. Include comparison tables when relevant.
- Write in the SAME language as the report title / query above. \
If the query is in Chinese, write in Chinese. If English, write in English.
- Use clear, professional, scholarly prose. Aim for 400-800 words per section.

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
        section_max_tokens: int = 2000,
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
                relevant, relevant_total, evidence_metrics = self.select_section_sources(
                    outline_sec.related_question_ids,
                    sources,
                    section_title=outline_sec.title,
                    objective=outline_sec.objective,
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
                    relevant,
                    intermediate_context=ir_context,
                )
                section.section_id = outline_sec.section_id
                return section, {
                    "section_id": outline_sec.section_id,
                    "title": outline_sec.title,
                    "relevant_source_count": relevant_total,
                    "selected_source_count": len(relevant),
                    **evidence_metrics,
                    "intermediate_context_chars": len(ir_context),
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
            )
            section = await self._generate_section(
                plan.query,
                outline_sec.title,
                outline_sec.objective,
                relevant,
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
    ) -> tuple[list[SourceReference], int, dict[str, int]]:
        """Select section sources via ranked recall, candidate-pool build, and final assembly."""

        ranked = _relevant_sources(
            question_ids,
            sources,
            section_title=section_title,
            objective=objective,
        )
        source_question_counts = _build_source_question_counts(sources)
        candidate_pool = _build_section_candidate_pool(
            ranked,
            max_sources=self._max_section_sources,
            policy=self._section_evidence_policy,
        )
        selected = _assemble_section_sources(
            candidate_pool,
            max_sources=self._max_section_sources,
            policy=self._section_evidence_policy,
        )
        return (
            selected,
            len(ranked),
            {
                "selected_domain_count": len(
                    {_source_domain(src) for src in selected if _source_domain(src)}
                ),
                "authority_source_count": sum(
                    1 for src in selected if _looks_authoritative_source(src)
                ),
                "cross_question_source_count": sum(
                    1 for src in selected if source_question_counts.get(src.reference_id, 0) > 1
                ),
                "content_usable_source_count": sum(
                    1 for src in selected if _is_content_usable(src)
                ),
            },
        )

    async def _generate_section(
        self,
        query: str,
        title: str,
        objective: str,
        sources: list[SourceReference],
        intermediate_context: str = "",
    ) -> ReportSection:
        snip = self._snippet_max_chars
        sources_text = "\n".join(
            f"  {s.reference_id} | {s.title} | {s.snippet[:snip]}"
            for s in sources[: self._max_source_display]
        )
        prompt = _SECTION_PROMPT.format(
            query=query,
            title=title,
            objective=objective,
            sources_text=sources_text or "(no sources)",
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
        return _parse_section(title, resp.content)

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
    if not ref_ids:
        candidates = list(agg.sources)
    else:
        candidates = [lookup[rid] for rid in ref_ids if rid in lookup]

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
    question_ids: list[str],
    source_question_counts: dict[str, int],
) -> float:
    keywords = _extract_section_keywords(f"{section_title} {objective}")
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
    *,
    max_sources: int,
    policy: SectionEvidencePolicy,
) -> list[SourceReference]:
    selected: list[SourceReference] = []
    seen_ids: set[str] = set()
    seen_domains: set[str] = set()

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

    return selected[:max_sources]


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
        body = _strip_leading_heading(title, data["content"])
        return ReportSection(title=title, content=body, citations=citations)
    return ReportSection(title=title, content=_strip_leading_heading(title, text))


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
