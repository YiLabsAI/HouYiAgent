"""ReportGenerator — structured Markdown report with citations.

Generates reports section-by-section via LLM, annotating inline citations
that link back to ``SourceReference`` entries. Supports both batch and
streaming (``AsyncIterator[ReportChunk]``) output.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import (
    AggregatedSources,
    Citation,
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

_SECTION_PROMPT = """\
You are writing a section of a research report.

Report title context: {query}
Section: {title}
Section objective: {objective}
Available sources (reference_id | title | snippet):
{sources_text}

Write the section in Markdown. Rules:
- Do NOT include a heading for this section (the heading is added externally).
- For every factual claim, insert an inline citation as [ref_id].
- Use multiple citations when appropriate.
- Write in the SAME language as the report title / query above. \
If the query is in Chinese, write in Chinese. If English, write in English.
- Use clear, professional prose.

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

    def __init__(self, llm_adapter: LLMAdapter, **llm_kwargs: Any) -> None:
        self._llm = llm_adapter
        self._llm_kwargs = llm_kwargs

    async def generate(
        self,
        plan: ResearchPlan,
        sources: AggregatedSources,
        style: ReportStyle = ReportStyle.DETAILED,
    ) -> ResearchReport:
        """Generate a complete report (non-streaming).

        Iterates over plan outline sections, calling the LLM once per section
        to produce content with inline citations.
        """
        start = time.monotonic()
        report_id = f"rpt_{uuid.uuid4().hex[:8]}"
        sections: list[ReportSection] = []

        for outline_sec in plan.outline:
            relevant = _relevant_sources(outline_sec.related_question_ids, sources)
            section = await self._generate_section(
                plan.query,
                outline_sec.title,
                outline_sec.objective,
                relevant,
            )
            sections.append(section)

        summary = await self._generate_summary(plan.query, sections)
        duration = time.monotonic() - start

        return ResearchReport(
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
            ),
        )

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

            relevant = _relevant_sources(outline_sec.related_question_ids, sources)
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

    async def _generate_section(
        self,
        query: str,
        title: str,
        objective: str,
        sources: list[SourceReference],
    ) -> ReportSection:
        sources_text = "\n".join(
            f"  {s.reference_id} | {s.title} | {s.snippet[:200]}" for s in sources[:20]
        )
        prompt = _SECTION_PROMPT.format(
            query=query,
            title=title,
            objective=objective,
            sources_text=sources_text or "(no sources)",
        )
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            **self._llm_kwargs,
        )
        return _parse_section(title, resp.content)

    async def _generate_summary(
        self,
        query: str,
        sections: list[ReportSection],
    ) -> str:
        sections_text = "\n\n".join(f"## {s.title}\n{s.content[:500]}" for s in sections)
        prompt = _SUMMARY_PROMPT.format(sections_text=sections_text)
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            **self._llm_kwargs,
        )
        return resp.content.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relevant_sources(
    question_ids: list[str],
    agg: AggregatedSources,
) -> list[SourceReference]:
    """Pick sources relevant to the given question IDs."""
    ref_ids: set[str] = set()
    for qid in question_ids:
        ref_ids.update(agg.grouped_by_question.get(qid, []))
    if not ref_ids:
        return agg.sources[:10]
    lookup = {s.reference_id: s for s in agg.sources}
    return [lookup[rid] for rid in ref_ids if rid in lookup]


def _parse_section(title: str, content: str) -> ReportSection:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()

    try:
        data = json.loads(text)
        citations = [
            Citation(
                reference_id=c.get("reference_id", ""),
                text_span=c.get("text_span", ""),
                context=c.get("context", ""),
            )
            for c in data.get("citations", [])
        ]
        body = _strip_leading_heading(title, data.get("content", ""))
        return ReportSection(
            title=title,
            content=body,
            citations=citations,
        )
    except json.JSONDecodeError:
        return ReportSection(title=title, content=_strip_leading_heading(title, text))


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
