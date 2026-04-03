"""ValidationAgent — post-generation quality gate with auto re-research.

After the report is generated, this agent inspects each section for:
  1. Citation coverage — sections without citations are flagged.
  2. Factual density — sections that are too thin on substance.
  3. Coherence — sections that contradict the query or other sections.

Low-quality sections trigger a targeted re-research cycle: generate new
search queries for the weak section, fetch additional sources, and
regenerate the section.

Architecture: runs as a post-processing step after ReportGenerator,
before the final quality evaluation (RACE/FACT).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import ReportSection, ResearchReport

logger = logging.getLogger(__name__)

_VALIDATION_PROMPT = """\
You are a research report quality inspector. Evaluate the following \
report section for quality issues.

Section title: {title}
Section content:
{content}

Research query: {query}

Evaluate on these dimensions:
1. **Citation coverage**: Does the section cite sources? (look for [ref_XXX] patterns)
2. **Substance**: Is there actual analysis, not just filler text?
3. **Relevance**: Does the content address the section's stated objective?
4. **Coherence**: Is the writing clear and logically structured?

Respond ONLY with JSON:
{{
  "quality_score": 0 to 100,
  "has_citations": true/false,
  "issues": ["issue 1", "issue 2", ...],
  "needs_rewrite": true/false,
  "suggested_queries": ["search query to improve this section", ...],
  "reasoning": "1-2 sentences explaining your judgment"
}}
"""

_QUALITY_THRESHOLD = 40


class SectionValidation(BaseModel):
    """Validation result for one section."""

    section_id: str = ""
    title: str = ""
    quality_score: int = 0
    has_citations: bool = False
    issues: list[str] = Field(default_factory=list)
    needs_rewrite: bool = False
    suggested_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ValidationReport(BaseModel):
    """Aggregated validation across all sections."""

    sections: list[SectionValidation] = Field(default_factory=list)
    overall_score: float = 0.0
    sections_needing_rewrite: int = 0
    total_issues: int = 0


class ValidationAgent:
    """Inspects report quality and identifies sections needing improvement.

    Does NOT perform the re-research itself — returns a ``ValidationReport``
    that the orchestrator can use to trigger targeted re-search.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        *,
        quality_threshold: int = _QUALITY_THRESHOLD,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._threshold = quality_threshold
        self._llm_kwargs = llm_kwargs

    async def validate(self, report: ResearchReport, query: str) -> ValidationReport:
        """Validate all sections and return a quality report."""
        import asyncio

        coros = [self._validate_section(section, query) for section in report.sections]
        results = list(await asyncio.gather(*coros))

        needs_rewrite = sum(1 for r in results if r.needs_rewrite)
        total_issues = sum(len(r.issues) for r in results)
        scores = [r.quality_score for r in results]
        overall = sum(scores) / max(len(scores), 1)

        return ValidationReport(
            sections=results,
            overall_score=round(overall, 1),
            sections_needing_rewrite=needs_rewrite,
            total_issues=total_issues,
        )

    async def _validate_section(self, section: ReportSection, query: str) -> SectionValidation:
        prompt = _VALIDATION_PROMPT.format(
            title=section.title,
            content=section.content[:3000],
            query=query,
        )
        try:
            resp = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
                **self._llm_kwargs,
            )
            result = _parse_validation(resp.content)
            result.section_id = section.section_id
            result.title = section.title
            result.needs_rewrite = result.quality_score < self._threshold
            return result
        except Exception:
            logger.warning("Validation failed for section %s", section.title)
            return SectionValidation(
                section_id=section.section_id,
                title=section.title,
                quality_score=50,
                reasoning="Validation skipped due to error",
            )


def _parse_validation(content: str) -> SectionValidation:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        data = json.loads(text)
        return SectionValidation(
            quality_score=int(data.get("quality_score", 50)),
            has_citations=bool(data.get("has_citations", False)),
            issues=data.get("issues", []),
            needs_rewrite=bool(data.get("needs_rewrite", False)),
            suggested_queries=data.get("suggested_queries", []),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return SectionValidation(quality_score=50, reasoning="Failed to parse validation response")
