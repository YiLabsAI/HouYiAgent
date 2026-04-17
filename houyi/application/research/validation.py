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
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.taxonomy import IDENTITY_CONTEXT_HINTS
from houyi.application.research.types import (
    OutlineSection,
    ReportSection,
    ResearchPlan,
    ResearchReport,
)

logger = logging.getLogger(__name__)

# Imported from taxonomy module (single source of truth).
_IDENTITY_CONTEXT_HINTS = IDENTITY_CONTEXT_HINTS

_VALIDATION_PROMPT = """\
You are a research report quality inspector. Evaluate the following \
report section for quality issues.

Research query: {query}
Section title: {title}
Section objective: {objective}
Section position in report: {section_position}
Previous section: {previous_section}
Next section: {next_section}
Must-cover obligations:
{coverage_facets}
Required caveats:
{required_caveats}
Section content:
{content}

Evaluate on these dimensions:
1. **Citation coverage**: Does the section cite sources? (look for [ref_XXX] patterns)
2. **Substance**: Is there actual analysis, not just filler text?
3. **Relevance and completeness**: Does the content address the section objective and cover the must-cover obligations?
4. **Coherence and continuity**: Is the writing clear, logically structured, and well-aligned with surrounding sections?
5. **Caveats and tensions**: Does the section acknowledge important caveats, uncertainty, or unresolved gaps when needed?
6. **Narrative control**: Does the section present claims first, then support them with evidence, instead of narrating retrieval noise or source-hunting process?
7. **Entity anchoring**: When the topic could be ambiguous, does the section stay anchored to the intended entity and avoid drifting into same-name or weakly related material?

If the section is weak, suggested_queries must be targeted web search queries that fill the missing evidence or coverage gap. Keep suggested_queries concise and specific.

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


@dataclass(frozen=True, slots=True)
class ValidationSectionContext:
    objective: str
    section_position: str
    previous_section: str
    next_section: str
    coverage_facets: list[str]
    required_caveats: list[str]
    # Planner-assigned metadata; True when any related sub-question was marked
    # disambiguation_needed by the planner.  Supersedes keyword-hint detection.
    disambiguation_needed: bool = False


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

    async def validate(
        self,
        report: ResearchReport,
        query: str,
        *,
        plan: ResearchPlan | None = None,
        section_titles: set[str] | None = None,
        content_char_limit: int | None = None,
    ) -> ValidationReport:
        """Validate all sections and return a quality report."""
        import asyncio

        sections = [
            section
            for section in report.sections
            if not section_titles or section.title in section_titles
        ]
        context_by_title = _build_validation_contexts(report, plan)
        coros = [
            self._validate_section(
                section,
                query,
                context=context_by_title.get(section.title),
                content_char_limit=content_char_limit,
            )
            for section in sections
        ]
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

    async def _validate_section(
        self,
        section: ReportSection,
        query: str,
        *,
        context: ValidationSectionContext | None = None,
        content_char_limit: int | None = None,
    ) -> SectionValidation:
        section_context = context or _default_validation_context()
        prompt = _VALIDATION_PROMPT.format(
            query=query,
            title=section.title,
            objective=section_context.objective,
            section_position=section_context.section_position,
            previous_section=section_context.previous_section,
            next_section=section_context.next_section,
            coverage_facets=_format_validation_lines(section_context.coverage_facets),
            required_caveats=_format_validation_lines(section_context.required_caveats),
            content=section.content[: content_char_limit or 3000],
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
            if result.needs_rewrite:
                result.suggested_queries = _normalize_suggested_queries(
                    result.suggested_queries,
                    query=query,
                    section=section,
                    context=section_context,
                )
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


def _build_validation_contexts(
    report: ResearchReport,
    plan: ResearchPlan | None,
) -> dict[str, ValidationSectionContext]:
    contexts: dict[str, ValidationSectionContext] = {}
    outline_by_title = {
        section.title: section for section in (plan.outline if plan is not None else [])
    }
    total_sections = len(report.sections)
    for index, section in enumerate(report.sections):
        outline_section = outline_by_title.get(section.title)
        disambiguation_needed = _section_disambiguation_needed(
            outline_section,
            plan,
        )
        contexts[section.title] = ValidationSectionContext(
            objective=getattr(outline_section, "objective", ""),
            section_position=f"{index + 1} of {total_sections}" if total_sections else "(unknown)",
            previous_section=report.sections[index - 1].title if index > 0 else "(none)",
            next_section=report.sections[index + 1].title
            if index + 1 < total_sections
            else "(none)",
            coverage_facets=_outline_coverage_facets(outline_section),
            required_caveats=_outline_required_caveats(outline_section),
            disambiguation_needed=disambiguation_needed,
        )
    return contexts


def _default_validation_context() -> ValidationSectionContext:
    return ValidationSectionContext(
        objective="",
        section_position="(unknown)",
        previous_section="(none)",
        next_section="(none)",
        coverage_facets=[],
        required_caveats=[],
    )


def _outline_coverage_facets(outline_section: OutlineSection | None) -> list[str]:
    if outline_section is None:
        return []
    return [facet.name for facet in outline_section.coverage_contract.must_cover_facets]


def _outline_required_caveats(outline_section: OutlineSection | None) -> list[str]:
    if outline_section is None:
        return []
    return list(outline_section.coverage_contract.required_caveats)


def _format_validation_lines(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _normalize_suggested_queries(
    queries: list[str],
    *,
    query: str,
    section: ReportSection,
    context: ValidationSectionContext,
) -> list[str]:
    cleaned = [item.strip() for item in queries if item and item.strip()]
    if cleaned:
        return cleaned[:2]
    if _needs_identity_repair(context, section):
        focus = section.title.strip() or query.strip()
        return [
            f"{query.strip()} {focus} official profile".strip(),
            f"{focus} employer biography disambiguation".strip(),
        ]
    fallback_parts = [section.title.strip()]
    if context.objective.strip():
        fallback_parts.append(context.objective.strip())
    if context.coverage_facets:
        fallback_parts.append(" ".join(context.coverage_facets[:2]))
    fallback = " ".join(part for part in fallback_parts if part)
    if not fallback:
        fallback = section.title.strip() or query.strip()
    secondary = section.title.strip() or query.strip()
    return [
        f"{query.strip()} {fallback}".strip(),
        f"{secondary} evidence analysis".strip(),
    ]


def _needs_identity_repair(
    context: ValidationSectionContext,
    section: ReportSection,
) -> bool:
    # Planner metadata takes precedence over keyword heuristic.
    if context.disambiguation_needed:
        return True
    text = " ".join(
        context.coverage_facets + context.required_caveats + [context.objective, section.title]
    ).lower()
    return any(hint in text for hint in _IDENTITY_CONTEXT_HINTS)


def _section_disambiguation_needed(
    outline_section: OutlineSection | None,
    plan: ResearchPlan | None,
) -> bool:
    """Check if any sub-question related to this section has disambiguation_needed set."""
    if outline_section is None or plan is None:
        return False
    related_ids = set(outline_section.related_question_ids)
    if not related_ids:
        return False
    return any(
        sq.disambiguation_needed for sq in plan.sub_questions if sq.question_id in related_ids
    )
