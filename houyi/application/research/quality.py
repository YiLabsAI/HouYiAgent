"""QualityEvaluator — RACE + FACT dual-framework report quality assessment.

Implements the DeepResearch Bench evaluation methodology:
  - **RACE**: Comprehensiveness, Depth, Instruction-Following, Readability
  - **FACT**: Citation Accuracy, Effective Citations

All evaluations use a separate LLM "judge" call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import (
    AggregatedSources,
    FACTScore,
    QualityDetail,
    QualityScore,
    RACEScore,
    ResearchReport,
)

logger = logging.getLogger(__name__)

_RACE_PROMPT = """\
You are a research report quality evaluator using the RACE framework.

Evaluate the following report on four dimensions (score 0-100 each):
1. **Comprehensiveness**: Does the report cover all relevant dimensions?
2. **Depth**: Is the analysis deep with multiple reasoning layers?
3. **Instruction-Following**: Does the report address the research query?
4. **Readability**: Is the structure clear and presentation polished?

Research query: {query}

Report:
{report_text}

{reference_section}

Respond ONLY with JSON:
{{
  "comprehensiveness": {{"score": <int>, "reasoning": "..."}},
  "depth": {{"score": <int>, "reasoning": "..."}},
  "instruction_following": {{"score": <int>, "reasoning": "..."}},
  "readability": {{"score": <int>, "reasoning": "..."}}
}}
"""

_FACT_PROMPT = """\
You are a citation quality evaluator using the FACT framework.

For each citation in the report, verify:
1. Does the cited claim accurately reflect the source content?
2. Is the citation non-redundant and meaningful?

Report sections with citations:
{sections_text}

Available sources (reference_id → snippet):
{sources_text}

Respond ONLY with JSON:
{{
  "citation_accuracy": <float 0-100>,
  "effective_citations": <int>,
  "details": [
    {{"reference_id": "...", "accurate": true/false, "reasoning": "..."}}
  ]
}}
"""


class QualityEvaluator:
    """DeepResearch Bench dual-framework quality evaluator (RACE + FACT)."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._llm_kwargs = llm_kwargs

    async def evaluate(
        self,
        report: ResearchReport,
        sources: AggregatedSources,
        reference_answer: str | None = None,
    ) -> QualityScore:
        """Run both RACE and FACT evaluations and combine scores."""
        race = await self.evaluate_race(report, reference_answer)
        fact = await self.evaluate_fact(report, sources)
        overall = race.overall * 0.6 + (fact.citation_accuracy * 0.4)
        return QualityScore(
            race=race,
            fact=fact,
            overall=round(overall, 2),
        )

    async def evaluate_race(
        self,
        report: ResearchReport,
        reference_answer: str | None = None,
    ) -> RACEScore:
        """RACE framework evaluation with adaptive weighting."""
        report_text = _report_to_text(report)
        ref_section = ""
        if reference_answer:
            ref_section = f"Reference answer for comparison:\n{reference_answer}"

        prompt = _RACE_PROMPT.format(
            query=report.title,
            report_text=report_text[:8000],
            reference_section=ref_section,
        )
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            **self._llm_kwargs,
        )
        return _parse_race(resp.content)

    async def evaluate_fact(
        self,
        report: ResearchReport,
        sources: AggregatedSources,
    ) -> FACTScore:
        """FACT framework evaluation for citation trustworthiness."""
        sections_text = _sections_with_citations(report)
        sources_text = _sources_lookup_text(sources)

        prompt = _FACT_PROMPT.format(
            sections_text=sections_text[:6000],
            sources_text=sources_text[:4000],
        )
        resp = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            **self._llm_kwargs,
        )
        return _parse_fact(resp.content)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _report_to_text(report: ResearchReport) -> str:
    parts = [f"# {report.title}", "", report.summary, ""]
    for sec in report.sections:
        parts.append(f"## {sec.title}")
        parts.append(sec.content)
        parts.append("")
    return "\n".join(parts)


def _sections_with_citations(report: ResearchReport) -> str:
    parts: list[str] = []
    for sec in report.sections:
        if sec.citations:
            cites = ", ".join(f'[{c.reference_id}: "{c.text_span[:80]}"]' for c in sec.citations)
            parts.append(f"## {sec.title}\nCitations: {cites}")
    return "\n\n".join(parts) if parts else "(no citations found)"


def _sources_lookup_text(sources: AggregatedSources) -> str:
    return "\n".join(
        f"{s.reference_id}: {s.title} — {s.snippet[:200]}" for s in sources.sources[:30]
    )


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_race(content: str) -> RACEScore:
    data = _safe_json(content)
    dims = {}
    details: list[QualityDetail] = []
    for key in ("comprehensiveness", "depth", "instruction_following", "readability"):
        entry = data.get(key, {})
        score = float(entry.get("score", 0)) if isinstance(entry, dict) else 0.0
        dims[key] = max(0.0, min(100.0, score))
        details.append(
            QualityDetail(
                criterion=key,
                score=dims[key],
                reasoning=entry.get("reasoning", "") if isinstance(entry, dict) else "",
            )
        )
    overall = sum(dims.values()) / max(len(dims), 1)
    return RACEScore(
        comprehensiveness=dims.get("comprehensiveness", 0),
        depth=dims.get("depth", 0),
        instruction_following=dims.get("instruction_following", 0),
        readability=dims.get("readability", 0),
        overall=round(overall, 2),
    )


def _parse_fact(content: str) -> FACTScore:
    data = _safe_json(content)
    return FACTScore(
        citation_accuracy=max(0.0, min(100.0, float(data.get("citation_accuracy", 0)))),
        effective_citations=max(0, int(data.get("effective_citations", 0))),
    )


def _safe_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse quality evaluation JSON")
        return {}
