"""IntermediateReportGenerator — per-sub-question structured report.

After each Research Agent completes, this generates a structured report
for that sub-question with proper inline citations. The Final Report
Agent then synthesizes these intermediate reports (rather than raw
search results), preserving citation fidelity across the pipeline.

Architecture reference: Onyx's "Intermediate Report → Final Report" pattern.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.context.reminders import (
    CITATION_REMINDER,
    LANGUAGE_REMINDER,
    ReminderInjector,
)
from houyi.application.research.types import SearchResult, SourceReference

logger = logging.getLogger(__name__)

_INTERMEDIATE_PROMPT = """\
You are writing a focused research report section for a specific sub-question.

Sub-question: {question}
Overall research topic: {topic}

Available sources (reference_id | title | snippet):
{sources_text}

Summary of findings: {summary}

Write a structured analysis in Markdown. Rules:
- Synthesize findings — do NOT just list sources.
- For every factual claim, insert an inline citation as [ref_id].
- Highlight key insights, contradictions, and gaps.
- Write in the SAME language as the question / topic.
- Keep the section focused and concise (300-800 words).

Respond ONLY with JSON:
{{
  "analysis": "Markdown analysis with [ref_id] citations...",
  "key_findings": ["finding 1", "finding 2", ...],
  "confidence": 0.0 to 1.0 (how confident are you in the findings),
  "gaps": ["any information gaps identified"]
}}
"""


class IntermediateReport(BaseModel):
    """Structured report for one sub-question."""

    question_id: str = ""
    question: str = ""
    analysis: str = ""
    key_findings: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: float = 0.5
    gaps: list[str] = Field(default_factory=list)


class IntermediateReportGenerator:
    """Generates per-sub-question intermediate reports with citations.

    This sits between the search phase and the final report generation,
    ensuring each sub-question's findings are structured and cited before
    the final synthesis.
    """

    def __init__(self, llm_adapter: LLMAdapter, **llm_kwargs: Any) -> None:
        self._llm = llm_adapter
        self._llm_kwargs = llm_kwargs
        self._reminders = ReminderInjector([CITATION_REMINDER, LANGUAGE_REMINDER])

    async def generate(
        self,
        search_result: SearchResult,
        question_text: str,
        topic: str,
    ) -> IntermediateReport:
        """Generate an intermediate report for one sub-question."""
        sources_text = "\n".join(
            f"  {s.reference_id} | {s.title} | {s.snippet[:200]}"
            for s in search_result.sources[:15]
        )
        prompt = _INTERMEDIATE_PROMPT.format(
            question=question_text,
            topic=topic,
            sources_text=sources_text or "(no sources found)",
            summary=search_result.summary[:500],
        )
        try:
            injected = self._reminders.inject(
                [{"role": "user", "content": prompt}],
            )
            resp = await self._llm.chat(
                messages=injected,  # type: ignore[arg-type]
                temperature=0.3,
                max_tokens=1500,
                **self._llm_kwargs,
            )
            return _parse_response(
                resp.content,
                search_result.question_id,
                question_text,
                search_result.sources,
            )
        except Exception:
            logger.warning(
                "Intermediate report generation failed for %s",
                search_result.question_id,
            )
            return IntermediateReport(
                question_id=search_result.question_id,
                question=question_text,
                analysis=search_result.summary,
                sources=search_result.sources,
                confidence=0.3,
            )

    async def generate_batch(
        self,
        search_results: list[SearchResult],
        questions: dict[str, str],
        topic: str,
    ) -> list[IntermediateReport]:
        """Generate intermediate reports for all sub-questions."""
        import asyncio

        coros = [
            self.generate(sr, questions.get(sr.question_id, ""), topic) for sr in search_results
        ]
        return list(await asyncio.gather(*coros))


def _parse_response(
    content: str,
    question_id: str,
    question_text: str,
    sources: list[SourceReference],
) -> IntermediateReport:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        data = json.loads(text)
        return IntermediateReport(
            question_id=question_id,
            question=question_text,
            analysis=data.get("analysis", ""),
            key_findings=data.get("key_findings", []),
            sources=sources,
            confidence=float(data.get("confidence", 0.5)),
            gaps=data.get("gaps", []),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return IntermediateReport(
            question_id=question_id,
            question=question_text,
            analysis=content[:2000],
            sources=sources,
            confidence=0.3,
        )
