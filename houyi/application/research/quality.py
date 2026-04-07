"""QualityEvaluator — RACE + FACT dual-framework report quality assessment.

Implements the DeepResearch Bench evaluation methodology:
  - **RACE**: Comprehensiveness, Depth, Instruction-Following, Readability
  - **FACT**: Citation Accuracy, Effective Citations

All evaluations use a separate LLM "judge" call.
"""

from __future__ import annotations

import json
import logging
import re
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
        """Run both RACE and FACT evaluations and combine scores.

        Both evaluations are best-effort: if the LLM is unavailable (e.g.
        billing / rate-limit), the stage returns a zero-score placeholder
        instead of crashing the entire pipeline.
        """
        import asyncio

        race, fact = await asyncio.gather(
            self._safe_race(report, reference_answer),
            self._safe_fact(report, sources),
        )
        overall = race.overall * 0.6 + (fact.citation_accuracy * 0.4)
        return QualityScore(
            race=race,
            fact=fact,
            overall=round(overall, 2),
        )

    async def _safe_race(
        self,
        report: ResearchReport,
        reference_answer: str | None,
    ) -> RACEScore:
        try:
            return await self.evaluate_race(report, reference_answer)
        except Exception:
            logger.warning(
                "RACE evaluation failed — returning zero-score placeholder", exc_info=True
            )
            return RACEScore()

    async def _safe_fact(
        self,
        report: ResearchReport,
        sources: AggregatedSources,
    ) -> FACTScore:
        try:
            return await self.evaluate_fact(report, sources)
        except Exception:
            logger.warning(
                "FACT evaluation failed — returning zero-score placeholder", exc_info=True
            )
            return FACTScore()

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
            max_tokens=1000,
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
            max_tokens=1000,
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
    text = _strip_code_fence(content.strip())
    attempts: list[str] = []
    if text:
        attempts.append(text)

    extracted = _extract_first_json_object(text)
    if extracted and extracted not in attempts:
        attempts.append(extracted)

    if extracted:
        repaired = _remove_trailing_commas(extracted)
        if repaired and repaired not in attempts:
            attempts.append(repaired)

    recoverable = _close_json_fragment(text)
    if recoverable and recoverable not in attempts:
        attempts.append(recoverable)

    last_exc: json.JSONDecodeError | None = None
    for candidate in attempts:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            last_exc = exc

    scalar_fallback = _extract_quality_scalars(text)
    if scalar_fallback:
        return scalar_fallback

    err_type = type(last_exc).__name__ if last_exc else "Unknown"
    err_msg = last_exc.msg if last_exc else "No JSON object found"
    preview = text[:200].replace("\n", "\\n")
    logger.warning(
        "Failed to parse quality evaluation JSON error_type=%s error_msg=%s content_len=%d content_preview=%r",
        err_type,
        err_msg,
        len(text),
        preview,
    )
    return {}


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    first_nl = text.find("\n")
    if first_nl == -1:
        return text
    body = text[first_nl + 1 :].strip()
    if body.endswith("```"):
        body = body[:-3].rstrip()
    return body


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        parsed, consumed = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return text[start : start + consumed]


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _advance_json_scan_state(
    ch: str,
    *,
    in_string: bool,
    escaped: bool,
    brace_open: int,
    bracket_open: int,
) -> tuple[bool, bool, int, int]:
    if in_string:
        if escaped:
            return True, False, brace_open, bracket_open
        if ch == "\\":
            return True, True, brace_open, bracket_open
        if ch == '"':
            return False, False, brace_open, bracket_open
        return True, False, brace_open, bracket_open

    if ch == '"':
        return True, False, brace_open, bracket_open
    if ch == "{":
        return False, False, brace_open + 1, bracket_open
    if ch == "}":
        return False, False, max(0, brace_open - 1), bracket_open
    if ch == "[":
        return False, False, brace_open, bracket_open + 1
    if ch == "]":
        return False, False, brace_open, max(0, bracket_open - 1)
    return False, False, brace_open, bracket_open


def _close_json_fragment(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    fragment = text[start:]
    in_string = False
    escaped = False
    brace_open = 0
    bracket_open = 0

    for ch in fragment:
        in_string, escaped, brace_open, bracket_open = _advance_json_scan_state(
            ch,
            in_string=in_string,
            escaped=escaped,
            brace_open=brace_open,
            bracket_open=bracket_open,
        )

    repaired = fragment
    if in_string:
        repaired += '"'
    if bracket_open > 0:
        repaired += "]" * bracket_open
    if brace_open > 0:
        repaired += "}" * brace_open
    repaired = _remove_trailing_commas(repaired)
    return repaired


def _extract_quality_scalars(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}

    citation_match = re.search(r'"citation_accuracy"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if citation_match:
        data["citation_accuracy"] = float(citation_match.group(1))

    effective_match = re.search(r'"effective_citations"\s*:\s*(-?\d+)', text)
    if effective_match:
        data["effective_citations"] = int(effective_match.group(1))

    return data
