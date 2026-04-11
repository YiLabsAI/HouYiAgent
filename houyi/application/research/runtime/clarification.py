"""ClarificationAgent — pre-research intent clarification.

Analyzes the user query for ambiguity, scope issues, or implicit
assumptions, and generates clarifying questions. This is an optional
step that can be skipped by the user.

Architecture: 2-layer orchestration — runs as a pre-processing agent
before the Planner, ensuring the decomposed sub-questions are more
targeted and reducing downstream search noise.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import ClarificationResult

logger = logging.getLogger(__name__)

_CLARIFICATION_PROMPT = """\
You are a research planning assistant. Before decomposing a research query, \
analyze it for potential ambiguity, scope issues, or implicit assumptions.

User query: {query}

Evaluate:
1. Is the query clear and specific enough for web research?
2. Are there ambiguous terms that could lead to irrelevant results?
3. Is the scope too broad or too narrow?
4. Are there implicit time/geographic/domain constraints?

Respond ONLY with JSON:
{{
  "needs_clarification": true/false,
  "confidence": 0.0 to 1.0 (how confident you are the query is clear),
  "issues": ["issue 1", "issue 2", ...],
  "suggested_questions": ["clarifying question 1", ...],
  "refined_query": "optional: if you can improve the query without user input"
}}
"""


class ClarificationAgent:
    """Analyzes query clarity and suggests improvements.

    Use as an optional pre-processing step before ``ResearchPlanner``.
    When ``needs_clarification`` is True, the UI should present the
    suggested questions to the user.
    """

    def __init__(self, llm_adapter: LLMAdapter, **llm_kwargs: Any) -> None:
        self._llm = llm_adapter
        self._llm_kwargs = llm_kwargs

    async def analyze(self, query: str) -> ClarificationResult:
        """Analyze query for ambiguity and suggest clarifications."""
        if not query.strip():
            return ClarificationResult(confidence=0.5)
        prompt = _CLARIFICATION_PROMPT.format(query=query)
        try:
            resp = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
                **self._llm_kwargs,
            )
            return _parse_clarification(resp.content)
        except Exception:
            logger.warning("ClarificationAgent failed, proceeding without clarification")
            return ClarificationResult(confidence=0.5)


def _parse_clarification(content: str) -> ClarificationResult:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        data = json.loads(text)
        return ClarificationResult(
            needs_clarification=bool(data.get("needs_clarification", False)),
            confidence=float(data.get("confidence", 0.5)),
            issues=data.get("issues", []),
            suggested_questions=data.get("suggested_questions", []),
            refined_query=data.get("refined_query"),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return ClarificationResult(confidence=0.5)
