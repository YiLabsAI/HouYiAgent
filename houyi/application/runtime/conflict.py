"""Conflict detection and resolution for multi-agent orchestration.

When parallel agents investigate overlapping questions, their conclusions
may conflict. This module provides:

1. **Detection** — pairwise comparison of agent outputs using text similarity.
2. **Source voting** — heuristic resolution based on evidence density.
3. **LLM arbitration** — an impartial LLM judge evaluates credibility.
4. **Dual presentation** — when both perspectives are valid, preserves both
   with a complementary summary.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ARBITER_PROMPT = """\
You are an impartial research arbitrator. Two research agents investigated \
the same topic and reached different conclusions.

Agent A ({agent_a_id}):
{conclusion_a}

Agent B ({agent_b_id}):
{conclusion_b}

Evaluate which conclusion is more credible, well-sourced, and accurate.
Consider: source diversity, specificity of evidence, logical consistency.

Respond ONLY with JSON:
{{
  "winner": "agent_a" or "agent_b" or "both" (if both are valid perspectives),
  "reasoning": "1-3 sentences explaining your judgment",
  "confidence": 0.0 to 1.0,
  "dual_perspective": "optional: when both are valid, summarize how they complement each other"
}}
"""


class ConflictResolution(BaseModel):
    """Outcome of resolving a single conflict."""

    method: str = ""
    winner: str | None = None
    reasoning: str = ""
    confidence: float = 0.0
    dual_perspective: str | None = None


class ConflictRecord(BaseModel):
    """A detected disagreement between two agent results."""

    question_id: str = ""
    agent_a_id: str = ""
    agent_a_conclusion: str = ""
    agent_b_id: str = ""
    agent_b_conclusion: str = ""
    conflict_type: str = "factual"
    resolution: ConflictResolution | None = None


class AgentTaskResult(BaseModel):
    """Lightweight result wrapper for conflict detection and error policy."""

    agent_id: str = ""
    task: str = ""
    output: Any = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictResolver:
    """Detects and resolves conflicting conclusions from parallel agents.

    Three-tier resolution strategy:

    1. **Source voting** — compare source overlap and volume.
    2. **LLM arbitration** — an LLM judge evaluates credibility (when
       ``llm_adapter`` is provided).
    3. **Dual presentation** — when the arbiter declares "both" valid,
       both perspectives are preserved with a complementary summary.
    """

    def __init__(self, *, llm_adapter: Any = None, **llm_kwargs: Any) -> None:
        self._llm = llm_adapter
        self._llm_kwargs = llm_kwargs

    async def detect(self, results: list[AgentTaskResult]) -> list[ConflictRecord]:
        """Compare pairwise results and return detected conflicts.

        Only flags genuine disagreements where both agents succeeded and
        produced substantive, differing outputs.
        """
        conflicts: list[ConflictRecord] = []
        for i, a in enumerate(results):
            for b in results[i + 1 :]:
                if not a.success or not b.success:
                    continue
                a_text = str(a.output).strip().lower() if a.output else ""
                b_text = str(b.output).strip().lower() if b.output else ""
                if not a_text or not b_text or a_text == b_text:
                    continue
                if _text_similarity(a_text, b_text) > 0.85:
                    continue
                conflicts.append(
                    ConflictRecord(
                        agent_a_id=a.agent_id,
                        agent_a_conclusion=str(a.output)[:2000],
                        agent_b_id=b.agent_id,
                        agent_b_conclusion=str(b.output)[:2000],
                    )
                )
        return conflicts

    async def resolve(self, conflict: ConflictRecord) -> ConflictResolution:
        """Resolve a conflict via source voting + optional LLM arbitration."""
        if self._llm is not None:
            return await self._resolve_via_llm(conflict)
        return self._resolve_via_voting(conflict)

    def _resolve_via_voting(self, conflict: ConflictRecord) -> ConflictResolution:
        """Heuristic resolution: longer, more detailed answer wins."""
        a_score = _source_vote_score(conflict.agent_a_conclusion)
        b_score = _source_vote_score(conflict.agent_b_conclusion)
        if a_score > b_score:
            winner, confidence = conflict.agent_a_id, 0.6
        elif b_score > a_score:
            winner, confidence = conflict.agent_b_id, 0.6
        else:
            winner, confidence = conflict.agent_a_id, 0.5
        return ConflictResolution(
            method="source_voting",
            winner=winner,
            reasoning=f"Source voting: A={a_score:.2f}, B={b_score:.2f}",
            confidence=confidence,
        )

    async def _resolve_via_llm(self, conflict: ConflictRecord) -> ConflictResolution:
        """LLM arbitration with dual-perspective support."""
        prompt = _ARBITER_PROMPT.format(
            agent_a_id=conflict.agent_a_id,
            conclusion_a=conflict.agent_a_conclusion[:1500],
            agent_b_id=conflict.agent_b_id,
            conclusion_b=conflict.agent_b_conclusion[:1500],
        )
        try:
            resp = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                **self._llm_kwargs,
            )
            return _parse_arbiter_response(resp.content, conflict)
        except Exception:
            logger.warning("LLM arbitration failed, falling back to voting")
            return self._resolve_via_voting(conflict)


def _source_vote_score(conclusion: str) -> float:
    """Score based on evidence density: source count, URL count, length."""
    urls = conclusion.lower().count("http")
    length = len(conclusion)
    return urls * 0.3 + min(length / 2000, 1.0) * 0.7


def _text_similarity(a: str, b: str) -> float:
    """Quick Jaccard similarity on word bigrams."""
    a_bigrams = {a[i : i + 2] for i in range(len(a) - 1)} if len(a) > 1 else set()
    b_bigrams = {b[i : i + 2] for i in range(len(b) - 1)} if len(b) > 1 else set()
    if not a_bigrams or not b_bigrams:
        return 0.0
    return len(a_bigrams & b_bigrams) / len(a_bigrams | b_bigrams)


def _parse_arbiter_response(content: str, conflict: ConflictRecord) -> ConflictResolution:
    """Parse LLM arbiter JSON response into a ConflictResolution."""
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ConflictResolution(
            method="llm_arbitration",
            reasoning=content[:300],
            confidence=0.4,
        )

    winner_raw = data.get("winner", "")
    if winner_raw == "agent_a":
        winner = conflict.agent_a_id
    elif winner_raw == "agent_b":
        winner = conflict.agent_b_id
    else:
        winner = None

    return ConflictResolution(
        method="llm_arbitration",
        winner=winner,
        reasoning=data.get("reasoning", ""),
        confidence=float(data.get("confidence", 0.5)),
        dual_perspective=data.get("dual_perspective"),
    )
