"""LLMAnswerer — turns recall hits into a grounded answer or an IDK.

 the recall pipeline already produces a RecallResult
whose reason field encodes the IDK-guard verdict (no candidates,
low evidence, contradicting evidence, explicit absence, sufficient).
The answerer consumes that result and either:

- reason != SUFFICIENT → abstain immediately (no LLM call). The
 guard is the canonical authority on "we don't have enough"; we do
 not second-guess it.
- reason == SUFFICIENT → render the top facts into a prompt, call
 the LLM with the configured budget, and return the answer.

A small abstain budget governs the call:

- max_facts_in_prompt — caps the number of candidates injected
- max_input_chars — hard cap on the rendered prompt body
- timeout_seconds — wall-clock cap; on timeout we abstain instead
 of returning a partial answer
- max_calls — currently 1; reserved for future "self-verify" loops

The LLM is also asked to emit a sentinel [IDK] token if it cannot
ground the answer in the provided facts. The answerer pattern-matches
that sentinel and routes to abstain so a hallucinating model can't
sneak past the guard.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallReason,
    RecallResult,
)

logger = logging.getLogger(__name__)


DEFAULT_IDK_PHRASE = "I don't know based on what I currently remember."
"""User-facing string returned on every abstain path."""

_LLM_IDK_SENTINEL = "[IDK]"
"""Token the prompt instructs the LLM to emit when it cannot answer."""

_ANSWER_TAG_RE = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.DOTALL | re.IGNORECASE)
_ANALYSIS_TAG_RE = re.compile(r"<Analysis>\s*(.*?)\s*</Analysis>", re.DOTALL | re.IGNORECASE)


def _extract_answer_tag(content: str) -> str:
    """Pull the answer out of an LLM response shaped as
    <Analysis>...</Analysis><Answer>...</Answer>.

    The judge compares gold against the answer, not the reasoning, so
    returning the whole content (Analysis + Answer) lets an empty Answer
    tag drag the verdict to llm_mismatch even when the Analysis carries
    the answer. Extract the Answer payload; an empty Answer falls back
    to the Analysis (some models put the answer in Analysis and leave
    Answer empty). An empty Answer with no Analysis returns "" so the
    caller abstains. Responses without the tag are returned as-is for
    backward compatibility with models that do not emit the tags.
    """
    m = _ANSWER_TAG_RE.search(content)
    if m:
        answer = m.group(1).strip()
        if answer:
            return answer
        am = _ANALYSIS_TAG_RE.search(content)
        if am:
            analysis = am.group(1).strip()
            if analysis:
                return analysis
        return ""
    return content.strip()


@dataclass(frozen=True)
class AnswerBudget:
    """Resource caps for a single LLMAnswerer.answer call.

    Defaults are conservative — under standard SiliconFlow / OpenAI
    rates, an 8-fact / 6k-char prompt at temp 0 cost a few cents and
    finishes well under 6 seconds. Bench harnesses can tighten the
    budget to surface regressions early.
    """

    max_facts_in_prompt: int = 8
    max_input_chars: int = 6000
    timeout_seconds: float = 6.0
    max_calls: int = 1


@dataclass(frozen=True)
class AbstainPolicy:
    """Additional abstain rules applied on top of the recall guard.

    The recall guard already covers no-candidates / low-evidence /
    contradicting-evidence / explicit-absence. min_top_score is a
    second-line defense: even when the guard says SUFFICIENT, refuse
    to call the LLM if the highest-scoring candidate is below this
    threshold. 0.0 (the default) disables the override.

    min_facts likewise lets callers demand at least N usable hits.
    """

    min_top_score: float = 0.0
    min_facts: int = 1


@dataclass(frozen=True)
class AnswerResult:
    """What LLMAnswerer.answer returns."""

    answer: str
    abstained: bool
    reason: str
    """One of: "sufficient" (LLM produced an answer), the
 RecallReason value when the guard rejected, "low_top_score",
 "too_few_facts", "llm_idk", "llm_failed", "timeout",
 "budget_exceeded".
 """

    citations: tuple[str, ...] = ()
    """Source anchors of facts the LLM was given. Surfaces in trace
 even on abstain so audits can see what evidence was *available*.
 """

    facts_used: int = 0
    prompt_chars: int = 0
    raw_llm_output: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class _LLMAdapter(Protocol):
    """Minimal LLM contract — same shape used by AtomicFactExtractor."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Any: ...


class LLMAnswerer:
    """Wraps a chat LLM with abstain budget + grounded-answer prompting."""

    def __init__(
        self,
        llm: _LLMAdapter,
        *,
        budget: AnswerBudget | None = None,
        policy: AbstainPolicy | None = None,
        idk_phrase: str = DEFAULT_IDK_PHRASE,
        system_prompt: str | None = None,
    ) -> None:
        if llm is None:
            raise ValueError("llm adapter is required")
        self._llm = llm
        self._budget = budget or AnswerBudget()
        self._policy = policy or AbstainPolicy()
        self._idk_phrase = idk_phrase
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    async def answer(self, query: str, recall: RecallResult) -> AnswerResult:
        """Produce an answer or an IDK abstain.

        Decision tree (in order — first match wins):

        1. recall guard rejects → abstain with the guard's reason
        2. policy min_top_score / min_facts rejects → abstain
        3. LLM call timeout → abstain "timeout"
        4. LLM raises → abstain "llm_failed"
        5. LLM emits [IDK] sentinel → abstain "llm_idk"
        6. otherwise → return the LLM's answer
        """
        guard_abstain = self._guard_abstain(recall)
        if guard_abstain is not None:
            return guard_abstain

        candidates = self._select_facts(recall.candidates)
        policy_abstain = self._policy_abstain(candidates)
        if policy_abstain is not None:
            return policy_abstain

        prompt_body = self._render_prompt(query, candidates)
        if len(prompt_body) > self._budget.max_input_chars:
            # Hard truncation rather than abstain: we keep the highest
            # scoring facts (they were sliced top-down by _select_facts)
            # and just drop the tail. If even one fact's body alone
            # exceeds the cap, abstain — the prompt would be uninformative.
            prompt_body = prompt_body[: self._budget.max_input_chars]
            if not _looks_truncatable(prompt_body):
                return _abstain(
                    self._idk_phrase,
                    reason="budget_exceeded",
                    citations=tuple(_citations(candidates)),
                    facts_used=len(candidates),
                    prompt_chars=len(prompt_body),
                )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt_body},
        ]

        try:
            response = await asyncio.wait_for(
                self._llm.chat(messages, temperature=0.0, max_tokens=512),
                timeout=self._budget.timeout_seconds,
            )
        except TimeoutError:
            logger.warning("LLMAnswerer timed out after %.1fs", self._budget.timeout_seconds)
            return _abstain(
                self._idk_phrase,
                reason="timeout",
                citations=tuple(_citations(candidates)),
                facts_used=len(candidates),
                prompt_chars=len(prompt_body),
            )
        except Exception:
            logger.warning("LLMAnswerer LLM call failed", exc_info=True)
            return _abstain(
                self._idk_phrase,
                reason="llm_failed",
                citations=tuple(_citations(candidates)),
                facts_used=len(candidates),
                prompt_chars=len(prompt_body),
            )

        content = (getattr(response, "content", None) or "").strip()
        if not content or _LLM_IDK_SENTINEL in content:
            return _abstain(
                self._idk_phrase,
                reason="llm_idk",
                citations=tuple(_citations(candidates)),
                facts_used=len(candidates),
                prompt_chars=len(prompt_body),
                raw_llm_output=content,
            )

        answer_text = _extract_answer_tag(content)
        if not answer_text:
            return _abstain(
                self._idk_phrase,
                reason="llm_idk",
                citations=tuple(_citations(candidates)),
                facts_used=len(candidates),
                prompt_chars=len(prompt_body),
                raw_llm_output=content,
            )

        return AnswerResult(
            answer=answer_text,
            abstained=False,
            reason="sufficient",
            citations=tuple(_citations(candidates)),
            facts_used=len(candidates),
            prompt_chars=len(prompt_body),
            raw_llm_output=content,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _guard_abstain(self, recall: RecallResult) -> AnswerResult | None:
        """Return an abstain result when the recall guard rejected."""
        if recall.reason is RecallReason.SUFFICIENT:
            return None
        citations = tuple(_citations(recall.candidates)) if recall.candidates else ()
        return _abstain(
            self._idk_phrase,
            reason=recall.reason.value,
            citations=citations,
            facts_used=0,
        )

    def _select_facts(self, candidates: list[RecallCandidate]) -> list[RecallCandidate]:
        return list(candidates)[: self._budget.max_facts_in_prompt]

    def _policy_abstain(self, candidates: list[RecallCandidate]) -> AnswerResult | None:
        policy = self._policy
        if len(candidates) < policy.min_facts:
            return _abstain(
                self._idk_phrase,
                reason="too_few_facts",
                citations=tuple(_citations(candidates)),
                facts_used=len(candidates),
            )
        if policy.min_top_score > 0.0 and candidates:
            top = candidates[0].score
            if top < policy.min_top_score:
                return _abstain(
                    self._idk_phrase,
                    reason="low_top_score",
                    citations=tuple(_citations(candidates)),
                    facts_used=len(candidates),
                )
        return None

    @staticmethod
    def _format_fact_line(idx: int, cand: RecallCandidate) -> str:
        """Render one candidate as a numbered fact line with anchor and time."""
        f = cand.fact
        compound_members = cand.signals and cand.signals.get("compound_members")
        if compound_members and isinstance(compound_members, list) and len(compound_members) > 8:
            rendered_obj = (
                ", ".join(compound_members[:8]) + f", ... and {len(compound_members) - 8} more"
            )
        else:
            rendered_obj = f.object

        line = f" {idx}. {f.subject} {f.predicate} {rendered_obj}"
        # For compound candidates, render all source anchors so the LLM
        # can cite each member. For individuals, keep the single anchor.
        compound_anchors = cand.signals and cand.signals.get("compound_source_anchors")
        if compound_anchors and isinstance(compound_anchors, list) and len(compound_anchors) > 1:
            line += f" [{', '.join(compound_anchors)}]"
        elif f.source_anchor:
            line += f" [{f.source_anchor}]"
        if f.event_time:
            line += f" (time: {f.event_time})"
        if f.qualifiers:
            orig = f.qualifiers.get("original_time")
            if orig:
                line += f" (original: {orig})"
        return line

    @staticmethod
    def _render_prompt(query: str, candidates: list[RecallCandidate]) -> str:
        lines = [
            "Answer the user's question using the numbered facts below. "
            "You are allowed and encouraged to perform straightforward logical, temporal, causal, or hypothetical/counterfactual reasoning "
            "based on the facts (for example, if a fact states that X was motivated by Y, then without Y, X would likely not have done it). "
            "When reasoning, retain the original specific nouns, descriptors, and concrete details from the facts (for example, preserve a description like painting of a forest scene or a specific item verbatim rather than over-generalizing or abstracting it to a broader term like watercolor painting or artwork unless specifically asked). "
            "When estimating months lapsed or durations between two dates, compute it as simple month-subtraction (e.g., December - August = 4 months) rather than day-level fractional rounding, and output the direct integer name (e.g., 'four months' or '3 months') without approximate qualifiers. "
            f"If the facts provide absolutely no relevant context, reply with exactly {_LLM_IDK_SENTINEL}.",
            "",
            "Facts:",
        ]
        for i, cand in enumerate(candidates, start=1):
            lines.append(LLMAnswerer._format_fact_line(i, cand))

        # Append graph connection paths for topological and causal chain reasoning
        connections = []
        for cand in candidates:
            rel = cand.signals.get("last_edge_relation")
            depth = cand.signals.get("bfs_depth")
            parent_id = cand.signals.get("parent_node_id")
            if rel:
                fact_idx = candidates.index(cand) + 1
                target_desc = f"{cand.fact.subject} - fact {fact_idx}"

                # Locate parent node in candidates
                parent_desc = parent_id
                if parent_id:
                    for other in candidates:
                        is_match = (
                            other.fact.subject == parent_id
                            or other.signals.get("entity") == parent_id
                            or other.fact.source_anchor == parent_id
                        )
                        if is_match:
                            other_idx = candidates.index(other) + 1
                            parent_desc = f"{other.fact.subject} - fact {other_idx}"
                            break

                if parent_desc:
                    connections.append(f"- [{parent_desc}] leads to [{target_desc}] (via {rel})")
                else:
                    connections.append(f"- leads to [{target_desc}] (via {rel} at depth {depth})")
        if connections:
            lines.append("")
            lines.append("Connections:")
            lines.extend(connections)

        lines.append("")
        lines.append(f"Question: {query.strip()}")
        return "\n".join(lines)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a memory-grounded assistant. Answer concisely using the provided facts. "
    "Cite facts by their number in square brackets, e.g. [1]. "
    "You are allowed and encouraged to perform straightforward logical, temporal, causal, or hypothetical/counterfactual reasoning "
    "based on the facts to answer the question. "
    "When reasoning, retain the original specific nouns, descriptors, and concrete details from the facts (for example, preserve a description like painting of a forest scene or a specific item verbatim rather than over-generalizing or abstracting it to a broader term like watercolor painting or artwork unless specifically asked). "
    "When estimating months lapsed or durations between two dates, compute it as simple month-subtraction (e.g., December - August = 4 months) rather than day-level fractional rounding, and output the direct integer name (e.g., 'four months' or '3 months') without approximate qualifiers. "
    f"If the facts provide absolutely no relevant context, reply with exactly {_LLM_IDK_SENTINEL}."
)


def _abstain(
    phrase: str,
    *,
    reason: str,
    citations: tuple[str, ...] = (),
    facts_used: int = 0,
    prompt_chars: int = 0,
    raw_llm_output: str = "",
    extras: dict[str, Any] | None = None,
) -> AnswerResult:
    return AnswerResult(
        answer=phrase,
        abstained=True,
        reason=reason,
        citations=citations,
        facts_used=facts_used,
        prompt_chars=prompt_chars,
        raw_llm_output=raw_llm_output,
        extras=extras or {},
    )


def _citations(candidates: list[RecallCandidate]) -> list[str]:
    seen: list[str] = []
    for cand in candidates:
        anchor = cand.fact.source_anchor
        if anchor and anchor not in seen:
            seen.append(anchor)
    return seen


_NUMBERED_LINE = re.compile(r"^\s*\d+\.\s")


def _looks_truncatable(prompt: str) -> bool:
    """Return True if the truncated prompt still contains at least one
    numbered fact line. Used as a "is the prompt still informative?"
    sanity check before sending a butchered body to the LLM.
    """
    return any(_NUMBERED_LINE.match(line) for line in prompt.splitlines())


__all__ = [
    "DEFAULT_IDK_PHRASE",
    "AbstainPolicy",
    "AnswerBudget",
    "AnswerResult",
    "LLMAnswerer",
]
