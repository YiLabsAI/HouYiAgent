"""MemoryJudge — score one bench case against an AnswerResult.

 every bench harness ends in a "did the system get it right?"
question. Two judge flavours cover the cases we care about:

- DeterministicJudge — pure substring + abstain-reason check.
 Used by the adversarial fixture () where every case ships
 with explicit contains / forbid / reason assertions, so
 no LLM is needed and the judge is a few hundred microseconds.
- LLMMemoryJudge — LLM-as-judge over a free-form expected
 answer. Used by LoCoMo / HaluMem where the gold answer is natural
 language and lexical overlap is unreliable. The prompt asks the
 judge LLM to emit MATCH / MISMATCH / ABSTAIN_OK and we
 parse a single token.

Both implement MemoryJudge; the bench runner doesn't care
which one it gets.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from houyi.adapters.memory.answerer import AnswerResult
from houyi.adapters.memory.bench.adversarial import AdversarialCase
from houyi.adapters.memory.bench.locomo import LoCoMoCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge call's outcome.

    correct is the only field the runner uses for aggregate
    accuracy; reason is surfaced into the per-case trace so a
    failing run can be triaged.
    """

    correct: bool
    reason: str
    """Short tag, e.g. "contains_match", "forbid_violation",
 "abstain_unexpected", "llm_match", "llm_mismatch".
 """

    detail: str = ""
    """Optional free-form explanation. Empty for deterministic verdicts;
 populated with the judge LLM's raw output for LLM verdicts.
 """


class MemoryJudge(Protocol):
    """Score one bench case + answer pair."""

    async def judge(
        self,
        case: Any,
        answer: AnswerResult,
    ) -> JudgeVerdict:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Deterministic judge — pure-Python, no LLM
# ---------------------------------------------------------------------------


class DeterministicJudge:
    """Substring + abstain-reason judge for AdversarialCase.

    Decision tree:

    1. case expects abstain:
    - answer must have abstained=True
    - if case.expected.reason != "any_abstain", the answer's
    reason must equal it (so a case marked no_candidates
    fails when the system abstains via llm_idk instead).
    2. case expects answer:
    - answer must have abstained=False
    - every entry in contains must appear in the answer
    (case-insensitive)
    - no entry in forbid may appear in the answer
    """

    async def judge(
        self,
        case: AdversarialCase,
        answer: AnswerResult,
    ) -> JudgeVerdict:
        expected = case.expected
        text = (answer.answer or "").lower()

        if expected.mode == "abstain":
            if not answer.abstained:
                return JudgeVerdict(
                    correct=False,
                    reason="answered_when_abstain_expected",
                    detail=answer.answer,
                )
            if expected.reason != "any_abstain" and answer.reason != expected.reason:
                return JudgeVerdict(
                    correct=False,
                    reason="abstain_reason_mismatch",
                    detail=f"expected={expected.reason} got={answer.reason}",
                )
            return JudgeVerdict(correct=True, reason="abstain_ok")

        # mode == "answer"
        if answer.abstained:
            return JudgeVerdict(
                correct=False,
                reason="abstained_when_answer_expected",
                detail=f"abstain reason={answer.reason}",
            )
        for needle in expected.contains:
            if needle.lower() not in text:
                return JudgeVerdict(
                    correct=False,
                    reason="contains_miss",
                    detail=f"missing={needle!r}",
                )
        for forbidden in expected.forbid:
            if forbidden.lower() in text:
                return JudgeVerdict(
                    correct=False,
                    reason="forbid_violation",
                    detail=f"contained={forbidden!r}",
                )
        return JudgeVerdict(correct=True, reason="contains_match")


# ---------------------------------------------------------------------------
# LLM-as-judge for free-form expected answers
# ---------------------------------------------------------------------------


_LLM_JUDGE_SYSTEM = (
    "You grade memory-system answers. Read the question, the gold "
    "answer, and the system's reply. Reply with EXACTLY one token:\n"
    " MATCH — system's reply is semantically equivalent to gold. When the "
    "gold answer is a disjunction of alternatives (e.g. 'A or B'), a reply "
    "matching ANY one alternative is a MATCH. Grade at the granularity the "
    "question asks for: if the question asks 'which month' and the reply "
    "names the same month as gold, it is a MATCH even when gold also "
    "carries a year. For enumeration/list questions, a reply that covers "
    "every gold item is a MATCH even if it adds a small number of extra "
    "related items.\n"
    " MISMATCH — system's reply contradicts or omits the gold answer\n"
    " ABSTAIN_OK — system abstained AND the gold answer is plausibly "
    "unknowable from the conversation\n"
    "Do not output anything else."
)


_VERDICT_RE = re.compile(r"\b(MATCH|MISMATCH|ABSTAIN_OK)\b")


class _JudgeLLM(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> Any: ...


class LLMMemoryJudge:
    """LLM-as-judge over LoCoMo-style free-form gold answers.

    The judge is intentionally simple: one chat call, one parsed
    token, no chain-of-thought. Keeping the surface small makes the
    judge cheap to re-run on every CI sweep and easy to ablate.
    """

    def __init__(
        self,
        llm: _JudgeLLM,
        *,
        timeout_seconds: float = 6.0,
        max_tokens: int = 8,
    ) -> None:
        if llm is None:
            raise ValueError("llm is required")
        self._llm = llm
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    async def judge(
        self,
        case: LoCoMoCase,
        answer: AnswerResult,
    ) -> JudgeVerdict:
        # Cheap structural shortcut: if the system abstained AND the
        # gold answer is empty/unknown, we accept without an LLM call.
        if answer.abstained and not (case.answer or "").strip():
            return JudgeVerdict(correct=True, reason="empty_gold_abstain")

        messages = [
            {"role": "system", "content": _LLM_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {case.question}\n"
                    f"Gold answer: {case.answer}\n"
                    f"System reply: {answer.answer}\n"
                    f"System abstained: {answer.abstained}\n"
                ),
            },
        ]

        try:
            response = await self._llm.chat(messages, temperature=0.0, max_tokens=self._max_tokens)
        except Exception as exc:
            logger.warning("LLM judge call failed: %s", exc)
            return JudgeVerdict(
                correct=False,
                reason="judge_llm_failed",
                detail=str(exc)[:200],
            )

        raw = (getattr(response, "content", None) or "").strip()
        m = _VERDICT_RE.search(raw.upper())
        if m is None:
            return JudgeVerdict(
                correct=False,
                reason="judge_parse_failed",
                detail=raw[:200],
            )
        token = m.group(1)
        if token == "MATCH":
            return JudgeVerdict(correct=True, reason="llm_match", detail=raw)
        if token == "ABSTAIN_OK":
            return JudgeVerdict(correct=True, reason="llm_abstain_ok", detail=raw)
        return JudgeVerdict(correct=False, reason="llm_mismatch", detail=raw)


__all__ = [
    "DeterministicJudge",
    "JudgeVerdict",
    "LLMMemoryJudge",
    "MemoryJudge",
]
