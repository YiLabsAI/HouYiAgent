"""Memory benchmark judges.

Two implementations:

- StubMemoryJudge — deterministic, network-free, used by smoke
 tests and offline harness debugging. Decides correctness by lower-cased
 substring containment / exact equality.
- LLMMemoryJudge — production judge that delegates the binary
 equivalence call to a small chat model (defaults to Qwen2.5-7B-Instruct
 via the project's LLMAdapter; HaluMem-grade evaluation should
 override the model via MEMORY_BENCH_JUDGE_MODEL env or the
 model kwarg).

The judge surface intentionally returns a JudgeVerdict with
the explicit reason — both for telemetry and so smoke tests can assert
the reasoning path without re-running the LLM.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge surface
# ---------------------------------------------------------------------------


VerdictKind = Literal["correct", "wrong", "missing"]


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One judgment produced by the judge.

    - kind == "correct": prediction matches gold (semantically).
    - kind == "wrong": prediction is non-empty but does not match
    gold (a fabrication).
    - kind == "missing": prediction is empty / refused (an omission).
    """

    kind: VerdictKind
    reason: str = ""


class MemoryJudge(ABC):
    """Abstract correctness oracle for the three benchmark tasks.

    Each method receives one gold-vs-predicted pair and returns a
    JudgeVerdict. Implementations decide how strict the
    semantic match is; the runner only consumes the verdict.
    """

    @abstractmethod
    def judge_extraction(self, gold: str, predicted: str | None) -> JudgeVerdict:
        """Decide whether predicted faithfully captures gold."""

    @abstractmethod
    def judge_update(
        self,
        old_value: str,
        new_value_gold: str,
        new_value_predicted: str | None,
    ) -> JudgeVerdict:
        """Decide whether predicted correctly supersedes old_value."""

    @abstractmethod
    def judge_qa(self, question: str, gold_answer: str, predicted: str | None) -> JudgeVerdict:
        """Decide whether predicted matches gold_answer for question."""


# ---------------------------------------------------------------------------
# Stub judge
# ---------------------------------------------------------------------------


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _token_overlap(gold: str, predicted: str) -> bool:
    """Loose containment: any non-trivial gold token must appear in pred.

    Used to keep the stub judge useful when extraction emits a tighter
    paraphrase rather than the exact gold string.
    """

    gold_tokens = {tok for tok in gold.split() if len(tok) >= 3}
    if not gold_tokens:
        return gold == predicted
    return any(tok in predicted for tok in gold_tokens)


class StubMemoryJudge(MemoryJudge):
    """Deterministic judge for tests and offline plumbing checks.

    The matching rule is intentionally generous so smoke tests stay
    green when the SUT emits paraphrases of the gold text. It is
    NOT a substitute for the real LLM judge in scoring runs — the
    distribution of false positives is undefined.
    """

    def judge_extraction(self, gold: str, predicted: str | None) -> JudgeVerdict:
        gold_n = _normalize(gold)
        pred_n = _normalize(predicted)
        if not pred_n:
            return JudgeVerdict("missing", "predicted is empty")
        if gold_n in pred_n or pred_n in gold_n or _token_overlap(gold_n, pred_n):
            return JudgeVerdict("correct", "substring or token overlap")
        return JudgeVerdict("wrong", "no overlap with gold")

    def judge_update(
        self,
        old_value: str,
        new_value_gold: str,
        new_value_predicted: str | None,
    ) -> JudgeVerdict:
        gold_n = _normalize(new_value_gold)
        pred_n = _normalize(new_value_predicted)
        old_n = _normalize(old_value)
        if not pred_n:
            return JudgeVerdict("missing", "no successor recorded")
        if pred_n == old_n:
            # The system kept the old value: also an omission, not a fab.
            return JudgeVerdict("missing", "old value still active")
        if gold_n in pred_n or pred_n in gold_n or _token_overlap(gold_n, pred_n):
            return JudgeVerdict("correct", "successor matches gold")
        return JudgeVerdict("wrong", "successor differs from gold")

    def judge_qa(self, question: str, gold_answer: str, predicted: str | None) -> JudgeVerdict:
        del question  # the stub ignores the question; semantic judges use it
        return self.judge_extraction(gold_answer, predicted)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM_PROMPT = """\
You are a strict memory-evaluation judge. Decide whether the candidate \
text correctly conveys the gold reference. Respond with EXACTLY one \
word on the first line:

- "CORRECT" — the candidate semantically matches the gold (paraphrases ok).
- "WRONG" — the candidate is non-empty but conflicts with the gold.
- "MISSING" — the candidate is empty, refused, or off-topic.

You may add a short reason on the second line. No extra prose.
"""


_DEFAULT_JUDGE_MODEL_ENV = "MEMORY_BENCH_JUDGE_MODEL"
_DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LLMMemoryJudge(MemoryJudge):
    """Judge backed by a chat model.

    The default model is intentionally a small (~7B) instruct model:
    binary equivalence judgments do not benefit much from a 32B+ judge
    and the cost / latency difference per call is large. For grading
    runs that should be reproducible with the HaluMem paper's protocol,
    pass a stronger model= (e.g. "gpt-4o-2024-08-06").
    """

    def __init__(
        self,
        llm_adapter: Any,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        max_retries: int = 2,
    ) -> None:
        if llm_adapter is None:
            raise ValueError("llm_adapter is required")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        resolved_model = model or os.getenv(_DEFAULT_JUDGE_MODEL_ENV) or _DEFAULT_JUDGE_MODEL
        self._llm = llm_adapter
        self._model = resolved_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    @property
    def model(self) -> str:
        return self._model

    def judge_extraction(self, gold: str, predicted: str | None) -> JudgeVerdict:
        if not (predicted or "").strip():
            return JudgeVerdict("missing", "predicted is empty")
        prompt = (
            f"GOLD: {gold}\nCANDIDATE: {predicted}\n"
            "Is CANDIDATE a faithful paraphrase or restatement of GOLD?"
        )
        return self._call(prompt)

    def judge_update(
        self,
        old_value: str,
        new_value_gold: str,
        new_value_predicted: str | None,
    ) -> JudgeVerdict:
        if not (new_value_predicted or "").strip():
            return JudgeVerdict("missing", "no successor recorded")
        prompt = (
            f"OLD VALUE: {old_value}\n"
            f"GOLD NEW VALUE: {new_value_gold}\n"
            f"CANDIDATE NEW VALUE: {new_value_predicted}\n"
            "Did the system replace OLD with the GOLD NEW value? "
            "MISSING if the candidate equals OLD; "
            "WRONG if it differs from GOLD; "
            "CORRECT if it matches GOLD."
        )
        return self._call(prompt)

    def judge_qa(self, question: str, gold_answer: str, predicted: str | None) -> JudgeVerdict:
        if not (predicted or "").strip():
            return JudgeVerdict("missing", "predicted is empty")
        prompt = (
            f"QUESTION: {question}\n"
            f"GOLD ANSWER: {gold_answer}\n"
            f"CANDIDATE ANSWER: {predicted}\n"
            "Does CANDIDATE convey the same answer as GOLD?"
        )
        return self._call(prompt)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> JudgeVerdict:
        """Issue a judge call with bounded retries and parse the verdict.

        The judge entrypoint is sync because MemoryBenchRunner composes
        multiple judgments per session; making them all async would
        just shuffle blocking onto the runner without removing it.
        Instead the runner already wraps the bench loop in
        asyncio.to_thread when called from async contexts.

        Two failure classes are retried:
        - Transport / adapter exceptions (network blip, rate limit).
        - Unparseable verdicts (model emitted prose instead of one of
          CORRECT / WRONG / MISSING). Both are treated as transient.

        After exhausting retries we degrade to MISSING so a single
        flaky judge call cannot abort the whole bench.
        """
        attempts = self._max_retries + 1
        last_failure_reason = "judge call failed"
        for attempt in range(1, attempts + 1):
            try:
                response = self._invoke(prompt)
            except Exception as exc:
                last_failure_reason = f"judge call failed: {type(exc).__name__}"
                logger.warning(
                    "LLMMemoryJudge call failed (attempt %d/%d)",
                    attempt,
                    attempts,
                    exc_info=True,
                )
                if attempt >= attempts:
                    break
                continue

            content = getattr(response, "content", None)
            verdict = _parse_verdict(content)
            if not _is_parse_failure(verdict):
                return verdict
            last_failure_reason = verdict.reason
            if attempt >= attempts:
                break
            logger.debug(
                "LLMMemoryJudge unparseable verdict (attempt %d/%d): %s",
                attempt,
                attempts,
                last_failure_reason,
            )

        return JudgeVerdict("missing", last_failure_reason)

    def _invoke(self, prompt: str) -> Any:
        import asyncio

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            return asyncio.run(
                self._llm.chat(
                    messages,
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            )
        except RuntimeError:
            # Already inside a running loop: fall back to nest-friendly path.
            return _run_in_existing_loop(
                self._llm.chat(
                    messages,
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            )


def _run_in_existing_loop(coro: Any) -> Any:
    """Best-effort fallback when asyncio.run is not available.

    The judge is invoked from sync contexts inside the bench runner;
    however when the runner is itself wrapped in asyncio.to_thread
    from a parent loop, asyncio.run raises. We open a fresh loop
    on the worker thread to keep the call self-contained.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


_PARSE_FAILURE_REASONS: tuple[str, ...] = (
    "judge returned empty content",
    "unparseable judge output",
)


def _is_parse_failure(verdict: JudgeVerdict) -> bool:
    """Whether a verdict reflects a malformed judge response worth retrying.

    A legitimate MISSING verdict from the model (e.g. "MISSING: candidate
    contradicts gold") is preserved as-is. Empty payloads and prose that
    does not start with one of the canonical verdicts are treated as
    transient parse failures.
    """
    if verdict.kind != "missing":
        return False
    reason = verdict.reason or ""
    return any(reason.startswith(prefix) for prefix in _PARSE_FAILURE_REASONS)


def _parse_verdict(content: str | None) -> JudgeVerdict:
    """Parse the judge's free-form response into a JudgeVerdict.

    Accepts the leading word in case-insensitive form; anything else is
    treated as a refusal and downgraded to MISSING so the runner
    still makes progress.
    """
    text = (content or "").strip()
    if not text:
        return JudgeVerdict("missing", "judge returned empty content")

    first_line, _, rest = text.partition("\n")
    head = first_line.strip().rstrip(".:!").upper()
    reason = rest.strip()
    if head.startswith("CORRECT"):
        return JudgeVerdict("correct", reason)
    if head.startswith("WRONG"):
        return JudgeVerdict("wrong", reason)
    if head.startswith("MISSING"):
        return JudgeVerdict("missing", reason)
    return JudgeVerdict("missing", f"unparseable judge output: {first_line!r}")
