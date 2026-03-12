from __future__ import annotations

from typing import Any

from houyi.adapters.llm.models import DEFAULT_OUTPUT_RESERVE
from houyi.application.context.types import BudgetDecision


class TokenBudgetPolicy:
    def __init__(
        self,
        *,
        default_output_budget: int = DEFAULT_OUTPUT_RESERVE,
        default_answer_reserve: int = 512,
        default_reasoning_ratio: float = 0.5,
        default_tool_reserve: int = 0,
    ):
        self.default_output_budget = max(0, default_output_budget)
        self.default_answer_reserve = max(0, default_answer_reserve)
        self.default_reasoning_ratio = max(0.0, min(1.0, default_reasoning_ratio))
        self.default_tool_reserve = max(0, default_tool_reserve)

    def decide(
        self,
        *,
        context_window: int,
        enable_reasoning: bool = False,
        enable_tool_calls: bool = False,
        requested_max_tokens: int | None = None,
        provider_requires_max_tokens: bool = False,
        provider_default_output_budget: int | None = None,
        max_output_limit: int | None = None,
        provider_supports_reasoning_budget: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetDecision:
        effective_default_output = (
            provider_default_output_budget
            if provider_default_output_budget is not None
            else self.default_output_budget
        )
        output_budget = max(0, effective_default_output)
        should_set_max_tokens = False
        max_tokens_source: str | None = None

        if requested_max_tokens is not None:
            output_budget = max(0, requested_max_tokens)
            should_set_max_tokens = True
            max_tokens_source = "user"
        elif provider_requires_max_tokens:
            should_set_max_tokens = True
            max_tokens_source = "policy"

        if max_output_limit is not None:
            output_budget = min(output_budget, max(0, max_output_limit))

        answer_reserve = 0
        reasoning_budget = 0
        if enable_reasoning:
            answer_reserve = min(output_budget, self.default_answer_reserve)
            remainder = max(0, output_budget - answer_reserve)
            reasoning_budget = (
                remainder
                if provider_supports_reasoning_budget
                else int(remainder * self.default_reasoning_ratio)
            )

        tool_reserve = self.default_tool_reserve if enable_tool_calls else 0
        input_budget = max(0, context_window - output_budget - tool_reserve)

        return BudgetDecision(
            context_window=max(0, context_window),
            input_budget=input_budget,
            output_budget=output_budget,
            reasoning_budget=reasoning_budget,
            answer_reserve=answer_reserve,
            tool_reserve=tool_reserve,
            should_set_max_tokens=should_set_max_tokens,
            max_tokens_to_send=output_budget if should_set_max_tokens else None,
            max_tokens_source=max_tokens_source,
            finish_reason_policy="provider_raw_passthrough",
            metadata=metadata or {},
        )
