from __future__ import annotations

from typing import Any


class LLMRequestOptionsResolver:
    def __init__(self, *, apply_budget_policy: Any) -> None:
        self._apply_budget_policy = apply_budget_policy

    def resolve_llm_kwargs(
        self,
        *,
        model: str,
        request: Any,
        conversation: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        llm_kwargs: dict[str, Any] = {}
        if request.temperature is not None:
            llm_kwargs["temperature"] = request.temperature
        elif conversation.temperature is not None:
            llm_kwargs["temperature"] = conversation.temperature
        if request.max_tokens is not None:
            llm_kwargs["max_tokens"] = request.max_tokens
        elif conversation.max_tokens is not None:
            llm_kwargs["max_tokens"] = conversation.max_tokens
        if conversation.top_p is not None:
            llm_kwargs["top_p"] = conversation.top_p
        if request.enable_reasoning:
            llm_kwargs["enable_reasoning"] = True
        budget_metadata = self._apply_budget_policy(
            model=model,
            request=request,
            conversation=conversation,
            llm_kwargs=llm_kwargs,
        )
        reasoning_budget = None
        if isinstance(budget_metadata, dict):
            candidate = budget_metadata.get("reasoning_budget")
            if isinstance(candidate, int) and candidate > 0:
                reasoning_budget = candidate
        if request.enable_reasoning and reasoning_budget is not None:
            llm_kwargs["thinking_budget"] = reasoning_budget
        return llm_kwargs, budget_metadata
