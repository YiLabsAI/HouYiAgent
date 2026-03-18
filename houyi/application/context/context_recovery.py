from __future__ import annotations

from typing import Any

from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import PlannedContextUsage


class ContextRecoveryPolicy:
    """Recovers a minimal prompt when the requested input budget is exhausted."""

    def build_latest_message_recovery(
        self,
        history_messages: list[dict[str, Any]],
        *,
        estimator: TokenEstimator,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        if not history_messages:
            return None
        llm_messages = [history_messages[-1]]
        used_tokens = estimator.count_messages(llm_messages)
        return llm_messages, {
            "model": estimator.model,
            "max_context_tokens": estimator.context_window,
            "used_tokens": used_tokens,
            "reserved_output_tokens": estimator.context_window,
            "available_tokens": 0,
            "planned_prompt_tokens": used_tokens,
            "available_input_tokens": 0,
            "block_breakdown": {"recent": used_tokens},
            "dropped_blocks": [],
            "drop_reasons": {},
        }


class RenderRecoveryPolicy:
    """Recovers a renderable prompt when planning produced an empty visible result."""

    def apply_empty_render_recovery(
        self,
        history_messages: list[dict[str, Any]],
        *,
        estimator: TokenEstimator,
        usage: PlannedContextUsage,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        if not history_messages:
            return None
        llm_messages = [history_messages[-1]]
        used_tokens = estimator.count_messages(llm_messages)
        return llm_messages, usage.model_copy(
            update={
                "used_tokens": used_tokens,
                "planned_prompt_tokens": used_tokens,
                "available_tokens": max(0, usage.available_tokens - used_tokens),
                "available_input_tokens": max(0, usage.available_input_tokens - used_tokens),
                "block_breakdown": {"recent": used_tokens},
            }
        ).model_dump(mode="json")
