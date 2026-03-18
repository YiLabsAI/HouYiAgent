from __future__ import annotations

from houyi.application.context.types import ContextSelectionPolicy


def build_default_context_selection_policy() -> ContextSelectionPolicy:
    return ContextSelectionPolicy(
        policy_name="chat_default",
        allow_memory=True,
        allow_summaries=True,
        allow_tool_summaries=True,
        allow_pinned=True,
    )
