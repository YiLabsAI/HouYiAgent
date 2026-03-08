"""Converts ContextPlan into LLM-ready message list.

Takes the assembled ContextPlan (ordered blocks) and produces a flat
list of message dicts suitable for LLMAdapter.stream_chat().

"""

from __future__ import annotations

import logging
from typing import Any

from houyi.application.context.types import ContextBlock, ContextBlockType, ContextPlan

logger = logging.getLogger(__name__)


class ContextRenderer:
    """Renders a ContextPlan into a list of LLM message dicts.

    The renderer flattens ContextBlocks into the format expected by
    OpenAI-compatible chat APIs: [{"role": "...", "content": "..."}].

    Thread-safe: stateless after __init__.
    """

    def render(self, plan: ContextPlan) -> list[dict[str, Any]]:
        """Render a ContextPlan into LLM messages.

        Block rendering order follows plan.blocks order (set by ContextPlanner).
        Each block type maps to a specific rendering strategy.

        Args:
            plan: Assembled ContextPlan.

        Returns:
            List of message dicts ready for LLMAdapter.stream_chat().
        """
        messages: list[dict[str, Any]] = []

        for block in plan.blocks:
            rendered = self._render_block(block)
            messages.extend(rendered)

        if not messages:
            logger.warning("ContextRenderer produced empty message list from plan %s", plan.plan_id)

        return messages

    def _render_block(self, block: ContextBlock) -> list[dict[str, Any]]:
        """Render a single ContextBlock into message dicts.

        Args:
            block: A ContextBlock to render.

        Returns:
            List of message dicts (may be empty, one, or many).
        """
        if block.block_type == ContextBlockType.SYSTEM:
            return self._render_system(block)
        elif block.block_type == ContextBlockType.RECENT:
            return self._render_recent(block)
        elif block.block_type == ContextBlockType.SUMMARY:
            return self._render_summary(block)
        elif block.block_type == ContextBlockType.MEMORY:
            return self._render_memory(block)
        elif block.block_type == ContextBlockType.PINNED:
            return self._render_pinned(block)
        elif block.block_type == ContextBlockType.TOOL_SUMMARY:
            return self._render_tool_summary(block)
        else:
            logger.warning("Unknown block type: %s", block.block_type)
            return []

    @staticmethod
    def _render_system(block: ContextBlock) -> list[dict[str, Any]]:
        """Render system instructions block."""
        if not block.content:
            return []
        return [{"role": "system", "content": str(block.content)}]

    @staticmethod
    def _render_recent(block: ContextBlock) -> list[dict[str, Any]]:
        """Render recent messages block.

        Content is expected to be a list of message dicts.
        """
        if block.is_message_list:
            return list(block.content)  # type: ignore[arg-type]
        # Fallback: treat as a single user message
        if block.content:
            return [{"role": "user", "content": str(block.content)}]
        return []

    @staticmethod
    def _render_summary(block: ContextBlock) -> list[dict[str, Any]]:
        """Render summary block (Phase 2).

        Summaries are injected as system messages to provide compressed
        context from earlier conversation.
        """
        if not block.content:
            return []
        return [{"role": "system", "content": f"[Conversation Summary]\n{block.content}"}]

    @staticmethod
    def _render_memory(block: ContextBlock) -> list[dict[str, Any]]:
        """Render memory recall block.

        Memory context is injected as a system message before recent messages.
        """
        if not block.content:
            return []
        return [{"role": "system", "content": f"[Memory Context]\n{block.content}"}]

    @staticmethod
    def _render_pinned(block: ContextBlock) -> list[dict[str, Any]]:
        """Render pinned content block (Phase 2).

        Pinned quotes are user-protected content that survives compression.
        """
        if not block.content:
            return []
        return [{"role": "system", "content": f"[Pinned]\n{block.content}"}]

    @staticmethod
    def _render_tool_summary(block: ContextBlock) -> list[dict[str, Any]]:
        """Render tool result summary block (Phase 2).

        Compressed tool results injected as system context.
        """
        if not block.content:
            return []
        return [{"role": "system", "content": f"[Tool Results Summary]\n{block.content}"}]
