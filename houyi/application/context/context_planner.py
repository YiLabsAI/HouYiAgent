"""Assembles ContextPlan

Basic assembly (System + Recent Window, no compression).
Phase 2: Adds summary blocks, pinned quotes, tool summaries, compression triggers.

"""

from __future__ import annotations

import logging
from typing import Any

from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import (
    ContextBlock,
    ContextBlockType,
    ContextPlan,
    ContextUsage,
)

logger = logging.getLogger(__name__)


class ContextPlanner:
    """Assembles a ContextPlan from conversation messages and system instructions.

    The planner decides which messages fit within the token budget and
    produces an ordered list of ContextBlocks. Phase 1 uses a simple
    "system + all recent messages" strategy with no compression.

    Thread-safe: instances hold only configuration, no mutable state.
    """

    def __init__(
        self,
        token_estimator: TokenEstimator,
        system_instructions: str = "",
    ):
        """Initialize context planner.

        Args:
            token_estimator: TokenEstimator instance for token counting.
            system_instructions: Default system prompt. Can be overridden per-plan.
        """
        self.estimator = token_estimator
        self.system_instructions = system_instructions

    def plan(
        self,
        messages: list[dict[str, Any]],
        system_instructions: str | None = None,
        memory_context: str | None = None,
    ) -> ContextPlan:
        """Build a ContextPlan from conversation messages.

        Phase 1 strategy:
        1. System instructions block (always included, highest priority)
        2. Memory context block (if provided)
        3. Recent messages (newest first, fill remaining budget)

        Args:
            messages: Conversation history as list of message dicts
                      (role, content). Ordered oldest → newest.
            system_instructions: Override system prompt for this plan.
            memory_context: Optional memory recall text to inject.

        Returns:
            ContextPlan with blocks fitting within token budget.
        """
        sys_text = (
            system_instructions if system_instructions is not None else self.system_instructions
        )
        budget = self.estimator.max_input_tokens
        blocks: list[ContextBlock] = []
        used = 0

        # 1) System instructions (always first, always included)
        if sys_text:
            sys_tokens = self.estimator.count_message({"role": "system", "content": sys_text})
            blocks.append(
                ContextBlock(
                    block_type=ContextBlockType.SYSTEM,
                    content=sys_text,
                    token_count=sys_tokens,
                    pinned=True,
                )
            )
            used += sys_tokens

        # 2) Memory context (Phase 1: simple text injection)
        if memory_context:
            mem_tokens = self.estimator.count_text(memory_context) + 4  # message overhead
            if used + mem_tokens <= budget:
                blocks.append(
                    ContextBlock(
                        block_type=ContextBlockType.MEMORY,
                        content=memory_context,
                        token_count=mem_tokens,
                    )
                )
                used += mem_tokens

        # 3) Recent messages (fill remaining budget, newest first priority)
        remaining = budget - used
        included_messages, msg_tokens = self._select_recent_messages(messages, remaining)

        if included_messages:
            blocks.append(
                ContextBlock(
                    block_type=ContextBlockType.RECENT,
                    content=included_messages,
                    token_count=msg_tokens,
                )
            )
            used += msg_tokens

        truncated = len(messages) - len(included_messages)
        if truncated > 0:
            logger.info(
                "Context truncated: %d/%d messages included (%d tokens used / %d budget)",
                len(included_messages),
                len(messages),
                used,
                budget,
            )

        # Build usage snapshot
        usage = ContextUsage(
            model=self.estimator.model,
            max_context_tokens=self.estimator.context_window,
            used_tokens=used,
            reserved_output_tokens=self.estimator.output_reserve,
            available_tokens=max(0, budget - used),
            block_breakdown={b.block_type.value: b.token_count for b in blocks},
        )

        return ContextPlan(blocks=blocks, usage=usage)

    def _select_recent_messages(
        self,
        messages: list[dict[str, Any]],
        token_budget: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Select as many recent messages as fit within the token budget.

        Scans from newest to oldest, accumulating token counts.
        Returns messages in original (oldest → newest) order.

        Args:
            messages: All conversation messages (oldest → newest).
            token_budget: Available tokens for messages.

        Returns:
            Tuple of (selected_messages, total_token_count).
        """
        if not messages:
            return [], 0

        base_overhead = 3  # reply priming overhead
        remaining = token_budget - base_overhead
        if remaining <= 0:
            return [], 0

        # Scan from newest to oldest
        selected_reversed: list[tuple[dict[str, Any], int]] = []
        total = 0

        for msg in reversed(messages):
            msg_tokens = self.estimator.count_message(msg)
            if total + msg_tokens > remaining:
                break
            selected_reversed.append((msg, msg_tokens))
            total += msg_tokens

        # Restore original order (oldest → newest)
        selected_reversed.reverse()
        selected = [m for m, _ in selected_reversed]
        return selected, total + base_overhead
