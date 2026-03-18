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
    ContextCandidate,
    ContextPlan,
    ContextSelectionPolicy,
    ContextSourceKind,
    DroppedContextBlockDetail,
    PlannedContextUsage,
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
        input_budget: int | None = None,
        candidates: list[ContextCandidate] | None = None,
        selection_policy: ContextSelectionPolicy | None = None,
        boundary_id: str | None = None,
        truncation_log_label: str | None = None,
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
        budget = max(
            0, input_budget if input_budget is not None else self.estimator.max_input_tokens
        )
        policy = selection_policy or ContextSelectionPolicy()
        assembled_candidates = candidates or self._build_default_candidates(
            messages=messages,
            system_text=sys_text,
            memory_context=memory_context,
            selection_policy=policy,
            boundary_id=boundary_id,
        )
        blocks, used, dropped_blocks, drop_reasons, dropped_block_details = (
            self._assemble_candidates(
                candidates=assembled_candidates,
                budget=budget,
                selection_policy=policy,
                boundary_id=boundary_id,
                raw_message_count=len(messages),
                truncation_log_label=truncation_log_label,
            )
        )

        # Build usage snapshot
        reserved_output_tokens = max(0, self.estimator.context_window - budget)
        usage = PlannedContextUsage(
            model=self.estimator.model,
            max_context_tokens=self.estimator.context_window,
            used_tokens=used,
            reserved_output_tokens=reserved_output_tokens,
            available_tokens=max(0, budget - used),
            planned_prompt_tokens=used,
            available_input_tokens=max(0, budget - used),
            block_breakdown=self._build_block_breakdown(blocks),
            dropped_blocks=dropped_blocks,
            drop_reasons=drop_reasons,
            dropped_block_details=dropped_block_details,
        )

        return ContextPlan(blocks=blocks, usage=usage)

    def _build_default_candidates(
        self,
        *,
        messages: list[dict[str, Any]],
        system_text: str,
        memory_context: str | None,
        selection_policy: ContextSelectionPolicy,
        boundary_id: str | None = None,
    ) -> list[ContextCandidate]:
        boundary_metadata = self._boundary_metadata(boundary_id)
        candidates: list[ContextCandidate] = []
        if system_text:
            candidates.append(
                ContextCandidate(
                    source=ContextSourceKind.SYSTEM,
                    block_type=ContextBlockType.SYSTEM,
                    content=system_text,
                    pinned=True,
                    priority=0,
                )
            )

        if (memory_context and selection_policy.allow_memory) or memory_context:
            candidates.append(
                ContextCandidate(
                    source=ContextSourceKind.MEMORY,
                    block_type=ContextBlockType.MEMORY,
                    content=memory_context,
                    priority=150,
                )
            )

        if messages:
            latest = messages[-1:]
            earlier_recent = messages[:-1]
            candidates.append(
                ContextCandidate(
                    source=ContextSourceKind.CURRENT_TURN,
                    block_type=ContextBlockType.RECENT,
                    content=latest,
                    pinned=True,
                    priority=10,
                    metadata={
                        "message_count": len(latest),
                        **boundary_metadata,
                    },
                )
            )
            if earlier_recent:
                if selection_policy.max_recent_messages is not None:
                    earlier_recent = earlier_recent[-selection_policy.max_recent_messages :]
                candidates.append(
                    ContextCandidate(
                        source=ContextSourceKind.RECENT,
                        block_type=ContextBlockType.RECENT,
                        content=earlier_recent,
                        priority=100,
                        metadata={
                            "message_count": len(earlier_recent),
                            **boundary_metadata,
                        },
                    )
                )
        return candidates

    def _assemble_candidates(
        self,
        *,
        candidates: list[ContextCandidate],
        budget: int,
        selection_policy: ContextSelectionPolicy,
        boundary_id: str | None,
        raw_message_count: int,
        truncation_log_label: str | None = None,
    ) -> tuple[list[ContextBlock], int, list[str], dict[str, str], list[DroppedContextBlockDetail]]:
        ordered = sorted(
            candidates,
            key=lambda item: (
                item.priority,
                0 if item.pinned else 1,
                item.block_type.value,
            ),
        )
        blocks: list[ContextBlock] = []
        used = 0
        dropped_blocks: list[str] = []
        drop_reasons: dict[str, str] = {}
        dropped_block_details: list[DroppedContextBlockDetail] = []
        included_message_count = 0
        current_turn_excluded = False

        for candidate in ordered:
            if self._is_candidate_boundary_excluded(candidate, boundary_id):
                self._record_dropped_candidate(
                    candidate=candidate,
                    reason="boundary_excluded",
                    dropped_blocks=dropped_blocks,
                    drop_reasons=drop_reasons,
                    dropped_block_details=dropped_block_details,
                )
                continue
            if self._is_candidate_policy_excluded(candidate, selection_policy):
                self._record_dropped_candidate(
                    candidate=candidate,
                    reason="policy_excluded",
                    dropped_blocks=dropped_blocks,
                    drop_reasons=drop_reasons,
                    dropped_block_details=dropped_block_details,
                )
                continue
            if current_turn_excluded and candidate.source == ContextSourceKind.RECENT:
                self._record_dropped_candidate(
                    candidate=candidate,
                    reason="excluded_without_current_turn",
                    dropped_blocks=dropped_blocks,
                    drop_reasons=drop_reasons,
                    dropped_block_details=dropped_block_details,
                )
                continue

            token_count = self._estimate_candidate_tokens(candidate)
            if used + token_count > budget:
                (
                    used,
                    included_message_count,
                    current_turn_excluded,
                ) = self._handle_budget_exceeded_candidate(
                    candidate=candidate,
                    budget=budget,
                    used=used,
                    included_message_count=included_message_count,
                    current_turn_excluded=current_turn_excluded,
                    blocks=blocks,
                    dropped_blocks=dropped_blocks,
                    drop_reasons=drop_reasons,
                    dropped_block_details=dropped_block_details,
                )
                continue

            self._append_candidate_block(blocks, candidate=candidate, token_count=token_count)
            used += token_count
            if self._is_recent_message_candidate(candidate):
                included_message_count += len(candidate.content)

        blocks.sort(
            key=lambda block: (
                0
                if block.block_type == ContextBlockType.SYSTEM
                else 1
                if block.block_type == ContextBlockType.PINNED
                else 2
                if block.metadata.get("source") == ContextSourceKind.RECENT.value
                else 3
                if block.block_type == ContextBlockType.RECENT
                else 4
                if block.block_type == ContextBlockType.TOOL_SUMMARY
                else 5
                if block.block_type == ContextBlockType.MEMORY
                else 6,
                block.metadata.get("recent_start_index", -1)
                if block.metadata.get("source") == ContextSourceKind.RECENT.value
                else -1,
            )
        )

        truncated = max(0, raw_message_count - included_message_count)
        if truncated > 0 and truncation_log_label:
            logger.info(
                "Context trimmed [%s]: %d/%d messages included (%d tokens used / %d budget)",
                truncation_log_label,
                included_message_count,
                raw_message_count,
                used,
                budget,
            )

        return blocks, used, dropped_blocks, drop_reasons, dropped_block_details

    @staticmethod
    def _is_candidate_policy_excluded(
        candidate: ContextCandidate,
        selection_policy: ContextSelectionPolicy,
    ) -> bool:
        return (
            (candidate.source == ContextSourceKind.MEMORY and not selection_policy.allow_memory)
            or (
                candidate.source == ContextSourceKind.SUMMARY
                and not selection_policy.allow_summaries
            )
            or (
                candidate.source == ContextSourceKind.TOOL_SUMMARY
                and not selection_policy.allow_tool_summaries
            )
            or (candidate.source == ContextSourceKind.PINNED and not selection_policy.allow_pinned)
        )

    @staticmethod
    def _is_recent_message_candidate(candidate: ContextCandidate) -> bool:
        return candidate.block_type == ContextBlockType.RECENT and isinstance(
            candidate.content, list
        )

    @staticmethod
    def _boundary_metadata(boundary_id: str | None) -> dict[str, Any]:
        if not boundary_id:
            return {}
        return {"boundary_id": boundary_id}

    @staticmethod
    def _is_candidate_boundary_excluded(
        candidate: ContextCandidate,
        boundary_id: str | None,
    ) -> bool:
        if not boundary_id:
            return False
        if candidate.source not in {ContextSourceKind.CURRENT_TURN, ContextSourceKind.RECENT}:
            return False
        candidate_boundary_id = candidate.metadata.get("boundary_id")
        return candidate_boundary_id is not None and candidate_boundary_id != boundary_id

    def _try_truncate_recent_candidate(
        self,
        candidate: ContextCandidate,
        *,
        remaining_budget: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self._is_recent_message_candidate(candidate):
            return [], 0
        content = candidate.content
        assert isinstance(content, list)
        return self._select_recent_messages(content, remaining_budget)

    def _handle_budget_exceeded_candidate(
        self,
        *,
        candidate: ContextCandidate,
        budget: int,
        used: int,
        included_message_count: int,
        current_turn_excluded: bool,
        blocks: list[ContextBlock],
        dropped_blocks: list[str],
        drop_reasons: dict[str, str],
        dropped_block_details: list[DroppedContextBlockDetail],
    ) -> tuple[int, int, bool]:
        if self._is_recent_message_candidate(candidate):
            selected_messages, selected_tokens = self._try_truncate_recent_candidate(
                candidate,
                remaining_budget=budget - used,
            )
            if selected_messages:
                self._append_candidate_block(
                    blocks,
                    candidate=candidate,
                    content=selected_messages,
                    token_count=selected_tokens,
                )
                return (
                    used + selected_tokens,
                    included_message_count + len(selected_messages),
                    (current_turn_excluded),
                )
            if candidate.source == ContextSourceKind.CURRENT_TURN:
                current_turn_excluded = True
            self._record_dropped_candidate(
                candidate=candidate,
                reason="truncated_to_fit",
                dropped_blocks=dropped_blocks,
                drop_reasons=drop_reasons,
                dropped_block_details=dropped_block_details,
            )
            return used, included_message_count, current_turn_excluded

        if candidate.source == ContextSourceKind.CURRENT_TURN:
            current_turn_excluded = True
        self._record_dropped_candidate(
            candidate=candidate,
            reason="budget_exceeded",
            dropped_blocks=dropped_blocks,
            drop_reasons=drop_reasons,
            dropped_block_details=dropped_block_details,
        )
        return used, included_message_count, current_turn_excluded

    def _record_dropped_candidate(
        self,
        *,
        candidate: ContextCandidate,
        reason: str,
        dropped_blocks: list[str],
        drop_reasons: dict[str, str],
        dropped_block_details: list[DroppedContextBlockDetail],
    ) -> None:
        dropped_blocks.append(candidate.candidate_id)
        drop_reasons[candidate.candidate_id] = reason
        dropped_block_details.append(
            DroppedContextBlockDetail(
                candidate_id=candidate.candidate_id,
                block_type=candidate.block_type.value,
                source=candidate.source.value,
                token_count=self._estimate_candidate_tokens(candidate),
                message_count=self._message_count(candidate),
                pinned=candidate.pinned,
            )
        )

    @staticmethod
    def _message_count(candidate: ContextCandidate) -> int | None:
        metadata_count = candidate.metadata.get("message_count")
        if isinstance(metadata_count, int):
            return metadata_count
        if isinstance(candidate.content, list):
            return len(candidate.content)
        return None

    @staticmethod
    def _append_candidate_block(
        blocks: list[ContextBlock],
        *,
        candidate: ContextCandidate,
        token_count: int,
        content: Any | None = None,
    ) -> None:
        blocks.append(
            ContextBlock(
                block_type=candidate.block_type,
                content=candidate.content if content is None else content,
                token_count=token_count,
                pinned=candidate.pinned,
                metadata={
                    **dict(candidate.metadata),
                    "source": candidate.source.value,
                },
            )
        )

    def _estimate_candidate_tokens(self, candidate: ContextCandidate) -> int:
        if candidate.token_count is not None:
            return candidate.token_count
        if isinstance(candidate.content, list):
            return sum(self.estimator.count_message(message) for message in candidate.content) + 3
        if candidate.block_type == ContextBlockType.SYSTEM:
            return self.estimator.count_message({"role": "system", "content": candidate.content})
        return self.estimator.count_text(str(candidate.content or "")) + 4

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

    @staticmethod
    def _build_block_breakdown(blocks: list[ContextBlock]) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for block in blocks:
            key = block.block_type.value
            breakdown[key] = breakdown.get(key, 0) + block.token_count
        return breakdown
