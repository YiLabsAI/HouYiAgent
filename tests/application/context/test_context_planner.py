"""Unit tests for houyi.application.context.context_planner.ContextPlanner."""

from __future__ import annotations

import pytest

from houyi.adapters.llm.models import DEFAULT_MODEL
from houyi.application.context.context_planner import ContextPlanner
from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import (
    ContextBlockType,
    ContextCandidate,
    ContextSelectionPolicy,
    ContextSourceKind,
)


@pytest.fixture
def estimator():
    return TokenEstimator(context_window_override=10000, output_reserve=2000)


@pytest.fixture
def planner(estimator):
    return ContextPlanner(
        token_estimator=estimator,
        system_instructions="You are a helpful assistant.",
    )


class TestContextPlannerBasic:
    """Test basic context plan assembly."""

    def test_empty_messages(self, planner):
        plan = planner.plan(messages=[])
        # Should have system block only
        sys_blocks = plan.get_blocks_by_type(ContextBlockType.SYSTEM)
        assert len(sys_blocks) == 1
        assert sys_blocks[0].content == "You are a helpful assistant."
        assert plan.usage.used_tokens > 0

    def test_single_message(self, planner):
        messages = [{"role": "user", "content": "Hello"}]
        plan = planner.plan(messages=messages)
        # System + Recent
        assert len(plan.blocks) == 2
        recent = plan.get_blocks_by_type(ContextBlockType.RECENT)
        assert len(recent) == 1
        assert recent[0].is_message_list
        assert len(recent[0].content) == 1

    def test_multiple_messages(self, planner):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        plan = planner.plan(messages=messages)
        recent = plan.get_blocks_by_type(ContextBlockType.RECENT)
        assert len(recent) == 2
        assert len(recent[0].content) == 2
        assert len(recent[1].content) == 1
        assert recent[1].content[0]["content"] == "How are you?"

    def test_system_instructions_override(self, planner):
        plan = planner.plan(
            messages=[{"role": "user", "content": "Hi"}],
            system_instructions="Custom system prompt",
        )
        sys_blocks = plan.get_blocks_by_type(ContextBlockType.SYSTEM)
        assert sys_blocks[0].content == "Custom system prompt"

    def test_no_system_instructions(self, estimator):
        planner = ContextPlanner(token_estimator=estimator, system_instructions="")
        plan = planner.plan(messages=[{"role": "user", "content": "Hi"}])
        sys_blocks = plan.get_blocks_by_type(ContextBlockType.SYSTEM)
        assert len(sys_blocks) == 0


class TestContextPlannerMemory:
    """Test memory context injection."""

    def test_memory_context_included(self, planner):
        plan = planner.plan(
            messages=[{"role": "user", "content": "Hi"}],
            memory_context="User prefers formal tone.",
        )
        mem_blocks = plan.get_blocks_by_type(ContextBlockType.MEMORY)
        assert len(mem_blocks) == 1
        assert "formal tone" in mem_blocks[0].content

    def test_no_memory_context(self, planner):
        plan = planner.plan(messages=[{"role": "user", "content": "Hi"}])
        mem_blocks = plan.get_blocks_by_type(ContextBlockType.MEMORY)
        assert len(mem_blocks) == 0


class TestContextPlannerTruncation:
    """Test message truncation when budget is exceeded."""

    def test_truncation_keeps_newest(self):
        # Very small budget to force truncation
        est = TokenEstimator(context_window_override=200, output_reserve=50)
        planner = ContextPlanner(token_estimator=est, system_instructions="Sys")
        messages = [{"role": "user", "content": f"Message {i} " + "x" * 50} for i in range(20)]
        plan = planner.plan(messages=messages)
        recent = plan.get_blocks_by_type(ContextBlockType.RECENT)
        if recent:
            included = [msg for block in recent for msg in block.content]
            assert len(included) < 20
            last_included = included[-1]
            assert "Message 19" in last_included["content"]


class TestContextPlannerUsage:
    """Test usage tracking."""

    def test_usage_fields(self, planner):
        plan = planner.plan(messages=[{"role": "user", "content": "Hello"}])
        usage = plan.usage
        assert usage.model == DEFAULT_MODEL
        assert usage.max_context_tokens == 10000
        assert usage.reserved_output_tokens == 2000
        assert usage.used_tokens > 0
        assert usage.available_tokens >= 0
        assert (
            usage.used_tokens + usage.available_tokens + usage.reserved_output_tokens
            == usage.max_context_tokens
        )

    def test_usage_utilization(self, planner):
        plan = planner.plan(messages=[{"role": "user", "content": "Hello"}])
        assert 0.0 < plan.usage.utilization < 1.0

    def test_block_breakdown(self, planner):
        plan = planner.plan(
            messages=[{"role": "user", "content": "Hello"}],
            memory_context="Some memory",
        )
        breakdown = plan.usage.block_breakdown
        assert "system" in breakdown
        assert "recent" in breakdown
        assert "memory" in breakdown

    def test_input_budget(self, planner):
        plan = planner.plan(
            messages=[{"role": "user", "content": "Hello"}],
            input_budget=100,
        )
        usage = plan.usage
        assert usage.reserved_output_tokens == 9900
        assert usage.available_input_tokens <= 100
        assert usage.planned_prompt_tokens == usage.used_tokens

    def test_policy_can_exclude_memory(self, planner):
        plan = planner.plan(
            messages=[{"role": "user", "content": "Hello"}],
            memory_context="Some memory",
            selection_policy=ContextSelectionPolicy(allow_memory=False),
        )
        assert plan.get_blocks_by_type(ContextBlockType.MEMORY) == []
        assert plan.usage.drop_reasons

    def test_structured_candidates_priority(self, planner):
        candidates = [
            ContextCandidate(
                source=ContextSourceKind.RECENT,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "assistant", "content": "Older reply"}],
                priority=100,
            ),
            ContextCandidate(
                source=ContextSourceKind.PINNED,
                block_type=ContextBlockType.PINNED,
                content="Pinned fact",
                pinned=True,
                priority=20,
            ),
            ContextCandidate(
                source=ContextSourceKind.CURRENT_TURN,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "user", "content": "Current ask"}],
                pinned=True,
                priority=10,
            ),
        ]
        plan = planner.plan(messages=[], candidates=candidates)
        assert [block.block_type for block in plan.blocks] == [
            ContextBlockType.PINNED,
            ContextBlockType.RECENT,
            ContextBlockType.RECENT,
        ]

    def test_drops_recent_current_turn(self):
        est = TokenEstimator(context_window_override=120, output_reserve=20)
        planner = ContextPlanner(token_estimator=est, system_instructions="Sys")
        current_turn = {"role": "user", "content": "current " + ("x" * 400)}
        older_recent = {"role": "assistant", "content": "older " + ("y" * 20)}

        plan = planner.plan(
            messages=[older_recent, current_turn],
            input_budget=40,
        )

        recent = plan.get_blocks_by_type(ContextBlockType.RECENT)
        assert recent == []
        assert plan.usage.dropped_blocks
        assert "excluded_without_current_turn" in plan.usage.drop_reasons.values()
        assert plan.usage.dropped_block_details
        assert plan.usage.dropped_block_details[0].block_type == ContextBlockType.RECENT.value
        assert plan.usage.dropped_block_details[0].source in {
            ContextSourceKind.CURRENT_TURN.value,
            ContextSourceKind.RECENT.value,
        }
        assert plan.usage.dropped_block_details[0].token_count > 0

    def test_excludes_other_boundary_recent(self, planner):
        candidates = [
            ContextCandidate(
                source=ContextSourceKind.CURRENT_TURN,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "user", "content": "Current ask"}],
                pinned=True,
                priority=10,
                metadata={
                    "message_count": 1,
                    "boundary_id": "boundary-current",
                },
            ),
            ContextCandidate(
                source=ContextSourceKind.RECENT,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "assistant", "content": "Leaked older context"}],
                priority=100,
                metadata={
                    "message_count": 1,
                    "boundary_id": "boundary-previous",
                },
            ),
        ]

        plan = planner.plan(
            messages=[],
            candidates=candidates,
            boundary_id="boundary-current",
        )

        recent = plan.get_blocks_by_type(ContextBlockType.RECENT)
        assert len(recent) == 1
        assert recent[0].content == [{"role": "user", "content": "Current ask"}]
        assert "boundary_excluded" in plan.usage.drop_reasons.values()

    def test_prefers_recent_over_summary(self, estimator):
        planner = ContextPlanner(
            token_estimator=estimator,
            system_instructions="Sys",
        )
        candidates = [
            ContextCandidate(
                source=ContextSourceKind.SUMMARY,
                block_type=ContextBlockType.SUMMARY,
                content="Earlier summary " + ("s " * 800),
                priority=200,
            ),
            ContextCandidate(
                source=ContextSourceKind.RECENT,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "assistant", "content": "Older recent fact"}],
                priority=100,
                metadata={"message_count": 1},
            ),
            ContextCandidate(
                source=ContextSourceKind.CURRENT_TURN,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "user", "content": "Current ask must survive"}],
                pinned=True,
                priority=10,
                metadata={"message_count": 1},
            ),
        ]

        plan = planner.plan(
            messages=[],
            candidates=candidates,
            input_budget=80,
        )

        recent = plan.get_blocks_by_type(ContextBlockType.RECENT)
        assert len(recent) == 2
        assert recent[0].content == [{"role": "assistant", "content": "Older recent fact"}]
        assert recent[1].content == [{"role": "user", "content": "Current ask must survive"}]
        assert plan.get_blocks_by_type(ContextBlockType.SUMMARY) == []
        assert "budget_exceeded" in plan.usage.drop_reasons.values()

    def test_keeps_memory_over_assistant(self, estimator):
        planner = ContextPlanner(
            token_estimator=estimator,
            system_instructions="Sys",
        )
        candidates = [
            ContextCandidate(
                source=ContextSourceKind.MEMORY,
                block_type=ContextBlockType.MEMORY,
                content="Stable memory rule",
                priority=150,
            ),
            ContextCandidate(
                source=ContextSourceKind.RECENT,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "assistant", "content": "Older recent fact " + ("x " * 400)}],
                priority=160,
                metadata={"message_count": 1},
            ),
            ContextCandidate(
                source=ContextSourceKind.CURRENT_TURN,
                block_type=ContextBlockType.RECENT,
                content=[{"role": "user", "content": "Current ask must survive"}],
                pinned=True,
                priority=10,
                metadata={"message_count": 1},
            ),
        ]

        plan = planner.plan(
            messages=[],
            candidates=candidates,
            input_budget=40,
        )

        recent = plan.get_blocks_by_type(ContextBlockType.RECENT)
        assert len(recent) == 1
        assert recent[0].content == [{"role": "user", "content": "Current ask must survive"}]
        memory = plan.get_blocks_by_type(ContextBlockType.MEMORY)
        assert len(memory) == 1
        assert memory[0].content == "Stable memory rule"
        assert "truncated_to_fit" in plan.usage.drop_reasons.values()
