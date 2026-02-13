"""Unit tests for houyi.context.context_planner.ContextPlanner."""

from __future__ import annotations

import pytest

from houyi.context.context_planner import ContextPlanner
from houyi.context.token_estimator import TokenEstimator
from houyi.context.types import ContextBlockType
from houyi.llm.models import DEFAULT_MODEL


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
        assert len(recent) == 1
        assert len(recent[0].content) == 3

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
            # Should include fewer than all 20 messages
            assert len(recent[0].content) < 20
            # Last message should be the newest
            last_included = recent[0].content[-1]
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
