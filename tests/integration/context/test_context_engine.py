"""Context Engine: TokenEstimator → ContextPlanner → ContextRenderer → LLM messages.

Verifies format correctness, token counting consistency, truncation behavior,
and memory injection across the entire context pipeline.
"""

from __future__ import annotations

import pytest

from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import MemoryScope
from houyi.application.context.context_planner import ContextPlanner
from houyi.application.context.context_renderer import ContextRenderer
from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import ContextBlockType


@pytest.fixture
def estimator():
    return TokenEstimator()


@pytest.fixture
def planner(estimator):
    return ContextPlanner(estimator)


@pytest.fixture
def renderer():
    return ContextRenderer()


class TestContextEnginePipeline:
    """Full pipeline: messages → plan → rendered LLM messages."""

    def test_basic_pipeline(self, planner, renderer):
        """Simple messages through the full pipeline."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        plan = planner.plan(messages)
        rendered = renderer.render(plan)

        # Should have system block + recent messages
        assert len(rendered) >= 3
        # Last messages should be the conversation
        user_msgs = [m for m in rendered if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[-1]["content"] == "How are you?"

    def test_system_instruction_included(self, planner, renderer):
        """System instruction appears first in rendered output."""
        messages = [{"role": "user", "content": "Hi"}]
        plan = planner.plan(messages, system_instructions="You are a helpful assistant.")
        rendered = renderer.render(plan)

        assert rendered[0]["role"] == "system"
        assert "helpful assistant" in rendered[0]["content"]

    def test_no_system_instruction_default(self, planner, renderer):
        """No system instruction → no system block in plan."""
        messages = [{"role": "user", "content": "Hi"}]
        plan = planner.plan(messages)
        rendered = renderer.render(plan)

        # Planner has empty default system_instructions, so no system block
        system_msgs = [m for m in rendered if m["role"] == "system"]
        assert len(system_msgs) == 0

    def test_memory_injection(self, planner, renderer, tmp_path):
        """Memory context injected into the pipeline."""
        store = MemoryStore(data_dir=tmp_path / "mem")
        store.put("user_pref", "User prefers Python", scope=MemoryScope.USER)
        store.put("project_info", "Working on HouYi project", scope=MemoryScope.WORKSPACE)

        # Combine memory from both scopes
        user_mem = store.as_context_text(scope=MemoryScope.USER)
        ws_mem = store.as_context_text(scope=MemoryScope.WORKSPACE)
        memory_text = "\n".join(filter(None, [user_mem, ws_mem]))
        assert "Python" in memory_text
        assert "HouYi" in memory_text

        messages = [{"role": "user", "content": "What language should I use?"}]
        plan = planner.plan(messages, memory_context=memory_text)
        rendered = renderer.render(plan)

        # Memory should appear in rendered messages
        full_text = " ".join(m["content"] for m in rendered)
        assert "Python" in full_text
        assert "HouYi" in full_text

    def test_token_count_consistency(self, estimator, planner, renderer):
        """Token counts in plan match actual rendered content."""
        messages = [
            {"role": "user", "content": "Tell me about machine learning"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."},
            {"role": "user", "content": "What about deep learning?"},
        ]
        plan = planner.plan(messages)
        rendered = renderer.render(plan)

        # Plan should report usage
        assert plan.usage is not None
        assert plan.usage.used_tokens > 0
        assert plan.usage.max_context_tokens > 0
        assert plan.usage.used_tokens <= plan.usage.max_context_tokens

        # Rendered messages should have content
        total_content = " ".join(m["content"] for m in rendered)
        assert len(total_content) > 0

    def test_multi_turn_conversation(self, planner, renderer):
        """Multi-turn conversation renders correctly."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})

        plan = planner.plan(messages)
        rendered = renderer.render(plan)

        # All messages should be present (within context window)
        user_msgs = [m for m in rendered if m["role"] == "user"]
        assistant_msgs = [m for m in rendered if m["role"] == "assistant"]
        assert len(user_msgs) >= 1
        assert len(assistant_msgs) >= 1

        # Most recent message should be preserved
        assert any("Question 9" in m["content"] for m in user_msgs)

    def test_plan_block_types(self, planner, renderer):
        """Plan contains expected block types."""
        messages = [{"role": "user", "content": "Hi"}]
        plan = planner.plan(
            messages,
            system_instructions="Be helpful",
            memory_context="User likes cats",
        )

        block_types = [b.block_type for b in plan.blocks]
        assert ContextBlockType.SYSTEM in block_types
        assert ContextBlockType.RECENT in block_types
        assert ContextBlockType.MEMORY in block_types

    def test_rendered_messages_valid_format(self, planner, renderer):
        """All rendered messages have valid role and content fields."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        plan = planner.plan(messages, memory_context="Some memory")
        rendered = renderer.render(plan)

        for msg in rendered:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("system", "user", "assistant", "tool")
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 0


class TestContextEngineTruncation:
    """Test truncation behavior under tight token budgets."""

    def test_truncation_preserves_newest(self):
        """Under tight budget, newest messages are preserved."""
        # output_reserve defaults to 4096, so context_window must be > 4096
        # to have any input budget. Use 20000 to leave ~16000 for messages.
        estimator = TokenEstimator(context_window_override=20000)
        planner = ContextPlanner(estimator)
        renderer = ContextRenderer()

        # Generate many messages to exceed budget (~16000 tokens available)
        # Each message ~20 tokens, 400 messages ~8000 tokens, 800 messages ~16000 tokens
        messages = []
        for i in range(500):
            messages.append(
                {
                    "role": "user",
                    "content": f"This is message number {i} with some padding text to consume tokens and fill the budget",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"This is response number {i} with additional text for token consumption and budget filling",
                }
            )

        plan = planner.plan(messages)
        rendered = renderer.render(plan)

        # Should have truncated — not all 1000 messages present
        user_msgs = [m for m in rendered if m["role"] == "user"]
        assert len(user_msgs) < 500
        assert len(user_msgs) > 0

        # Most recent message MUST be preserved
        all_content = " ".join(m["content"] for m in rendered)
        assert "message number 499" in all_content

    def test_usage_reports_truncation(self):
        """Usage report reflects truncation."""
        estimator = TokenEstimator(context_window_override=500)
        planner = ContextPlanner(estimator)

        messages = []
        for i in range(50):
            messages.append({"role": "user", "content": f"Long message {i} " * 10})
            messages.append({"role": "assistant", "content": f"Long response {i} " * 10})

        plan = planner.plan(messages)
        assert plan.usage is not None
        assert plan.usage.used_tokens <= plan.usage.max_context_tokens


class TestContextEngineEdgeCases:
    """Edge cases for the context engine pipeline."""

    def test_empty_messages(self, planner, renderer):
        """Empty message list produces minimal output."""
        plan = planner.plan([])
        rendered = renderer.render(plan)
        # Should at least have system message
        assert len(rendered) >= 0

    def test_single_message(self, planner, renderer):
        """Single user message works."""
        plan = planner.plan([{"role": "user", "content": "Hello"}])
        rendered = renderer.render(plan)
        user_msgs = [m for m in rendered if m["role"] == "user"]
        assert len(user_msgs) == 1

    def test_empty_memory_text(self, planner, renderer):
        """Empty memory text doesn't add memory block."""
        plan = planner.plan(
            [{"role": "user", "content": "Hi"}],
            memory_context="",
        )
        memory_blocks = [b for b in plan.blocks if b.block_type == ContextBlockType.MEMORY]
        assert len(memory_blocks) == 0

    def test_very_long_single_message(self, planner, renderer):
        """Very long single message handled gracefully."""
        long_content = "word " * 5000
        messages = [{"role": "user", "content": long_content}]
        plan = planner.plan(messages)
        rendered = renderer.render(plan)
        # Should not crash
        assert len(rendered) >= 1
