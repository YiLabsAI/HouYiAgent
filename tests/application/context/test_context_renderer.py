"""Unit tests for houyi.application.context.context_renderer.ContextRenderer."""

from __future__ import annotations

import pytest

from houyi.application.context.context_renderer import ContextRenderer
from houyi.application.context.types import ContextBlock, ContextBlockType, ContextPlan


@pytest.fixture
def renderer():
    return ContextRenderer()


class TestContextRendererSystem:
    """Test system block rendering."""

    def test_system_block(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.SYSTEM, content="Be helpful.")
        msgs = renderer._render_block(block)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be helpful."

    def test_system_block_empty(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.SYSTEM, content="")
        msgs = renderer._render_block(block)
        assert len(msgs) == 0


class TestContextRendererRecent:
    """Test recent messages block rendering."""

    def test_recent_message_list(self, renderer):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        block = ContextBlock(block_type=ContextBlockType.RECENT, content=messages)
        msgs = renderer._render_block(block)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_recent_string_fallback(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.RECENT, content="Just a string")
        msgs = renderer._render_block(block)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_recent_empty(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.RECENT, content="")
        msgs = renderer._render_block(block)
        assert len(msgs) == 0


class TestContextRendererMemory:
    """Test memory block rendering."""

    def test_memory_block(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.MEMORY, content="User likes Python.")
        msgs = renderer._render_block(block)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "[Memory Context]" in msgs[0]["content"]
        assert "User likes Python." in msgs[0]["content"]

    def test_memory_block_empty(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.MEMORY, content="")
        msgs = renderer._render_block(block)
        assert len(msgs) == 0


class TestContextRendererSummary:
    """Test summary block rendering (Phase 2 prep)."""

    def test_summary_block(self, renderer):
        block = ContextBlock(
            block_type=ContextBlockType.SUMMARY, content="Earlier discussion about X."
        )
        msgs = renderer._render_block(block)
        assert len(msgs) == 1
        assert "[Conversation Summary]" in msgs[0]["content"]

    def test_summary_block_empty(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.SUMMARY, content="")
        msgs = renderer._render_block(block)
        assert len(msgs) == 0


class TestContextRendererPinned:
    """Test pinned block rendering (Phase 2 prep)."""

    def test_pinned_block(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.PINNED, content="Important quote here.")
        msgs = renderer._render_block(block)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "[Pinned]" in msgs[0]["content"]
        assert "Important quote here." in msgs[0]["content"]

    def test_pinned_block_empty(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.PINNED, content="")
        msgs = renderer._render_block(block)
        assert len(msgs) == 0


class TestContextRendererToolSummary:
    """Test tool summary block rendering (Phase 2 prep)."""

    def test_tool_summary_block(self, renderer):
        block = ContextBlock(
            block_type=ContextBlockType.TOOL_SUMMARY, content="Search returned 5 results."
        )
        msgs = renderer._render_block(block)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "[Tool Results Summary]" in msgs[0]["content"]
        assert "Search returned 5 results." in msgs[0]["content"]

    def test_tool_summary_block_empty(self, renderer):
        block = ContextBlock(block_type=ContextBlockType.TOOL_SUMMARY, content="")
        msgs = renderer._render_block(block)
        assert len(msgs) == 0


class TestContextRendererFullPlan:
    """Test rendering a complete ContextPlan."""

    def test_full_plan_ordering(self, renderer):
        plan = ContextPlan(
            blocks=[
                ContextBlock(block_type=ContextBlockType.SYSTEM, content="System prompt"),
                ContextBlock(block_type=ContextBlockType.MEMORY, content="Memory text"),
                ContextBlock(
                    block_type=ContextBlockType.RECENT,
                    content=[
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi!"},
                    ],
                ),
            ]
        )
        msgs = renderer.render(plan)
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System prompt"
        assert msgs[1]["role"] == "system"
        assert "[Memory Context]" in msgs[1]["content"]
        assert msgs[2]["role"] == "user"
        assert msgs[3]["role"] == "assistant"

    def test_empty_plan(self, renderer):
        plan = ContextPlan(blocks=[])
        msgs = renderer.render(plan)
        assert msgs == []

    def test_plan_all_block_types(self, renderer):
        """Render plan with every block type to verify complete coverage."""
        plan = ContextPlan(
            blocks=[
                ContextBlock(block_type=ContextBlockType.SYSTEM, content="Sys"),
                ContextBlock(block_type=ContextBlockType.SUMMARY, content="Summary text"),
                ContextBlock(block_type=ContextBlockType.MEMORY, content="Mem text"),
                ContextBlock(block_type=ContextBlockType.PINNED, content="Pinned text"),
                ContextBlock(block_type=ContextBlockType.TOOL_SUMMARY, content="Tool text"),
                ContextBlock(
                    block_type=ContextBlockType.RECENT,
                    content=[{"role": "user", "content": "Hi"}],
                ),
            ]
        )
        msgs = renderer.render(plan)
        assert len(msgs) == 6
        # All should be system except the last (user)
        assert all(m["role"] == "system" for m in msgs[:5])
        assert msgs[5]["role"] == "user"
