"""Tests for kb-graph skill hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class MockHookContext:
    """Mock HookContext for testing."""

    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    metadata: dict[str, Any] | None = None


class TestGraphState:
    """Tests for graph state management."""

    def test_reset_graph_state(self) -> None:
        """Test reset_graph_state clears all state."""
        from houyi.rag.skills.kb_graph.hooks import (
            _graph_state,
            reset_graph_state,
        )

        _graph_state["entities_found"] = 10
        _graph_state["relations_found"] = 20
        _graph_state["hops_traversed"] = 3

        reset_graph_state()

        assert _graph_state["entities_found"] == 0
        assert _graph_state["relations_found"] == 0
        assert _graph_state["hops_traversed"] == 0

    def test_get_graph_state_returns_copy(self) -> None:
        """Test get_graph_state returns a copy."""
        from houyi.rag.skills.kb_graph.hooks import (
            _graph_state,
            get_graph_state,
            reset_graph_state,
        )

        reset_graph_state()
        _graph_state["entities_found"] = 5

        state = get_graph_state()
        state["entities_found"] = 100

        assert _graph_state["entities_found"] == 5


class TestPreGraphHook:
    """Tests for pre_graph_hook."""

    @pytest.mark.asyncio
    async def test_pre_graph_hook_grep(self) -> None:
        """Test pre_graph_hook with grep for entity search."""
        from houyi.rag.skills.kb_graph.hooks import pre_graph_hook

        context = MockHookContext(
            tool_name="grep",
            tool_args={"pattern": "entity:Person"},
        )

        result = await pre_graph_hook(context)

        assert result.success is True
        assert "kb-graph" in result.output
        assert "entity:Person" in result.output

    @pytest.mark.asyncio
    async def test_pre_graph_hook_grep_no_pattern(self) -> None:
        """Test pre_graph_hook with grep but no pattern."""
        from houyi.rag.skills.kb_graph.hooks import pre_graph_hook

        context = MockHookContext(tool_name="grep", tool_args={})

        result = await pre_graph_hook(context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_pre_graph_hook_other_tool(self) -> None:
        """Test pre_graph_hook with other tool."""
        from houyi.rag.skills.kb_graph.hooks import pre_graph_hook

        context = MockHookContext(tool_name="read", tool_args={})

        result = await pre_graph_hook(context)

        assert result.success is True


class TestPostGraphHook:
    """Tests for post_graph_hook."""

    @pytest.mark.asyncio
    async def test_post_graph_hook_with_entities(self) -> None:
        """Test post_graph_hook with entity results."""
        from houyi.rag.skills.kb_graph.hooks import (
            _graph_state,
            post_graph_hook,
            reset_graph_state,
        )

        reset_graph_state()

        context = MockHookContext(
            tool_result={
                "entities": [{"name": "Alice"}, {"name": "Bob"}],
                "relations": [{"source": "Alice", "target": "Bob", "type": "knows"}],
            },
        )

        result = await post_graph_hook(context)

        assert result.success is True
        assert _graph_state["entities_found"] == 2
        assert _graph_state["relations_found"] == 1
        assert "2 entities" in result.output
        assert "1 relations" in result.output

    @pytest.mark.asyncio
    async def test_post_graph_hook_no_entities(self) -> None:
        """Test post_graph_hook with no entities."""
        from houyi.rag.skills.kb_graph.hooks import (
            _graph_state,
            post_graph_hook,
            reset_graph_state,
        )

        reset_graph_state()

        context = MockHookContext(tool_result={"entities": [], "relations": []})

        result = await post_graph_hook(context)

        assert result.success is True
        assert _graph_state["entities_found"] == 0

    @pytest.mark.asyncio
    async def test_post_graph_hook_non_dict_result(self) -> None:
        """Test post_graph_hook with non-dict result."""
        from houyi.rag.skills.kb_graph.hooks import post_graph_hook, reset_graph_state

        reset_graph_state()

        context = MockHookContext(tool_result="string result")

        result = await post_graph_hook(context)

        assert result.success is True


class TestStopHook:
    """Tests for stop_hook."""

    @pytest.mark.asyncio
    async def test_stop_hook_with_results(self) -> None:
        """Test stop_hook with graph results."""
        from houyi.rag.skills.kb_graph.hooks import (
            _graph_state,
            reset_graph_state,
            stop_hook,
        )

        reset_graph_state()
        _graph_state["entities_found"] = 10
        _graph_state["relations_found"] = 15
        _graph_state["hops_traversed"] = 2

        context = MockHookContext()

        result = await stop_hook(context)

        assert result.success is True
        assert "Query complete" in result.output
        assert "10 entities" in result.output
        assert "15 relations" in result.output
        assert "2 hops" in result.output

    @pytest.mark.asyncio
    async def test_stop_hook_resets_state(self) -> None:
        """Test stop_hook resets state."""
        from houyi.rag.skills.kb_graph.hooks import (
            _graph_state,
            reset_graph_state,
            stop_hook,
        )

        reset_graph_state()
        _graph_state["entities_found"] = 10

        context = MockHookContext()

        await stop_hook(context)

        assert _graph_state["entities_found"] == 0
