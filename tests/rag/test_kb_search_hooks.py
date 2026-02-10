"""Tests for kb-search skill hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest


@dataclass
class MockHookContext:
    """Mock HookContext for testing."""

    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    metadata: dict[str, Any] | None = None


class TestSearchState:
    """Tests for search state management."""

    def test_reset_search_state(self) -> None:
        """Test reset_search_state clears all state."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            reset_search_state,
        )

        # Modify state
        _search_state["query"] = "test query"
        _search_state["files_searched"] = 10
        _search_state["matches_found"] = 5

        # Reset
        reset_search_state()

        # Verify reset
        assert _search_state["query"] == ""
        assert _search_state["files_searched"] == 0
        assert _search_state["matches_found"] == 0
        assert _search_state["sources_collected"] == []

    def test_get_search_state_returns_copy(self) -> None:
        """Test get_search_state returns a copy, not the original."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            get_search_state,
            reset_search_state,
        )

        reset_search_state()
        _search_state["query"] = "original"

        state = get_search_state()
        state["query"] = "modified"

        # Original should be unchanged
        assert _search_state["query"] == "original"


class TestPreSearchHook:
    """Tests for pre_search_hook."""

    @pytest.mark.asyncio
    async def test_pre_search_hook_grep(self) -> None:
        """Test pre_search_hook with Grep tool."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            pre_search_hook,
            reset_search_state,
        )

        reset_search_state()

        context = MockHookContext(
            tool_name="grep",
            tool_args={"pattern": "test pattern"},
            metadata={"knowledge_dir": "/test/knowledge"},
        )

        result = await pre_search_hook(context)

        assert result.success is True
        assert "kb-search" in result.output
        assert "test pattern" in result.output
        assert result.inject_to_prompt is True
        assert _search_state["query"] == "test pattern"

    @pytest.mark.asyncio
    async def test_pre_search_hook_read(self) -> None:
        """Test pre_search_hook with Read tool."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            pre_search_hook,
            reset_search_state,
        )

        reset_search_state()

        context = MockHookContext(
            tool_name="read",
            tool_args={"file_path": "/test/knowledge/doc.md"},
            metadata={"knowledge_dir": "/test/knowledge"},
        )

        result = await pre_search_hook(context)

        assert result.success is True
        assert _search_state["files_searched"] == 1

    @pytest.mark.asyncio
    async def test_pre_search_hook_glob(self) -> None:
        """Test pre_search_hook with Glob tool."""
        from houyi.rag.skills.kb_search.hooks import pre_search_hook, reset_search_state

        reset_search_state()

        context = MockHookContext(
            tool_name="glob",
            tool_args={"pattern": "**/*.md"},
        )

        result = await pre_search_hook(context)

        assert result.success is True
        assert "Exploring" in result.output
        assert "**/*.md" in result.output

    @pytest.mark.asyncio
    async def test_pre_search_hook_uses_env_knowledge_dir(self) -> None:
        """Test pre_search_hook uses KNOWLEDGE_DIR env var."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            pre_search_hook,
            reset_search_state,
        )

        reset_search_state()

        with patch.dict("os.environ", {"KNOWLEDGE_DIR": "/env/knowledge"}):
            context = MockHookContext(
                tool_name="grep",
                tool_args={"pattern": "test"},
            )

            await pre_search_hook(context)

            assert _search_state["knowledge_dir"] == "/env/knowledge"

    @pytest.mark.asyncio
    async def test_pre_search_hook_default_knowledge_dir(self) -> None:
        """Test pre_search_hook uses default knowledge dir."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            pre_search_hook,
            reset_search_state,
        )

        reset_search_state()

        with patch.dict("os.environ", {}, clear=True):
            context = MockHookContext(
                tool_name="grep",
                tool_args={"pattern": "test"},
                metadata={},
            )

            await pre_search_hook(context)

            assert _search_state["knowledge_dir"] == "knowledge/"

    @pytest.mark.asyncio
    async def test_pre_search_hook_no_args(self) -> None:
        """Test pre_search_hook with no tool args."""
        from houyi.rag.skills.kb_search.hooks import pre_search_hook, reset_search_state

        reset_search_state()

        context = MockHookContext(tool_name="grep", tool_args={})

        result = await pre_search_hook(context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_pre_search_hook_unknown_tool(self) -> None:
        """Test pre_search_hook with unknown tool."""
        from houyi.rag.skills.kb_search.hooks import pre_search_hook, reset_search_state

        reset_search_state()

        context = MockHookContext(tool_name="unknown", tool_args={"arg": "value"})

        result = await pre_search_hook(context)

        assert result.success is True


class TestPostSearchHook:
    """Tests for post_search_hook."""

    @pytest.mark.asyncio
    async def test_post_search_hook_grep_with_matches(self) -> None:
        """Test post_search_hook with grep matches."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            post_search_hook,
            reset_search_state,
        )

        reset_search_state()

        context = MockHookContext(
            tool_name="grep",
            tool_result={
                "matches": [
                    {"file": "/test/doc1.md", "line": 10, "content": "match 1"},
                    {"file": "/test/doc2.md", "line": 20, "content": "match 2"},
                ]
            },
        )

        result = await post_search_hook(context)

        assert result.success is True
        assert _search_state["matches_found"] == 2
        assert len(_search_state["sources_collected"]) == 2
        assert "Progress" in result.output

    @pytest.mark.asyncio
    async def test_post_search_hook_grep_no_matches(self) -> None:
        """Test post_search_hook with grep but no matches."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            post_search_hook,
            reset_search_state,
        )

        reset_search_state()

        context = MockHookContext(
            tool_name="grep",
            tool_result={"matches": []},
        )

        result = await post_search_hook(context)

        assert result.success is True
        assert _search_state["matches_found"] == 0

    @pytest.mark.asyncio
    async def test_post_search_hook_read_with_content(self) -> None:
        """Test post_search_hook with read result."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            post_search_hook,
            reset_search_state,
        )

        reset_search_state()

        context = MockHookContext(
            tool_name="read",
            tool_result="File content here...",
        )

        result = await post_search_hook(context)

        assert result.success is True
        assert _search_state["files_searched"] == 1

    @pytest.mark.asyncio
    async def test_post_search_hook_limits_sources(self) -> None:
        """Test post_search_hook limits sources to 5."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            post_search_hook,
            reset_search_state,
        )

        reset_search_state()

        matches = [
            {"file": f"/test/doc{i}.md", "line": i, "content": f"match {i}"}
            for i in range(10)
        ]

        context = MockHookContext(
            tool_name="grep",
            tool_result={"matches": matches},
        )

        await post_search_hook(context)

        # Should collect only first 5 sources
        assert len(_search_state["sources_collected"]) == 5
        # But count all matches
        assert _search_state["matches_found"] == 10

    @pytest.mark.asyncio
    async def test_post_search_hook_no_result(self) -> None:
        """Test post_search_hook with no result."""
        from houyi.rag.skills.kb_search.hooks import post_search_hook, reset_search_state

        reset_search_state()

        context = MockHookContext(tool_name="grep", tool_result=None)

        result = await post_search_hook(context)

        assert result.success is True


class TestStopHook:
    """Tests for stop_hook."""

    @pytest.mark.asyncio
    async def test_stop_hook_with_results(self) -> None:
        """Test stop_hook with search results."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            reset_search_state,
            stop_hook,
        )

        reset_search_state()
        _search_state["files_searched"] = 5
        _search_state["matches_found"] = 10
        _search_state["sources_collected"] = [
            {"file_path": "/test/doc1.md", "location": "line 1", "snippet": "..."},
            {"file_path": "/test/doc2.md", "location": "line 2", "snippet": "..."},
        ]

        context = MockHookContext()

        result = await stop_hook(context)

        assert result.success is True
        assert "Search complete" in result.output
        assert "5 files" in result.output
        assert "10 matches" in result.output
        assert result.metadata["total_files"] == 5
        assert result.metadata["total_matches"] == 10
        assert result.metadata["sources_count"] == 2

    @pytest.mark.asyncio
    async def test_stop_hook_no_results(self) -> None:
        """Test stop_hook with no results shows warning."""
        from houyi.rag.skills.kb_search.hooks import reset_search_state, stop_hook

        reset_search_state()

        context = MockHookContext()

        result = await stop_hook(context)

        assert result.success is True
        assert "WARNING" in result.output
        assert "No search results" in result.output

    @pytest.mark.asyncio
    async def test_stop_hook_resets_state(self) -> None:
        """Test stop_hook resets state after completion."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            reset_search_state,
            stop_hook,
        )

        reset_search_state()
        _search_state["files_searched"] = 5
        _search_state["matches_found"] = 10

        context = MockHookContext()

        await stop_hook(context)

        # State should be reset
        assert _search_state["files_searched"] == 0
        assert _search_state["matches_found"] == 0

    @pytest.mark.asyncio
    async def test_stop_hook_many_sources_truncates(self) -> None:
        """Test stop_hook truncates source list display."""
        from houyi.rag.skills.kb_search.hooks import (
            _search_state,
            reset_search_state,
            stop_hook,
        )

        reset_search_state()
        _search_state["files_searched"] = 10
        _search_state["matches_found"] = 20
        _search_state["sources_collected"] = [
            {"file_path": f"/test/doc{i}.md", "location": f"line {i}", "snippet": "..."}
            for i in range(10)
        ]

        context = MockHookContext()

        result = await stop_hook(context)

        assert "... and 7 more" in result.output
