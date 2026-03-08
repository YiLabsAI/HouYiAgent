"""Tests for kb-analyze skill hooks."""

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


class TestAnalyzeState:
    """Tests for analyze state management."""

    def test_reset_analyze_state(self) -> None:
        """Test reset_analyze_state clears all state."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            reset_analyze_state,
        )

        _analyze_state["files_scanned"] = 100
        _analyze_state["indexes_checked"] = 5
        _analyze_state["issues_found"] = ["issue1"]

        reset_analyze_state()

        assert _analyze_state["files_scanned"] == 0
        assert _analyze_state["indexes_checked"] == 0
        assert _analyze_state["issues_found"] == []

    def test_get_analyze_state_returns_copy(self) -> None:
        """Test get_analyze_state returns a copy."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            get_analyze_state,
            reset_analyze_state,
        )

        reset_analyze_state()
        _analyze_state["files_scanned"] = 50

        state = get_analyze_state()
        state["files_scanned"] = 200

        assert _analyze_state["files_scanned"] == 50


class TestPreAnalyzeHook:
    """Tests for pre_analyze_hook."""

    @pytest.mark.asyncio
    async def test_pre_analyze_hook_glob(self) -> None:
        """Test pre_analyze_hook with glob."""
        from houyi.rag.skills.kb_analyze.hooks import pre_analyze_hook

        context = MockHookContext(tool_name="glob", tool_args={"pattern": "**/*.md"})

        result = await pre_analyze_hook(context)

        assert result.success is True
        assert "kb-analyze" in result.output
        assert "Scanning" in result.output

    @pytest.mark.asyncio
    async def test_pre_analyze_hook_other_tool(self) -> None:
        """Test pre_analyze_hook with other tool."""
        from houyi.rag.skills.kb_analyze.hooks import pre_analyze_hook

        context = MockHookContext(tool_name="read", tool_args={})

        result = await pre_analyze_hook(context)

        assert result.success is True


class TestPostAnalyzeHook:
    """Tests for post_analyze_hook."""

    @pytest.mark.asyncio
    async def test_post_analyze_hook_glob_list_result(self) -> None:
        """Test post_analyze_hook with glob list result."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            post_analyze_hook,
            reset_analyze_state,
        )

        reset_analyze_state()

        context = MockHookContext(
            tool_name="glob",
            tool_result=["file1.md", "file2.md", "file3.md"],
        )

        result = await post_analyze_hook(context)

        assert result.success is True
        assert _analyze_state["files_scanned"] == 3
        assert "Scanned 3 files" in result.output

    @pytest.mark.asyncio
    async def test_post_analyze_hook_glob_dict_result(self) -> None:
        """Test post_analyze_hook with glob dict result."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            post_analyze_hook,
            reset_analyze_state,
        )

        reset_analyze_state()

        context = MockHookContext(
            tool_name="glob",
            tool_result={"files": ["a.md", "b.md"]},
        )

        result = await post_analyze_hook(context)

        assert result.success is True
        assert _analyze_state["files_scanned"] == 2

    @pytest.mark.asyncio
    async def test_post_analyze_hook_no_result(self) -> None:
        """Test post_analyze_hook with no result."""
        from houyi.rag.skills.kb_analyze.hooks import post_analyze_hook, reset_analyze_state

        reset_analyze_state()

        context = MockHookContext(tool_name="glob", tool_result=None)

        result = await post_analyze_hook(context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_post_analyze_hook_other_tool(self) -> None:
        """Test post_analyze_hook with other tool."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            post_analyze_hook,
            reset_analyze_state,
        )

        reset_analyze_state()

        context = MockHookContext(tool_name="read", tool_result="content")

        await post_analyze_hook(context)

        assert _analyze_state["files_scanned"] == 0


class TestStopHook:
    """Tests for stop_hook."""

    @pytest.mark.asyncio
    async def test_stop_hook_with_results(self) -> None:
        """Test stop_hook with analysis results."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            reset_analyze_state,
            stop_hook,
        )

        reset_analyze_state()
        _analyze_state["files_scanned"] = 100
        _analyze_state["indexes_checked"] = 3

        context = MockHookContext()

        result = await stop_hook(context)

        assert result.success is True
        assert "Analysis complete" in result.output
        assert "100 files" in result.output
        assert "3 indexes" in result.output

    @pytest.mark.asyncio
    async def test_stop_hook_with_issues(self) -> None:
        """Test stop_hook with issues found."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            reset_analyze_state,
            stop_hook,
        )

        reset_analyze_state()
        _analyze_state["files_scanned"] = 50
        _analyze_state["issues_found"] = ["issue1", "issue2", "issue3"]

        context = MockHookContext()

        result = await stop_hook(context)

        assert result.success is True
        assert "Found 3 issues" in result.output

    @pytest.mark.asyncio
    async def test_stop_hook_resets_state(self) -> None:
        """Test stop_hook resets state."""
        from houyi.rag.skills.kb_analyze.hooks import (
            _analyze_state,
            reset_analyze_state,
            stop_hook,
        )

        reset_analyze_state()
        _analyze_state["files_scanned"] = 100

        context = MockHookContext()

        await stop_hook(context)

        assert _analyze_state["files_scanned"] == 0
