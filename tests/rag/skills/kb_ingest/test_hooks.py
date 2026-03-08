"""Tests for kb-ingest skill hooks."""

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


class TestIngestState:
    """Tests for ingest state management."""

    def test_reset_ingest_state(self) -> None:
        """Test reset_ingest_state clears all state."""
        from houyi.rag.skills.kb_ingest.hooks import (
            _ingest_state,
            reset_ingest_state,
        )

        _ingest_state["files_processed"] = 10
        _ingest_state["chunks_created"] = 50
        _ingest_state["errors"] = ["error1"]

        reset_ingest_state()

        assert _ingest_state["files_processed"] == 0
        assert _ingest_state["chunks_created"] == 0
        assert _ingest_state["errors"] == []

    def test_get_ingest_state_returns_copy(self) -> None:
        """Test get_ingest_state returns a copy."""
        from houyi.rag.skills.kb_ingest.hooks import (
            _ingest_state,
            get_ingest_state,
            reset_ingest_state,
        )

        reset_ingest_state()
        _ingest_state["files_processed"] = 5

        state = get_ingest_state()
        state["files_processed"] = 100

        assert _ingest_state["files_processed"] == 5


class TestPreIngestHook:
    """Tests for pre_ingest_hook."""

    @pytest.mark.asyncio
    async def test_pre_ingest_hook_index_write(self) -> None:
        """Test pre_ingest_hook with index file write."""
        from houyi.rag.skills.kb_ingest.hooks import pre_ingest_hook

        context = MockHookContext(
            tool_name="write",
            tool_args={"file_path": "/knowledge/.rag_index/vectors.bin"},
        )

        result = await pre_ingest_hook(context)

        assert result.success is True
        assert "kb-ingest" in result.output
        assert "Writing index" in result.output

    @pytest.mark.asyncio
    async def test_pre_ingest_hook_no_file_path(self) -> None:
        """Test pre_ingest_hook with no file path."""
        from houyi.rag.skills.kb_ingest.hooks import pre_ingest_hook

        context = MockHookContext(tool_name="write", tool_args={})

        result = await pre_ingest_hook(context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_pre_ingest_hook_regular_file(self) -> None:
        """Test pre_ingest_hook with regular file."""
        from houyi.rag.skills.kb_ingest.hooks import pre_ingest_hook

        context = MockHookContext(
            tool_name="write",
            tool_args={"file_path": "/some/other/file.txt"},
        )

        result = await pre_ingest_hook(context)

        assert result.success is True


class TestPostIngestHook:
    """Tests for post_ingest_hook."""

    @pytest.mark.asyncio
    async def test_post_ingest_hook_read_success(self) -> None:
        """Test post_ingest_hook with read result."""
        from houyi.rag.skills.kb_ingest.hooks import (
            _ingest_state,
            post_ingest_hook,
            reset_ingest_state,
        )

        reset_ingest_state()

        context = MockHookContext(
            tool_name="read",
            tool_result="file content",
        )

        result = await post_ingest_hook(context)

        assert result.success is True
        assert _ingest_state["files_processed"] == 1
        assert "Progress" in result.output

    @pytest.mark.asyncio
    async def test_post_ingest_hook_no_result(self) -> None:
        """Test post_ingest_hook with no result."""
        from houyi.rag.skills.kb_ingest.hooks import post_ingest_hook, reset_ingest_state

        reset_ingest_state()

        context = MockHookContext(tool_name="read", tool_result=None)

        result = await post_ingest_hook(context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_post_ingest_hook_other_tool(self) -> None:
        """Test post_ingest_hook with other tool."""
        from houyi.rag.skills.kb_ingest.hooks import (
            _ingest_state,
            post_ingest_hook,
            reset_ingest_state,
        )

        reset_ingest_state()

        context = MockHookContext(tool_name="write", tool_result="success")

        await post_ingest_hook(context)

        assert _ingest_state["files_processed"] == 0


class TestStopHook:
    """Tests for stop_hook."""

    @pytest.mark.asyncio
    async def test_stop_hook_with_results(self) -> None:
        """Test stop_hook with ingest results."""
        from houyi.rag.skills.kb_ingest.hooks import (
            _ingest_state,
            reset_ingest_state,
            stop_hook,
        )

        reset_ingest_state()
        _ingest_state["files_processed"] = 10
        _ingest_state["chunks_created"] = 50

        context = MockHookContext()

        result = await stop_hook(context)

        assert result.success is True
        assert "Ingest complete" in result.output
        assert "10 files" in result.output
        assert "50 chunks" in result.output

    @pytest.mark.asyncio
    async def test_stop_hook_with_errors(self) -> None:
        """Test stop_hook with errors."""
        from houyi.rag.skills.kb_ingest.hooks import (
            _ingest_state,
            reset_ingest_state,
            stop_hook,
        )

        reset_ingest_state()
        _ingest_state["files_processed"] = 5
        _ingest_state["errors"] = ["error1", "error2"]

        context = MockHookContext()

        result = await stop_hook(context)

        assert result.success is True
        assert "Warnings" in result.output
        assert "2 issues" in result.output

    @pytest.mark.asyncio
    async def test_stop_hook_resets_state(self) -> None:
        """Test stop_hook resets state."""
        from houyi.rag.skills.kb_ingest.hooks import (
            _ingest_state,
            reset_ingest_state,
            stop_hook,
        )

        reset_ingest_state()
        _ingest_state["files_processed"] = 10

        context = MockHookContext()

        await stop_hook(context)

        assert _ingest_state["files_processed"] == 0
