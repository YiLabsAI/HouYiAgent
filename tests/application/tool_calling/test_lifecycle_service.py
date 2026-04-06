from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.application.tool_calling.lifecycle_service import (
    _ToolCallLifecycleService,
)


def _mock_runner(*, hooks_mgr=None, policy_enforcer=None) -> MagicMock:
    runner = MagicMock()
    runner.skill_hooks_manager = hooks_mgr
    runner.policy_enforcer = policy_enforcer
    return runner


class TestSessionStartHook:
    @pytest.mark.asyncio
    async def test_no_hooks_manager(self) -> None:
        svc = _ToolCallLifecycleService(_mock_runner())
        await svc.trigger_session_start_hook(10, 3, 2)  # should not raise

    @pytest.mark.asyncio
    async def test_triggers_hook(self) -> None:
        mgr = AsyncMock()
        mgr.trigger_hook = AsyncMock()
        svc = _ToolCallLifecycleService(_mock_runner(hooks_mgr=mgr))
        await svc.trigger_session_start_hook(10, 3, 2)
        mgr.trigger_hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_error_ignored(self) -> None:
        mgr = AsyncMock()
        mgr.trigger_hook = AsyncMock(side_effect=RuntimeError("boom"))
        svc = _ToolCallLifecycleService(_mock_runner(hooks_mgr=mgr))
        await svc.trigger_session_start_hook(10, 3, 2)  # should not raise


class TestApplyToolRouter:
    def test_no_restrictions(self) -> None:
        svc = _ToolCallLifecycleService(_mock_runner())
        tools = [{"function": {"name": "a"}}, {"function": {"name": "b"}}]
        with patch("houyi.domain.skill.tool_router.ToolRouter") as MockRouter:
            instance = MockRouter.return_value
            instance.has_restrictions = False
            result = svc.apply_tool_router([], tools)
        assert result is tools

    def test_with_restrictions(self) -> None:
        svc = _ToolCallLifecycleService(_mock_runner())
        tools = [{"function": {"name": "a"}}, {"function": {"name": "b"}}]
        with patch("houyi.domain.skill.tool_router.ToolRouter") as MockRouter:
            instance = MockRouter.return_value
            instance.has_restrictions = True
            instance.filter_tools.return_value = [tools[0]]
            result = svc.apply_tool_router([], tools)
        assert len(result) == 1


class TestPostToolUseHook:
    @pytest.mark.asyncio
    async def test_no_hooks_manager(self) -> None:
        svc = _ToolCallLifecycleService(_mock_runner())
        await svc.trigger_post_tool_use_hook("t", {}, {"raw": "r"}, None)

    @pytest.mark.asyncio
    async def test_triggers_hook(self) -> None:
        mgr = AsyncMock()
        hook_result = MagicMock()
        hook_result.output = "some output"
        mgr.trigger_hook = AsyncMock(return_value=hook_result)
        svc = _ToolCallLifecycleService(_mock_runner(hooks_mgr=mgr))
        await svc.trigger_post_tool_use_hook("t", {}, {"raw": "r"}, None)
        mgr.trigger_hook.assert_called_once()


class TestStopHook:
    @pytest.mark.asyncio
    async def test_no_hooks_manager(self) -> None:
        svc = _ToolCallLifecycleService(_mock_runner())
        await svc.trigger_stop_hook([])

    @pytest.mark.asyncio
    async def test_triggers_with_trace(self) -> None:
        mgr = AsyncMock()
        mgr.trigger_hook = AsyncMock()
        svc = _ToolCallLifecycleService(_mock_runner(hooks_mgr=mgr))
        await svc.trigger_stop_hook([{"tool_call_id": "c1"}, {"no_id": True}])
        mgr.trigger_hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_error_ignored(self) -> None:
        mgr = AsyncMock()
        mgr.trigger_hook = AsyncMock(side_effect=RuntimeError("boom"))
        svc = _ToolCallLifecycleService(_mock_runner(hooks_mgr=mgr))
        await svc.trigger_stop_hook([])  # should not raise


class TestRunPreprocessors:
    @pytest.mark.asyncio
    async def test_runs_pipeline(self) -> None:
        svc = _ToolCallLifecycleService(_mock_runner())
        msgs = [{"role": "user", "content": "hi"}]
        with patch("houyi.domain.skill.preprocessor.PreprocessorPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            pp_result = MagicMock(success=True)
            instance.run = AsyncMock(return_value=[pp_result])
            instance.inject.return_value = msgs
            result = await svc.run_preprocessors([], msgs)
        assert result is msgs

    @pytest.mark.asyncio
    async def test_error_non_fatal(self) -> None:
        svc = _ToolCallLifecycleService(_mock_runner())
        msgs = [{"role": "user", "content": "hi"}]
        with patch("houyi.domain.skill.preprocessor.PreprocessorPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.run = AsyncMock(side_effect=RuntimeError("bad"))
            result = await svc.run_preprocessors([], msgs)
        assert result is msgs
