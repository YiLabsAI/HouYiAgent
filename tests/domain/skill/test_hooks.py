from pathlib import Path

import pytest

from houyi.domain.skill.hooks import (
    HookContext,
    HookEvent,
    HookResult,
    HookType,
    SkillHook,
    SkillHooksManager,
)
from houyi.domain.skill.schema import parse_hooks_config, parse_skill_md


class TestSkillHook:
    """Tests for SkillHook dataclass."""

    def test_matches_tool_default(self) -> None:
        """Hook without matcher matches all tools."""
        hook = SkillHook(event=HookEvent.PRE_TOOL_USE)
        assert hook.matches_tool("any_tool")
        assert hook.matches_tool("Write")

    def test_matches_tool_regex(self) -> None:
        """Hook with regex matcher filters tools."""
        hook = SkillHook(event=HookEvent.PRE_TOOL_USE, matcher="Write|Edit")
        assert hook.matches_tool("Write")
        assert hook.matches_tool("Edit")
        assert not hook.matches_tool("Read")
        assert not hook.matches_tool("Shell")

    def test_matches_tool_invalid(self) -> None:
        """Invalid regex returns False."""
        hook = SkillHook(event=HookEvent.PRE_TOOL_USE, matcher="[invalid")
        assert not hook.matches_tool("any_tool")


class TestSkillHooksManager:
    """Tests for SkillHooksManager."""

    def test_register_and_unregister(self) -> None:
        """Test hook registration and unregistration."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        # Create mock skill with hooks
        skill = MagicMock()
        skill.name = "test-skill"
        skill.hooks = [
            SkillHook(event=HookEvent.PRE_TOOL_USE, matcher="Write"),
            SkillHook(event=HookEvent.POST_TOOL_USE),
        ]

        # Register
        manager.register_hooks(skill)
        assert len(manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)) == 1
        assert len(manager.get_registered_hooks(HookEvent.POST_TOOL_USE)) == 1

        # Unregister
        manager.unregister_hooks("test-skill")
        assert len(manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)) == 0
        assert len(manager.get_registered_hooks(HookEvent.POST_TOOL_USE)) == 0

    @pytest.mark.asyncio
    async def test_trigger_hook_empty(self) -> None:
        """Trigger with no hooks returns empty result."""
        manager = SkillHooksManager()
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.PRE_TOOL_USE, context)

        assert result.success
        assert result.output is None

    @pytest.mark.asyncio
    async def test_trigger_handler_hook(self) -> None:
        """Test triggering a handler type hook."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        # Create a test handler
        called = []

        def test_handler(ctx: HookContext) -> str:
            called.append(ctx.tool_name)
            return "hook output"

        skill = MagicMock()
        skill.name = "test-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.PRE_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler=test_handler,
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.PRE_TOOL_USE, context)

        assert result.success
        assert result.output == "hook output"
        assert called == ["Write"]

    @pytest.mark.asyncio
    async def test_trigger_hook_filters(self) -> None:
        """Test that matcher correctly filters tool names."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        called = []

        def test_handler(ctx: HookContext) -> None:
            called.append(ctx.tool_name)

        skill = MagicMock()
        skill.name = "test-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.PRE_TOOL_USE,
                matcher="Write|Edit",
                hook_type=HookType.HANDLER,
                handler=test_handler,
            ),
        ]

        manager.register_hooks(skill)

        # Should trigger for Write
        await manager.trigger_hook(
            HookEvent.PRE_TOOL_USE, HookContext(tool_name="Write"), tool_name="Write"
        )
        assert called == ["Write"]

        # Should not trigger for Read
        called.clear()
        await manager.trigger_hook(
            HookEvent.PRE_TOOL_USE, HookContext(tool_name="Read"), tool_name="Read"
        )
        assert called == []


class TestSkillHooksManagerExtended:
    """Extended tests for SkillHooksManager."""

    @pytest.mark.asyncio
    async def test_trigger_command_hook(self) -> None:
        """Test triggering a command type hook."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "cmd-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.PRE_TOOL_USE,
                hook_type=HookType.COMMAND,
                command="echo hello",
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.PRE_TOOL_USE, context)

        assert result.success
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_command_hook_env(self) -> None:
        """Test command hook with environment variables."""
        import sys
        import tempfile
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            skill = MagicMock()
            skill.name = "env-skill"
            # Use Python to print env var (cross-platform)
            skill.hooks = [
                SkillHook(
                    event=HookEvent.PRE_TOOL_USE,
                    hook_type=HookType.COMMAND,
                    command=f"{sys.executable} -c \"import os; print(os.environ.get('SKILL_DIR', ''))\"",
                ),
            ]

            manager.register_hooks(skill)
            context = HookContext(
                tool_name="Write",
                skill_dir=Path(tmpdir),
                cwd=Path(tmpdir),
            )

            result = await manager.trigger_hook(HookEvent.PRE_TOOL_USE, context)

            assert result.success
            assert tmpdir in (result.output or "")

    @pytest.mark.asyncio
    async def test_command_hook_missing(self) -> None:
        """Test command hook with no command specified."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "no-cmd-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.PRE_TOOL_USE,
                hook_type=HookType.COMMAND,
                command=None,
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.PRE_TOOL_USE, context)
        # Hook fails but manager continues
        assert result.success

    @pytest.mark.asyncio
    async def test_stop_hook_blocks(self) -> None:
        """Test Stop hook that blocks termination (non-zero exit)."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "stop-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.STOP,
                hook_type=HookType.COMMAND,
                command="exit 1",  # Non-zero to block
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext()

        result = await manager.trigger_hook(HookEvent.STOP, context)

        assert result.should_block

    @pytest.mark.asyncio
    async def test_handler_hook_dict(self) -> None:
        """Test handler hook that returns a dict."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        def dict_handler(ctx: HookContext) -> dict:
            return {
                "success": True,
                "output": "dict output",
                "should_block": True,
            }

        skill = MagicMock()
        skill.name = "dict-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.POST_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler=dict_handler,
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.POST_TOOL_USE, context)

        assert result.output == "dict output"
        assert result.should_block

    @pytest.mark.asyncio
    async def test_handler_hook_result(self) -> None:
        """Test handler hook that returns a HookResult directly."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        def result_handler(ctx: HookContext) -> HookResult:
            return HookResult(
                success=True,
                output="direct result",
                inject_to_prompt=True,
            )

        skill = MagicMock()
        skill.name = "result-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.POST_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler=result_handler,
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.POST_TOOL_USE, context)

        assert result.output == "direct result"

    @pytest.mark.asyncio
    async def test_handler_hook_none(self) -> None:
        """Test handler hook that returns None."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        def none_handler(ctx: HookContext) -> None:
            pass

        skill = MagicMock()
        skill.name = "none-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.POST_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler=none_handler,
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.POST_TOOL_USE, context)

        assert result.success

    @pytest.mark.asyncio
    async def test_async_handler_hook(self) -> None:
        """Test async handler hook."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        async def async_handler(ctx: HookContext) -> str:
            return "async result"

        skill = MagicMock()
        skill.name = "async-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.POST_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler=async_handler,
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.POST_TOOL_USE, context)

        assert result.output == "async result"

    @pytest.mark.asyncio
    async def test_handler_hook_exception(self) -> None:
        """Test handler hook that raises an exception."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        def error_handler(ctx: HookContext) -> str:
            raise ValueError("Handler error")

        skill = MagicMock()
        skill.name = "error-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.POST_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler=error_handler,
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.POST_TOOL_USE, context)

        # Manager catches exceptions
        assert result.success

    @pytest.mark.asyncio
    async def test_handler_hook_loads(self) -> None:
        """Test loading handler from dotted path."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        # Use a real module path
        skill = MagicMock()
        skill.name = "path-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.POST_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler_path="os.path.exists",  # Real function
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.POST_TOOL_USE, context)

        assert result.success

    @pytest.mark.asyncio
    async def test_handler_hook_invalid(self) -> None:
        """Test handler hook with invalid dotted path."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "invalid-path-skill"
        skill.hooks = [
            SkillHook(
                event=HookEvent.POST_TOOL_USE,
                hook_type=HookType.HANDLER,
                handler_path="nonexistent.module.function",
            ),
        ]

        manager.register_hooks(skill)
        context = HookContext(tool_name="Write")

        result = await manager.trigger_hook(HookEvent.POST_TOOL_USE, context)

        # Fails gracefully
        assert result.success  # Aggregated result

    def test_get_registered_hooks(self) -> None:
        """Test getting all registered hooks."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "multi-hook-skill"
        skill.hooks = [
            SkillHook(event=HookEvent.PRE_TOOL_USE),
            SkillHook(event=HookEvent.POST_TOOL_USE),
            SkillHook(event=HookEvent.STOP),
        ]

        manager.register_hooks(skill)

        all_hooks = manager.get_registered_hooks()
        assert len(all_hooks) == 3

    def test_clear_hooks(self) -> None:
        """Test clearing all hooks."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "clear-skill"
        skill.hooks = [
            SkillHook(event=HookEvent.PRE_TOOL_USE),
            SkillHook(event=HookEvent.POST_TOOL_USE),
        ]

        manager.register_hooks(skill)
        assert len(manager.get_registered_hooks()) == 2

        manager.clear()
        assert len(manager.get_registered_hooks()) == 0

    def test_unregister_nonexistent_skill(self) -> None:
        """Test unregistering hooks for skill that doesn't exist."""
        manager = SkillHooksManager()

        # Should not raise
        manager.unregister_hooks("nonexistent-skill")

    def test_register_skill_without(self) -> None:
        """Test registering a skill without hooks attribute."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "no-hooks-skill"
        skill.hooks = None

        manager.register_hooks(skill)
        assert len(manager.get_registered_hooks()) == 0

    def test_register_skill_empty(self) -> None:
        """Test registering a skill with empty hooks list."""
        from unittest.mock import MagicMock

        manager = SkillHooksManager()

        skill = MagicMock()
        skill.name = "empty-hooks-skill"
        skill.hooks = []

        manager.register_hooks(skill)
        assert len(manager.get_registered_hooks()) == 0


class TestLoadHandler:
    """Test _load_handler method."""

    def test_load_valid_handler(self) -> None:
        """Test loading a valid handler from path."""
        manager = SkillHooksManager()

        handler = manager._load_handler("os.path.exists")
        assert handler is not None
        assert callable(handler)

    def test_load_handler_format(self) -> None:
        """Test loading handler with no dot in path."""
        manager = SkillHooksManager()

        handler = manager._load_handler("nomodule")
        assert handler is None

    def test_load_handler_import(self) -> None:
        """Test loading handler with import error."""
        manager = SkillHooksManager()

        handler = manager._load_handler("nonexistent.module.func")
        assert handler is None

    def test_load_handler_attr(self) -> None:
        """Test loading handler with missing attribute."""
        manager = SkillHooksManager()

        handler = manager._load_handler("os.path.nonexistent_func")
        assert handler is None


class TestParseSkillMd:
    """Tests for SKILL.md parsing."""

    def test_parse_yaml_frontmatter(self) -> None:
        """Test parsing YAML frontmatter."""
        content = """---
name: test-skill
version: "1.0.0"
description: A test skill
user-invocable: true
allowed-tools: [Read, Write, Edit]
---

# Test Skill

This is the skill body.
"""
        result = parse_skill_md(content)

        assert result["name"] == "test-skill"
        assert result["version"] == "1.0.0"
        assert result["description"] == "A test skill"
        assert result["user-invocable"] is True
        assert result["allowed-tools"] == ["Read", "Write", "Edit"]

    def test_parse_hooks_config(self) -> None:
        """Test parsing hooks configuration."""
        content = """---
name: planning-skill
hooks:
  PreToolUse:
    - matcher: "Write|Edit"
      type: command
      command: "cat task_plan.md"
  PostToolUse:
    - type: handler
      handler: module.function
---
"""
        result = parse_skill_md(content)
        hooks = result.get("hooks", [])

        assert len(hooks) == 2

        pre_hook = hooks[0]
        assert pre_hook.event == HookEvent.PRE_TOOL_USE
        assert pre_hook.matcher == "Write|Edit"
        assert pre_hook.hook_type == HookType.COMMAND
        assert pre_hook.command == "cat task_plan.md"

        post_hook = hooks[1]
        assert post_hook.event == HookEvent.POST_TOOL_USE
        assert post_hook.hook_type == HookType.HANDLER
        assert post_hook.handler_path == "module.function"

    def test_parse_legacy_format(self) -> None:
        """Test parsing legacy skill.md format without frontmatter."""
        content = """# Calculator

## Description

A simple calculator skill.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "expression": {"type": "string"}
  }
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "result": {"type": "number"}
  }
}
```
"""
        result = parse_skill_md(content)

        assert result["name"] == "Calculator"
        assert "calculator" in result["description"].lower()
        assert "input_schema" in result
        assert "output_schema" in result

    def test_parse_claude_hooks(self) -> None:
        """Test parsing Claude's nested hooks format."""
        config = {
            "PreToolUse": [
                {"matcher": "Write|Edit", "hooks": [{"type": "command", "command": "echo hello"}]}
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].event == HookEvent.PRE_TOOL_USE
        assert hooks[0].matcher == "Write|Edit"
        assert hooks[0].hook_type == HookType.COMMAND
        assert hooks[0].command == "echo hello"
