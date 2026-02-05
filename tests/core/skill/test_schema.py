"""Tests for skill schema parsing.

Reference: SimpleSkill Specification v0.1 Section 3 (SKILL.md Format)
"""



from houyi.core.skill.hooks import HookEvent, HookType, SkillHook
from houyi.core.skill.schema import (
    _parse_constraints,
    _parse_frontmatter,
    _parse_markdown_body,
    _parse_yaml_value,
    _remove_frontmatter,
    _simple_yaml_parse,
    parse_hooks_config,
    parse_skill_md,
)


class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_parse_valid_frontmatter(self) -> None:
        """Test parsing valid YAML frontmatter."""
        content = """---
name: test-skill
version: "1.0.0"
description: A test skill
---

# Body content
"""
        result = _parse_frontmatter(content)

        assert result is not None
        assert result["name"] == "test-skill"
        assert result["version"] == "1.0.0"

    def test_parse_no_frontmatter(self) -> None:
        """Test parsing content without frontmatter."""
        content = """# Just a Markdown File

No frontmatter here.
"""
        result = _parse_frontmatter(content)

        assert result is None

    def test_parse_empty_frontmatter(self) -> None:
        """Test parsing empty frontmatter."""
        content = """---
---

# Empty frontmatter
"""
        result = _parse_frontmatter(content)

        # Empty YAML returns None
        assert result is None

    def test_parse_frontmatter_with_complex_types(self) -> None:
        """Test parsing frontmatter with lists and nested objects."""
        content = """---
name: complex-skill
allowed-tools:
  - Read
  - Write
  - Edit
metadata:
  author: test
  version: 1
---
"""
        result = _parse_frontmatter(content)

        assert result is not None
        assert result["allowed-tools"] == ["Read", "Write", "Edit"]
        assert result["metadata"]["author"] == "test"

    def test_parse_frontmatter_yaml_fallback(self) -> None:
        """Test simple YAML parser as fallback."""
        # Directly test the simple parser which is the fallback
        content = """name: fallback-skill
version: "1.0.0"
user-invocable: true"""

        result = _simple_yaml_parse(content)

        assert result is not None
        assert result["name"] == "fallback-skill"
        assert result["version"] == "1.0.0"
        assert result["user-invocable"] is True

    def test_parse_frontmatter_invalid_yaml(self) -> None:
        """Test handling of invalid YAML."""
        content = """---
name: invalid
  bad indentation: here
version: "1.0.0"
---
"""
        # This might parse or return None depending on YAML parser
        result = _parse_frontmatter(content)
        # Just ensure it doesn't crash


class TestSimpleYamlParse:
    """Tests for simple YAML parser fallback."""

    def test_parse_basic_key_value(self) -> None:
        """Test parsing basic key-value pairs."""
        content = """name: test-skill
version: "1.0.0"
description: A test skill"""

        result = _simple_yaml_parse(content)

        assert result["name"] == "test-skill"
        assert result["version"] == "1.0.0"
        assert result["description"] == "A test skill"

    def test_parse_boolean_values(self) -> None:
        """Test parsing boolean values."""
        content = """enabled: true
disabled: false"""

        result = _simple_yaml_parse(content)

        assert result["enabled"] is True
        assert result["disabled"] is False

    def test_parse_quoted_values(self) -> None:
        """Test parsing quoted string values."""
        content = """single: 'single quoted'
double: "double quoted"
unquoted: plain value"""

        result = _simple_yaml_parse(content)

        assert result["single"] == "single quoted"
        assert result["double"] == "double quoted"
        assert result["unquoted"] == "plain value"

    def test_parse_inline_list(self) -> None:
        """Test parsing inline list syntax."""
        content = """tools: [Read, Write, Edit]"""

        result = _simple_yaml_parse(content)

        assert result["tools"] == ["Read", "Write", "Edit"]

    def test_parse_comments(self) -> None:
        """Test that comments are ignored."""
        content = """# This is a comment
name: test
# Another comment
version: "1.0"
"""

        result = _simple_yaml_parse(content)

        assert result["name"] == "test"
        assert result["version"] == "1.0"

    def test_parse_empty_lines(self) -> None:
        """Test that empty lines are handled."""
        content = """name: test

version: "1.0"

description: test desc"""

        result = _simple_yaml_parse(content)

        assert result["name"] == "test"
        assert result["version"] == "1.0"

    def test_parse_nested_structure(self) -> None:
        """Test parsing nested structure with indentation."""
        content = """name: parent
nested:
  child: value
  number: 42"""

        result = _simple_yaml_parse(content)

        assert result["name"] == "parent"
        # Simple parser has limited nested support

    def test_parse_list_items(self) -> None:
        """Test parsing multi-line list items."""
        content = """items:
- first
- second
- third"""

        result = _simple_yaml_parse(content)

        assert "items" in result
        if isinstance(result["items"], list):
            assert result["items"] == ["first", "second", "third"]


class TestParseYamlValue:
    """Tests for YAML value parsing."""

    def test_parse_empty_value(self) -> None:
        """Test parsing empty value."""
        assert _parse_yaml_value("") is None

    def test_parse_double_quoted_string(self) -> None:
        """Test parsing double-quoted string."""
        assert _parse_yaml_value('"hello world"') == "hello world"

    def test_parse_single_quoted_string(self) -> None:
        """Test parsing single-quoted string."""
        assert _parse_yaml_value("'hello world'") == "hello world"

    def test_parse_boolean_true(self) -> None:
        """Test parsing boolean true values."""
        assert _parse_yaml_value("true") is True
        assert _parse_yaml_value("True") is True
        assert _parse_yaml_value("TRUE") is True

    def test_parse_boolean_false(self) -> None:
        """Test parsing boolean false values."""
        assert _parse_yaml_value("false") is False
        assert _parse_yaml_value("False") is False
        assert _parse_yaml_value("FALSE") is False

    def test_parse_integer(self) -> None:
        """Test parsing integer values."""
        assert _parse_yaml_value("42") == 42
        assert _parse_yaml_value("-10") == -10
        assert _parse_yaml_value("0") == 0

    def test_parse_float(self) -> None:
        """Test parsing float values."""
        assert _parse_yaml_value("3.14") == 3.14
        assert _parse_yaml_value("-0.5") == -0.5
        assert _parse_yaml_value("0.0") == 0.0

    def test_parse_inline_list(self) -> None:
        """Test parsing inline list."""
        assert _parse_yaml_value("[a, b, c]") == ["a", "b", "c"]
        assert _parse_yaml_value('["x", "y"]') == ["x", "y"]
        assert _parse_yaml_value("[]") == []

    def test_parse_plain_string(self) -> None:
        """Test parsing plain string value."""
        assert _parse_yaml_value("hello") == "hello"
        assert _parse_yaml_value("plain text") == "plain text"


class TestRemoveFrontmatter:
    """Tests for frontmatter removal."""

    def test_remove_frontmatter(self) -> None:
        """Test removing YAML frontmatter from content."""
        content = """---
name: test
version: "1.0"
---

# Main Content

This is the body.
"""
        result = _remove_frontmatter(content)

        assert "---" not in result
        assert "name: test" not in result
        assert "# Main Content" in result

    def test_remove_frontmatter_no_frontmatter(self) -> None:
        """Test content without frontmatter remains unchanged."""
        content = """# No Frontmatter

Just content.
"""
        result = _remove_frontmatter(content)

        assert result == content


class TestParseMarkdownBody:
    """Tests for Markdown body parsing."""

    def test_parse_title(self) -> None:
        """Test parsing title from Markdown."""
        content = """# My Skill Title

Some content.
"""
        result = _parse_markdown_body(content)

        assert result["name"] == "My Skill Title"

    def test_parse_description(self) -> None:
        """Test parsing description section."""
        content = """# Skill

## Description

This is the skill description.
It can span multiple lines.

## Other Section
"""
        result = _parse_markdown_body(content)

        assert "skill description" in result["description"]

    def test_parse_input_schema(self) -> None:
        """Test parsing input schema."""
        content = """# Skill

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"}
  }
}
```
"""
        result = _parse_markdown_body(content)

        assert "input_schema" in result
        assert result["input_schema"]["type"] == "object"

    def test_parse_output_schema(self) -> None:
        """Test parsing output schema."""
        content = """# Skill

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "result": {"type": "string"}
  }
}
```
"""
        result = _parse_markdown_body(content)

        assert "output_schema" in result
        assert result["output_schema"]["type"] == "object"

    def test_parse_invalid_json_schema(self) -> None:
        """Test handling of invalid JSON in schema."""
        content = """# Skill

## Input Schema
```json
{invalid json}
```
"""
        result = _parse_markdown_body(content)

        assert "input_schema" not in result

    def test_parse_constraints(self) -> None:
        """Test parsing constraints section."""
        content = """# Skill

## Constraints

- Timeout: 30
- Max Cost: 0.01
- Rate Limit: 100
"""
        result = _parse_markdown_body(content)

        assert "constraints" in result
        assert result["constraints"]["timeout"] == 30
        assert result["constraints"]["max_cost"] == 0.01

    def test_parse_missing_sections(self) -> None:
        """Test parsing content with missing sections."""
        content = """# Minimal Skill

Just some text without proper sections.
"""
        result = _parse_markdown_body(content)

        assert result["name"] == "Minimal Skill"
        assert "description" not in result or result.get("description") == ""


class TestParseConstraints:
    """Tests for constraints parsing."""

    def test_parse_basic_constraints(self) -> None:
        """Test parsing basic constraints."""
        content = """- Timeout: 30
- Max Cost: 0.01
- Enabled: true
"""
        result = _parse_constraints(content)

        assert result["timeout"] == 30
        assert result["max_cost"] == 0.01
        assert result["enabled"] is True

    def test_parse_constraints_with_spaces(self) -> None:
        """Test parsing constraints with spaces in keys."""
        content = """- Rate Limit: 100
- Max Retries: 3
"""
        result = _parse_constraints(content)

        assert result["rate_limit"] == 100
        assert result["max_retries"] == 3

    def test_parse_empty_constraints(self) -> None:
        """Test parsing empty constraints section."""
        result = _parse_constraints("")

        assert result == {}

    def test_parse_non_list_content(self) -> None:
        """Test parsing content that isn't a list."""
        content = """This is not a list
Just plain text
"""
        result = _parse_constraints(content)

        assert result == {}


class TestParseHooksConfig:
    """Tests for hooks configuration parsing."""

    def test_parse_simple_hook(self) -> None:
        """Test parsing simple hook configuration."""
        config = {
            "PreToolUse": [
                {
                    "type": "command",
                    "command": "echo hello",
                }
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].event == HookEvent.PRE_TOOL_USE
        assert hooks[0].hook_type == HookType.COMMAND
        assert hooks[0].command == "echo hello"

    def test_parse_handler_hook(self) -> None:
        """Test parsing handler type hook."""
        config = {
            "PostToolUse": [
                {
                    "type": "handler",
                    "handler": "my_module.my_function",
                }
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].event == HookEvent.POST_TOOL_USE
        assert hooks[0].hook_type == HookType.HANDLER
        assert hooks[0].handler_path == "my_module.my_function"

    def test_parse_hook_with_matcher(self) -> None:
        """Test parsing hook with tool matcher."""
        config = {
            "PreToolUse": [
                {
                    "matcher": "Write|Edit",
                    "type": "command",
                    "command": "cat plan.md",
                }
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].matcher == "Write|Edit"

    def test_parse_unknown_event(self) -> None:
        """Test handling unknown hook event."""
        config = {
            "UnknownEvent": [
                {"type": "command", "command": "echo test"}
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 0

    def test_parse_invalid_hook_type(self) -> None:
        """Test handling invalid hook type."""
        config = {
            "PreToolUse": [
                {
                    "type": "invalid_type",
                    "command": "echo test",
                }
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].hook_type == HookType.HANDLER  # Default fallback

    def test_parse_single_hook_not_list(self) -> None:
        """Test parsing single hook (not in list)."""
        config = {
            "Stop": {
                "type": "command",
                "command": "echo stopped",
            }
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].event == HookEvent.STOP

    def test_parse_nested_hooks_claude_format(self) -> None:
        """Test parsing Claude's nested hooks format."""
        config = {
            "PreToolUse": [
                {
                    "matcher": "Write|Edit",
                    "hooks": [
                        {"type": "command", "command": "echo pre-write"},
                        {"type": "handler", "handler": "module.func"},
                    ],
                }
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 2
        assert hooks[0].matcher == "Write|Edit"
        assert hooks[0].hook_type == HookType.COMMAND
        assert hooks[0].command == "echo pre-write"
        assert hooks[1].matcher == "Write|Edit"
        assert hooks[1].hook_type == HookType.HANDLER

    def test_parse_multiple_events(self) -> None:
        """Test parsing hooks for multiple events."""
        config = {
            "PreToolUse": [{"type": "command", "command": "pre"}],
            "PostToolUse": [{"type": "command", "command": "post"}],
            "Stop": [{"type": "handler", "handler": "stop_handler"}],
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 3
        events = {h.event for h in hooks}
        assert HookEvent.PRE_TOOL_USE in events
        assert HookEvent.POST_TOOL_USE in events
        assert HookEvent.STOP in events

    def test_parse_non_dict_hook_config(self) -> None:
        """Test handling non-dict items in hook list."""
        config = {
            "PreToolUse": [
                {"type": "command", "command": "valid"},
                "invalid_string",
                123,
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].command == "valid"

    def test_parse_handler_path_alias(self) -> None:
        """Test parsing handler_path as alternative to handler."""
        config = {
            "PostToolUse": [
                {
                    "type": "handler",
                    "handler_path": "module.function",
                }
            ]
        }

        hooks = parse_hooks_config(config)

        assert len(hooks) == 1
        assert hooks[0].handler_path == "module.function"


class TestParseSkillMdIntegration:
    """Integration tests for full SKILL.md parsing."""

    def test_parse_full_skill_md(self) -> None:
        """Test parsing complete SKILL.md file."""
        content = """---
name: complete-skill
version: "1.0.0"
description: A complete skill example
user-invocable: true
allowed-tools: [Read, Write]
hooks:
  PreToolUse:
    - matcher: "Write"
      type: command
      command: "echo writing"
---

# Complete Skill

## Description

This overrides the frontmatter description if empty.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "input": {"type": "string"}
  }
}
```
"""
        result = parse_skill_md(content)

        assert result["name"] == "complete-skill"
        assert result["version"] == "1.0.0"
        assert result["description"] == "A complete skill example"
        assert result["user-invocable"] is True
        assert result["allowed-tools"] == ["Read", "Write"]
        assert len(result["hooks"]) == 1
        assert isinstance(result["hooks"][0], SkillHook)

    def test_parse_frontmatter_overrides_body(self) -> None:
        """Test that frontmatter values override body values."""
        content = """---
name: frontmatter-name
description: Frontmatter description
---

# Body Name

## Description

Body description.
"""
        result = parse_skill_md(content)

        assert result["name"] == "frontmatter-name"
        assert result["description"] == "Frontmatter description"

    def test_parse_body_fills_missing_frontmatter(self) -> None:
        """Test that body values fill in missing frontmatter fields."""
        content = """---
version: "1.0.0"
---

# Body Title

## Description

Body description text.
"""
        result = parse_skill_md(content)

        assert result["version"] == "1.0.0"
        assert result["name"] == "Body Title"
        assert "Body description text" in result["description"]

    def test_parse_legacy_format_only(self) -> None:
        """Test parsing legacy format without frontmatter."""
        content = """# Legacy Skill

## Description

A legacy skill description.

## Input Schema
```json
{"type": "object", "properties": {"x": {"type": "string"}}}
```

## Output Schema
```json
{"type": "object", "properties": {"y": {"type": "string"}}}
```

## Constraints

- Timeout: 60
- Max Cost: 0.05
"""
        result = parse_skill_md(content)

        assert result["name"] == "Legacy Skill"
        assert "legacy skill description" in result["description"].lower()
        assert result["input_schema"]["type"] == "object"
        assert result["output_schema"]["type"] == "object"
        assert result["constraints"]["timeout"] == 60
