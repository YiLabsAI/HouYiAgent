"""Skill schema parser for YAML frontmatter + Markdown format.

Supports both Claude/AgentSkills SKILL.md format and HouYi skill.md format.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from houyi.core.skill.hooks import HookEvent, HookType, SkillHook

logger = logging.getLogger(__name__)


def parse_skill_md(content: str) -> dict[str, Any]:
    """Parse SKILL.md content with YAML frontmatter.

    Args:
        content: Raw SKILL.md content

    Returns:
        Parsed skill data including metadata, schemas, and hooks
    """
    result: dict[str, Any] = {}

    # Try to parse YAML frontmatter first
    frontmatter = _parse_frontmatter(content)
    if frontmatter:
        result.update(frontmatter)
        # Remove frontmatter from content for body parsing
        content = _remove_frontmatter(content)

    # Parse Markdown body (fallback for fields not in frontmatter)
    body_data = _parse_markdown_body(content)

    # Merge: frontmatter takes precedence
    for key, value in body_data.items():
        if key not in result or result[key] is None:
            result[key] = value

    # Parse hooks if present in frontmatter
    if "hooks" in result and isinstance(result["hooks"], dict):
        result["hooks"] = parse_hooks_config(result["hooks"])

    return result


def _parse_frontmatter(content: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter from content.

    Frontmatter is delimited by --- at start and end.
    """
    # Match YAML frontmatter pattern
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return None

    yaml_content = match.group(1)

    try:
        # Try to import PyYAML
        import yaml

        return yaml.safe_load(yaml_content)
    except ImportError:
        # Fallback: simple YAML-like parsing for common fields
        logger.warning("PyYAML not installed, using simple parser")
        return _simple_yaml_parse(yaml_content)
    except Exception as e:
        logger.warning("Failed to parse YAML frontmatter: %s", e)
        return None


def _simple_yaml_parse(content: str) -> dict[str, Any]:
    """Simple YAML-like parser for basic key-value pairs.

    Handles:
    - name: value
    - key: "quoted value"
    - key: [list, items]
    - key: true/false
    """
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_indent = 0
    nested_dict: dict[str, Any] = {}

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check indentation level
        indent = len(line) - len(line.lstrip())

        # Simple key: value pattern
        kv_match = re.match(r"^([a-zA-Z_-]+):\s*(.*)$", stripped)
        if kv_match:
            key = kv_match.group(1)
            value = kv_match.group(2).strip()

            if indent == 0:
                # Top-level key
                if value:
                    result[key] = _parse_yaml_value(value)
                else:
                    # Might be a nested structure
                    current_key = key
                    current_indent = indent
                    nested_dict = {}
            elif current_key and indent > current_indent:
                # Nested key
                if value:
                    nested_dict[key] = _parse_yaml_value(value)

        # List item pattern
        list_match = re.match(r"^-\s+(.*)$", stripped)
        if list_match and current_key:
            value = list_match.group(1).strip()
            if current_key not in result:
                result[current_key] = []
            if isinstance(result[current_key], list):
                result[current_key].append(_parse_yaml_value(value))

    # Save any remaining nested dict
    if current_key and nested_dict:
        result[current_key] = nested_dict

    return result


def _parse_yaml_value(value: str) -> Any:
    """Parse a YAML value string into appropriate Python type."""
    if not value:
        return None

    # Remove quotes
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # List
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [item.strip().strip("\"'") for item in items if item.strip()]

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def _remove_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content."""
    pattern = r"^---\s*\n.*?\n---\s*\n"
    return re.sub(pattern, "", content, count=1, flags=re.DOTALL)


def _parse_markdown_body(content: str) -> dict[str, Any]:
    """Parse Markdown body for skill fields (HouYi legacy format)."""
    result: dict[str, Any] = {}

    # Title -> name
    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    if title_match:
        result["name"] = title_match.group(1).strip()

    # Description section
    desc_match = re.search(r"## Description\s+(.+?)(?=##|$)", content, re.DOTALL)
    if desc_match:
        result["description"] = desc_match.group(1).strip()

    # Input Schema
    input_match = re.search(r"## Input Schema\s+```json\s+(.+?)\s+```", content, re.DOTALL)
    if input_match:
        try:
            result["input_schema"] = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            pass

    # Output Schema
    output_match = re.search(r"## Output Schema\s+```json\s+(.+?)\s+```", content, re.DOTALL)
    if output_match:
        try:
            result["output_schema"] = json.loads(output_match.group(1))
        except json.JSONDecodeError:
            pass

    # Constraints section
    constraints_match = re.search(r"## Constraints\s+(.+?)(?=##|$)", content, re.DOTALL)
    if constraints_match:
        result["constraints"] = _parse_constraints(constraints_match.group(1))

    return result


def _parse_constraints(content: str) -> dict[str, Any]:
    """Parse constraints from Markdown list."""
    constraints: dict[str, Any] = {}
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            # Parse "- Key: Value" format
            kv_match = re.match(r"-\s*([^:]+):\s*(.+)", line)
            if kv_match:
                key = kv_match.group(1).strip().lower().replace(" ", "_")
                value = kv_match.group(2).strip()
                constraints[key] = _parse_yaml_value(value)
    return constraints


def parse_hooks_config(config: dict[str, Any]) -> list[SkillHook]:
    """Convert YAML hooks configuration to SkillHook objects.

    Args:
        config: Hooks config from YAML frontmatter, e.g.:
            {
                "PreToolUse": [
                    {"matcher": "Write|Edit", "type": "command", "command": "..."}
                ],
                "Stop": [
                    {"type": "handler", "handler": "module.function"}
                ]
            }

    Returns:
        List of SkillHook objects
    """
    hooks: list[SkillHook] = []

    for event_name, hook_list in config.items():
        # Map event name to HookEvent enum
        try:
            event = HookEvent(event_name)
        except ValueError:
            logger.warning("Unknown hook event: %s", event_name)
            continue

        # Handle both list and single hook
        if not isinstance(hook_list, list):
            hook_list = [hook_list]

        for hook_config in hook_list:
            if not isinstance(hook_config, dict):
                continue

            # Determine hook type
            hook_type_str = hook_config.get("type", "handler")
            try:
                hook_type = HookType(hook_type_str)
            except ValueError:
                hook_type = HookType.HANDLER

            # Handle nested 'hooks' structure (Claude format)
            nested_hooks = hook_config.get("hooks", [])
            if nested_hooks and isinstance(nested_hooks, list):
                # Claude format: {"matcher": "...", "hooks": [{"type": "command", ...}]}
                matcher = hook_config.get("matcher")
                for nested in nested_hooks:
                    if isinstance(nested, dict):
                        nested_type_str = nested.get("type", "handler")
                        try:
                            nested_type = HookType(nested_type_str)
                        except ValueError:
                            nested_type = HookType.HANDLER

                        hooks.append(
                            SkillHook(
                                event=event,
                                matcher=matcher,
                                hook_type=nested_type,
                                command=nested.get("command"),
                                handler_path=nested.get("handler") or nested.get("handler_path"),
                            )
                        )
            else:
                # Simple format: {"matcher": "...", "type": "command", "command": "..."}
                hooks.append(
                    SkillHook(
                        event=event,
                        matcher=hook_config.get("matcher"),
                        hook_type=hook_type,
                        command=hook_config.get("command"),
                        handler_path=hook_config.get("handler") or hook_config.get("handler_path"),
                    )
                )

    return hooks
