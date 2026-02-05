"""Tests for SimpleSkill manifest parsing.

Reference: SimpleSkill Specification v0.1 Section 3 (Layer A: Manifest)
"""

import json
import tempfile
from pathlib import Path

import pytest

from houyi.core.skill.manifest import (
    ActivationEvent,
    ActivationEventType,
    Contributions,
    ManifestRegistry,
    ResourceContribution,
    SkillContribution,
    SkillManifest,
    ToolContribution,
)
from houyi.core.skill.policy import ModelAutoInvoke, SideEffect


class TestActivationEvent:
    """Test ActivationEvent parsing."""

    def test_parse_always(self):
        event = ActivationEvent.parse("*")
        assert event.event_type == ActivationEventType.ALWAYS
        assert event.pattern is None

    def test_parse_on_startup(self):
        event = ActivationEvent.parse("onStartup")
        assert event.event_type == ActivationEventType.ON_STARTUP
        assert event.pattern is None

    def test_parse_on_command(self):
        event = ActivationEvent.parse("onCommand:planning.create")
        assert event.event_type == ActivationEventType.ON_COMMAND
        assert event.pattern == "planning.create"

    def test_parse_on_language(self):
        event = ActivationEvent.parse("onLanguage:python")
        assert event.event_type == ActivationEventType.ON_LANGUAGE
        assert event.pattern == "python"

    def test_parse_on_filesystem(self):
        event = ActivationEvent.parse("onFileSystem:**/*.md")
        assert event.event_type == ActivationEventType.ON_FILE_SYSTEM
        assert event.pattern == "**/*.md"

    def test_parse_unknown_type(self):
        event = ActivationEvent.parse("onUnknown:test")
        assert event.event_type == ActivationEventType.ON_STARTUP


class TestToolContribution:
    """Test ToolContribution parsing."""

    def test_from_dict(self):
        data = {
            "id": "file_write",
            "description": "Write to files",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "outputSchema": {"type": "object", "properties": {"success": {"type": "boolean"}}},
            "execution": {"form": "in-process", "entry": "module.function"},
        }
        tool = ToolContribution.from_dict(data)
        assert tool.id == "file_write"
        assert tool.description == "Write to files"
        assert tool.input_schema["type"] == "object"
        assert tool.execution["form"] == "in-process"


class TestSkillContribution:
    """Test SkillContribution parsing."""

    def test_from_dict(self):
        data = {
            "id": "planning",
            "description": "Task planning skill",
            "invocationPolicy": {
                "modelAutoInvoke": "allow",
                "userInvocable": True,
                "sideEffect": "filesystem",
            },
            "toolRefs": ["file_read", "file_write"],
            "resources": ["templates"],
        }
        skill = SkillContribution.from_dict(data)
        assert skill.id == "planning"
        assert skill.description == "Task planning skill"
        assert skill.invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW
        assert skill.invocation_policy.side_effect == SideEffect.FILESYSTEM
        assert skill.tool_refs == ["file_read", "file_write"]


class TestResourceContribution:
    """Test ResourceContribution parsing."""

    def test_from_dict(self):
        data = {
            "id": "templates",
            "uri": "file://templates/",
            "name": "Planning Templates",
            "description": "Template files for planning",
            "mimeType": "text/markdown",
        }
        resource = ResourceContribution.from_dict(data)
        assert resource.id == "templates"
        assert resource.uri == "file://templates/"
        assert resource.mime_type == "text/markdown"


class TestSkillManifest:
    """Test SkillManifest class."""

    def test_from_dict_minimal(self):
        data = {
            "id": "houyi.test-skill",
            "version": "1.0.0",
            "name": "Test Skill",
            "description": "A test skill",
        }
        manifest = SkillManifest.from_dict(data)
        assert manifest.id == "houyi.test-skill"
        assert manifest.version == "1.0.0"
        assert manifest.name == "Test Skill"

    def test_from_dict_full(self):
        data = {
            "id": "houyi.planning",
            "version": "1.0.0",
            "name": "Planning Skill",
            "description": "Task planning and tracking",
            "engines": {"host": "houyi>=0.1"},
            "activationEvents": ["onCommand:planning.create", "*"],
            "contributions": {
                "tools": [
                    {"id": "plan_create", "description": "Create a plan"}
                ],
                "skills": [
                    {"id": "planning", "description": "Planning skill"}
                ],
                "resources": [],
            },
            "permissions": {
                "filesystem": {"read": True, "write": True},
                "network": {"enabled": False},
            },
            "trust": {
                "provider": "github:houyi/skills",
                "verified": True,
            },
            "main": "index.py",
        }
        manifest = SkillManifest.from_dict(data)

        assert manifest.id == "houyi.planning"
        assert len(manifest.activation_events) == 2
        assert manifest.activation_events[0].event_type == ActivationEventType.ON_COMMAND
        assert manifest.activation_events[1].event_type == ActivationEventType.ALWAYS
        assert len(manifest.contributions.tools) == 1
        assert manifest.permissions.filesystem.read
        assert manifest.trust.provider == "github:houyi/skills"
        assert manifest.main == "index.py"

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "simpleskill.json"
            data = {
                "id": "houyi.test",
                "version": "1.0.0",
                "name": "Test",
                "description": "Test skill",
            }
            with open(manifest_path, "w") as f:
                json.dump(data, f)

            manifest = SkillManifest.from_file(manifest_path)
            assert manifest.id == "houyi.test"
            assert manifest.manifest_path == manifest_path

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SkillManifest.from_file("/nonexistent/path.json")

    def test_to_dict(self):
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test skill",
        )
        data = manifest.to_dict()
        assert data["id"] == "houyi.test"
        assert data["version"] == "1.0.0"

    def test_validate_valid(self):
        manifest = SkillManifest(
            id="houyi.test-skill",
            version="1.0.0",
            name="Test Skill",
            description="A test skill",
        )
        errors = manifest.validate()
        assert len(errors) == 0

    def test_validate_missing_id(self):
        manifest = SkillManifest(
            id="",
            version="1.0.0",
            name="Test",
            description="Test",
        )
        errors = manifest.validate()
        assert any("id" in e.lower() for e in errors)

    def test_validate_invalid_id_format(self):
        manifest = SkillManifest(
            id="invalid_id",
            version="1.0.0",
            name="Test",
            description="Test",
        )
        errors = manifest.validate()
        assert any("id format" in e.lower() for e in errors)

    def test_validate_invalid_version(self):
        manifest = SkillManifest(
            id="houyi.test",
            version="invalid",
            name="Test",
            description="Test",
        )
        errors = manifest.validate()
        assert any("version" in e.lower() for e in errors)

    def test_get_tool(self):
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
            contributions=Contributions(
                tools=[
                    ToolContribution(id="tool1", description="Tool 1"),
                    ToolContribution(id="tool2", description="Tool 2"),
                ],
            ),
        )
        tool = manifest.get_tool("tool1")
        assert tool is not None
        assert tool.id == "tool1"

        assert manifest.get_tool("nonexistent") is None

    def test_get_skill(self):
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
            contributions=Contributions(
                skills=[
                    SkillContribution(id="skill1", description="Skill 1"),
                ],
            ),
        )
        skill = manifest.get_skill("skill1")
        assert skill is not None
        assert skill.id == "skill1"

    def test_matches_activation_event_always(self):
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
            activation_events=[ActivationEvent(event_type=ActivationEventType.ALWAYS)],
        )
        assert manifest.matches_activation_event(ActivationEventType.ON_COMMAND, "any")
        assert manifest.matches_activation_event(ActivationEventType.ON_STARTUP)

    def test_matches_activation_event_on_command(self):
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
            activation_events=[
                ActivationEvent(event_type=ActivationEventType.ON_COMMAND, pattern="plan.create")
            ],
        )
        assert manifest.matches_activation_event(ActivationEventType.ON_COMMAND, "plan.create")
        assert not manifest.matches_activation_event(ActivationEventType.ON_COMMAND, "other")

    def test_matches_activation_event_on_filesystem(self):
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
            activation_events=[
                ActivationEvent(event_type=ActivationEventType.ON_FILE_SYSTEM, pattern="**/*.md")
            ],
        )
        assert manifest.matches_activation_event(ActivationEventType.ON_FILE_SYSTEM, "docs/README.md")
        assert not manifest.matches_activation_event(ActivationEventType.ON_FILE_SYSTEM, "script.py")


class TestManifestRegistry:
    """Test ManifestRegistry class."""

    def test_register_and_get(self):
        registry = ManifestRegistry()
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
        )
        registry.register(manifest)

        retrieved = registry.get_manifest("houyi.test")
        assert retrieved is not None
        assert retrieved.id == "houyi.test"

    def test_register_duplicate(self):
        registry = ManifestRegistry()
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
        )
        registry.register(manifest)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(manifest)

    def test_unregister(self):
        registry = ManifestRegistry()
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
        )
        registry.register(manifest)
        registry.unregister("houyi.test")

        assert registry.get_manifest("houyi.test") is None

    def test_get_manifest_for_tool(self):
        registry = ManifestRegistry()
        manifest = SkillManifest(
            id="houyi.test",
            version="1.0.0",
            name="Test",
            description="Test",
            contributions=Contributions(
                tools=[ToolContribution(id="my_tool", description="A tool")],
            ),
        )
        registry.register(manifest)

        found = registry.get_manifest_for_tool("my_tool")
        assert found is not None
        assert found.id == "houyi.test"

    def test_list_manifests(self):
        registry = ManifestRegistry()
        for i in range(3):
            manifest = SkillManifest(
                id=f"houyi.test{i}",
                version="1.0.0",
                name=f"Test {i}",
                description="Test",
            )
            registry.register(manifest)

        manifests = registry.list_manifests()
        assert len(manifests) == 3

    def test_find_by_activation(self):
        registry = ManifestRegistry()

        # Manifest that activates on command
        m1 = SkillManifest(
            id="houyi.cmd",
            version="1.0.0",
            name="Cmd",
            description="Test",
            activation_events=[
                ActivationEvent(event_type=ActivationEventType.ON_COMMAND, pattern="test")
            ],
        )
        registry.register(m1)

        # Manifest that always activates
        m2 = SkillManifest(
            id="houyi.always",
            version="1.0.0",
            name="Always",
            description="Test",
            activation_events=[
                ActivationEvent(event_type=ActivationEventType.ALWAYS)
            ],
        )
        registry.register(m2)

        # Find by command
        found = registry.find_by_activation(ActivationEventType.ON_COMMAND, "test")
        assert len(found) == 2  # Both match (always matches everything)

        found = registry.find_by_activation(ActivationEventType.ON_COMMAND, "other")
        assert len(found) == 1  # Only always matches
