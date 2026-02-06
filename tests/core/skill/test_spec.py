"""Tests for SkillSpec class.

Reference: SimpleSkill Specification v0.1 Section 2 (Skill Definition)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from houyi.core.skill.spec import ExecutionMode, SkillSpec


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_execution_modes(self) -> None:
        """Test all execution mode values."""
        assert ExecutionMode.PYTHON == "python"
        assert ExecutionMode.CLIENT == "client"
        assert ExecutionMode.PLUGIN == "plugin"
        assert ExecutionMode.MCP == "mcp"


class TestSkillSpecBasics:
    """Tests for basic SkillSpec functionality."""

    def test_create_skill_spec(self) -> None:
        """Test creating a SkillSpec with required fields."""

        class InputModel(BaseModel):
            query: str

        class OutputModel(BaseModel):
            result: str

        spec = SkillSpec(
            name="test-skill",
            description="A test skill",
            input_schema=InputModel,
            output_schema=OutputModel,
        )

        assert spec.name == "test-skill"
        assert spec.description == "A test skill"
        assert spec.input_schema == InputModel
        assert spec.output_schema == OutputModel
        assert spec.executor is None
        assert spec.execution_mode == ExecutionMode.PYTHON

    def test_skill_spec_defaults(self) -> None:
        """Test SkillSpec default values."""

        class EmptyInput(BaseModel):
            pass

        class EmptyOutput(BaseModel):
            pass

        spec = SkillSpec(
            name="default-skill",
            description="Test defaults",
            input_schema=EmptyInput,
            output_schema=EmptyOutput,
        )

        assert spec.skill_md_path is None
        assert spec.skill_dir is None
        assert spec.constraints == {}
        assert spec.metadata == {}
        assert spec.version is None
        assert spec.user_invocable is True
        assert spec.allowed_tools == []
        assert spec.hooks == []

    def test_to_tool_schema(self) -> None:
        """Test converting SkillSpec to OpenAI function calling schema."""

        class CalcInput(BaseModel):
            expression: str

        class CalcOutput(BaseModel):
            result: float

        spec = SkillSpec(
            name="calculator",
            description="Evaluate math expressions",
            input_schema=CalcInput,
            output_schema=CalcOutput,
        )

        schema = spec.to_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert schema["function"]["description"] == "Evaluate math expressions"
        assert "properties" in schema["function"]["parameters"]
        assert "expression" in schema["function"]["parameters"]["properties"]

    def test_bind_executor(self) -> None:
        """Test binding an executor function to a skill."""

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        spec = SkillSpec(
            name="test",
            description="test",
            input_schema=Input,
            output_schema=Output,
        )

        def my_executor(x: int) -> int:
            return x * 2

        assert spec.executor is None
        spec.bind_executor(my_executor)
        assert spec.executor is my_executor


class TestSkillSpecFromFile:
    """Tests for loading SkillSpec from files."""

    def test_from_file_with_yaml_frontmatter(self) -> None:
        """Test loading skill from SKILL.md with YAML frontmatter."""
        content = """---
name: yaml-skill
version: "1.0.0"
description: A skill loaded from YAML frontmatter
user-invocable: true
allowed-tools: [Read, Write]
---

# YAML Skill

This is the body content.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()

            spec = SkillSpec.from_file(f.name)

            assert spec.name == "yaml-skill"
            assert spec.version == "1.0.0"
            assert spec.description == "A skill loaded from YAML frontmatter"
            assert spec.user_invocable is True
            assert spec.allowed_tools == ["Read", "Write"]

    def test_from_file_legacy_format(self) -> None:
        """Test loading skill from legacy skill.md format."""
        content = """# Legacy Skill

## Description

A skill in the legacy markdown format.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "text": {"type": "string"}
  },
  "required": ["text"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "processed": {"type": "string"}
  }
}
```
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()

            spec = SkillSpec.from_file(f.name)

            assert spec.name == "Legacy Skill"
            assert "legacy markdown format" in spec.description

    def test_from_file_with_skill_dir(self) -> None:
        """Test loading skill with explicit skill_dir."""
        content = """---
name: dir-skill
description: Skill with directory
---

# Dir Skill
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_path = Path(tmpdir) / "SKILL.md"
            skill_path.write_text(content)

            spec = SkillSpec.from_file(str(skill_path), skill_dir=tmpdir)

            assert spec.skill_dir == Path(tmpdir)
            assert spec.skill_md_path == str(skill_path)

    def test_from_file_auto_detect_skill_dir(self) -> None:
        """Test auto-detection of skill directory from SKILL.md path."""
        content = """---
name: auto-dir-skill
description: Auto-detected directory
---
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_path = Path(tmpdir) / "SKILL.md"
            skill_path.write_text(content)

            spec = SkillSpec.from_file(str(skill_path))

            assert spec.skill_dir == Path(tmpdir)


class TestSkillSpecFromUrl:
    """Tests for loading SkillSpec from URLs."""

    @patch("urllib.request.urlopen")
    def test_from_url_success(self, mock_urlopen: MagicMock) -> None:
        """Test loading skill from URL."""
        content = """---
name: remote-skill
description: A remotely loaded skill
version: "2.0.0"
---

# Remote Skill
"""
        mock_response = MagicMock()
        mock_response.read.return_value = content.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                spec = SkillSpec.from_url("https://example.com/skills/test/skill.md", cache=True)

                assert spec.name == "remote-skill"
                assert spec.version == "2.0.0"

    @patch("urllib.request.urlopen")
    def test_from_url_no_cache(self, mock_urlopen: MagicMock) -> None:
        """Test loading skill from URL without caching."""
        content = """---
name: no-cache-skill
description: Not cached
---
"""
        mock_response = MagicMock()
        mock_response.read.return_value = content.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        spec = SkillSpec.from_url("https://example.com/skill.md", cache=False)

        assert spec.name == "no-cache-skill"
        assert spec.skill_md_path == "https://example.com/skill.md"

    @patch("urllib.request.urlopen")
    def test_from_url_network_error(self, mock_urlopen: MagicMock) -> None:
        """Test handling network errors when loading from URL."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Network error")

        with pytest.raises(urllib.error.URLError) as exc_info:
            SkillSpec.from_url("https://example.com/skill.md")

        assert "Failed to load skill" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_from_url_parse_error(self, mock_urlopen: MagicMock) -> None:
        """Test handling parse errors when loading from URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"invalid content that causes parse error"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # This should parse but with default/unknown values
        spec = SkillSpec.from_url("https://example.com/skill.md", cache=False)
        assert spec.name == "unknown"


class TestSkillSpecFromRegistry:
    """Tests for loading SkillSpec from registry."""

    @patch.object(SkillSpec, "from_url")
    def test_from_registry_latest(self, mock_from_url: MagicMock) -> None:
        """Test loading latest version from registry."""
        mock_from_url.return_value = MagicMock(spec=SkillSpec)

        SkillSpec.from_registry("web_search")

        mock_from_url.assert_called_once()
        call_args = mock_from_url.call_args
        assert "web_search/skill.md" in call_args[0][0]
        assert call_args[1]["cache"] is True

    @patch.object(SkillSpec, "from_url")
    def test_from_registry_with_version(self, mock_from_url: MagicMock) -> None:
        """Test loading specific version from registry."""
        mock_from_url.return_value = MagicMock(spec=SkillSpec)

        SkillSpec.from_registry("web_search", version="v1.0.0")

        mock_from_url.assert_called_once()
        call_args = mock_from_url.call_args
        assert "web_search/v1.0.0/skill.md" in call_args[0][0]


class TestSkillSpecParsing:
    """Tests for internal parsing methods."""

    def test_parse_skill_md_basic(self) -> None:
        """Test parsing basic skill.md content."""
        content = """# My Skill

## Description

This is a description.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "input": {"type": "string"}
  }
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "output": {"type": "string"}
  }
}
```
"""
        result = SkillSpec._parse_skill_md(content)

        assert result["name"] == "My Skill"
        assert result["description"] == "This is a description."
        assert "properties" in result["input_schema"]
        assert "properties" in result["output_schema"]

    def test_parse_skill_md_missing_sections(self) -> None:
        """Test parsing skill.md with missing sections."""
        content = """# Minimal Skill

Some content without proper sections.
"""
        result = SkillSpec._parse_skill_md(content)

        assert result["name"] == "Minimal Skill"
        assert result["description"] == ""
        assert "input_schema" not in result

    def test_json_to_pydantic_with_properties(self) -> None:
        """Test converting JSON schema to Pydantic model."""
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["name"],
        }

        model = SkillSpec._json_to_pydantic(json_schema, "TestModel")

        assert issubclass(model, BaseModel)
        # Check that the model has the expected fields
        fields = model.model_fields
        assert "name" in fields
        assert "age" in fields
        assert "active" in fields

    def test_json_to_pydantic_empty_schema(self) -> None:
        """Test converting empty JSON schema."""
        model = SkillSpec._json_to_pydantic({}, "EmptyModel")

        assert issubclass(model, BaseModel)
        assert "Empty" in model.__name__

    def test_json_to_pydantic_no_properties(self) -> None:
        """Test converting JSON schema without properties."""
        json_schema = {"type": "object"}

        model = SkillSpec._json_to_pydantic(json_schema, "NoPropsModel")

        assert issubclass(model, BaseModel)

    def test_json_type_to_python(self) -> None:
        """Test JSON type to Python type mapping."""
        assert SkillSpec._json_type_to_python("string") is str
        assert SkillSpec._json_type_to_python("integer") is int
        assert SkillSpec._json_type_to_python("number") is float
        assert SkillSpec._json_type_to_python("boolean") is bool
        assert SkillSpec._json_type_to_python("array") is list
        assert SkillSpec._json_type_to_python("object") is dict
        # Unknown type defaults to str
        assert SkillSpec._json_type_to_python("unknown") is str


class TestSkillSpecExport:
    """Tests for exporting SkillSpec to files."""

    def test_export_skill_md_basic(self) -> None:
        """Test basic skill.md export."""

        class InputModel(BaseModel):
            query: str

        class OutputModel(BaseModel):
            result: str

        spec = SkillSpec(
            name="export-skill",
            description="A skill to export",
            input_schema=InputModel,
            output_schema=OutputModel,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "exported_skill.md"
            spec.export_skill_md(str(export_path))

            content = export_path.read_text()

            assert "# export-skill" in content
            assert "## Description" in content
            assert "A skill to export" in content
            assert "## Input Schema" in content
            assert "## Output Schema" in content
            assert "```json" in content

    def test_export_skill_md_with_metadata(self) -> None:
        """Test exporting skill.md with metadata."""

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        spec = SkillSpec(
            name="meta-skill",
            description="Skill with metadata",
            input_schema=Input,
            output_schema=Output,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "meta_skill.md"
            spec.export_skill_md(
                str(export_path),
                metadata={"language": "Python", "runtime": "async"},
            )

            content = export_path.read_text()

            assert "## Implementation" in content
            assert "Language: Python" in content
            assert "Runtime: async" in content

    def test_export_skill_md_with_examples(self) -> None:
        """Test exporting skill.md with examples."""

        class Input(BaseModel):
            text: str

        class Output(BaseModel):
            length: int

        spec = SkillSpec(
            name="example-skill",
            description="Skill with examples",
            input_schema=Input,
            output_schema=Output,
        )

        examples = [
            {"input": {"text": "hello"}, "output": {"length": 5}},
            {"input": {"text": "world"}, "output": {"length": 5}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "example_skill.md"
            spec.export_skill_md(str(export_path), examples=examples)

            content = export_path.read_text()

            assert "## Examples" in content
            assert "### Example 1" in content
            assert "### Example 2" in content
            assert '"text": "hello"' in content

    def test_export_skill_md_with_constraints(self) -> None:
        """Test exporting skill.md with constraints."""

        class Input(BaseModel):
            data: str

        class Output(BaseModel):
            processed: str

        spec = SkillSpec(
            name="constrained-skill",
            description="Skill with constraints",
            input_schema=Input,
            output_schema=Output,
            constraints={"timeout": 30, "max_cost": 0.01},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "constrained_skill.md"
            spec.export_skill_md(str(export_path))

            content = export_path.read_text()

            assert "## Constraints" in content
            assert "Timeout: 30" in content
            assert "Max Cost: 0.01" in content

    def test_export_creates_parent_directories(self) -> None:
        """Test that export creates parent directories if needed."""

        class Input(BaseModel):
            x: str

        class Output(BaseModel):
            y: str

        spec = SkillSpec(
            name="nested-skill",
            description="Skill in nested directory",
            input_schema=Input,
            output_schema=Output,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "nested" / "path" / "skill.md"
            spec.export_skill_md(str(export_path))

            assert export_path.exists()
            assert "# nested-skill" in export_path.read_text()


class TestSkillSpecSimpleSkillExtensions:
    """Tests for SimpleSkill extension fields."""

    def test_skill_with_hooks(self) -> None:
        """Test SkillSpec with hooks."""
        from houyi.core.skill.hooks import HookEvent, SkillHook

        class Input(BaseModel):
            data: str

        class Output(BaseModel):
            result: str

        hooks = [
            SkillHook(event=HookEvent.PRE_TOOL_USE, matcher="Write"),
            SkillHook(event=HookEvent.POST_TOOL_USE),
        ]

        spec = SkillSpec(
            name="hooked-skill",
            description="Skill with hooks",
            input_schema=Input,
            output_schema=Output,
            hooks=hooks,
        )

        assert len(spec.hooks) == 2
        assert spec.hooks[0].event == HookEvent.PRE_TOOL_USE
        assert spec.hooks[0].matcher == "Write"

    def test_skill_with_version_and_allowed_tools(self) -> None:
        """Test SkillSpec with version and allowed_tools."""

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        spec = SkillSpec(
            name="versioned-skill",
            description="Skill with version",
            input_schema=Input,
            output_schema=Output,
            version="1.2.3",
            allowed_tools=["Read", "Write", "Edit"],
            user_invocable=False,
        )

        assert spec.version == "1.2.3"
        assert spec.allowed_tools == ["Read", "Write", "Edit"]
        assert spec.user_invocable is False
