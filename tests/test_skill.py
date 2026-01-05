"""Tests for Skill loading functionality (from_file, from_url, from_registry)."""

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from houyi.core.skill import SkillSpec


class TestSkillFromFile:
    """Test SkillSpec.from_file() method."""

    def test_from_file_basic(self):
        """Test loading a skill from a local file."""
        skill = SkillSpec.from_file("skills/web_search.md")

        assert skill.name == "web_search"
        assert skill.description is not None
        assert skill.input_schema is not None
        assert skill.output_schema is not None
        assert skill.executor is None  # Executor not bound yet
        assert skill.skill_md_path == "skills/web_search.md"

    def test_from_file_calculator(self):
        """Test loading calculator skill."""
        skill = SkillSpec.from_file("skills/calculator.md")

        # Verify it loaded successfully
        assert skill.name is not None
        assert skill.description is not None

    def test_skill_to_tool_schema(self):
        """Test converting skill to OpenAI tool schema."""
        from pydantic import BaseModel

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        skill = SkillSpec(
            name="search",
            description="Search tool",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        schema = skill.to_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search tool"
        assert "parameters" in schema["function"]

    def test_skill_with_constraints(self):
        """Test skill with constraints."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def processor(input: Input) -> Output:
            return Output(y=input.x * 2)

        skill = SkillSpec(
            name="processor",
            description="Process data",
            input_schema=Input,
            output_schema=Output,
            executor=processor,
            constraints={"timeout": 5000, "max_cost": 0.01}
        )

        assert skill.constraints["timeout"] == 5000
        assert skill.constraints["max_cost"] == 0.01

    def test_skill_execute_direct(self):
        """Test direct skill execution."""
        from pydantic import BaseModel

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            doubled: int

        def doubler(input: Input) -> Output:
            return Output(doubled=input.value * 2)

        skill = SkillSpec(
            name="doubler",
            description="Double a number",
            input_schema=Input,
            output_schema=Output,
            executor=doubler,
        )

        result = skill.executor(Input(value=5))
        assert result.doubled == 10
        assert skill.input_schema is not None
        assert skill.output_schema is not None

    def test_from_file_nonexistent(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            SkillSpec.from_file("skills/nonexistent.md")


# SkillRegistry tests removed - class doesn't exist in current implementation


class TestSkillExport:
    """Test skill export functionality."""

    def test_export_skill_md_basic(self, tmp_path):
        """Test exporting skill to markdown file."""
        from pydantic import BaseModel

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def executor(input: Input) -> Output:
            return Output(result="test")

        skill = SkillSpec(
            name="test_skill",
            description="A test skill",
            input_schema=Input,
            output_schema=Output,
            executor=executor
        )

        output_file = tmp_path / "skill.md"
        skill.export_skill_md(str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "# test_skill" in content
        assert "A test skill" in content

    def test_export_skill_md_with_metadata(self, tmp_path):
        """Test exporting skill with metadata."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def executor(input: Input) -> Output:
            return Output(y=input.x * 2)

        skill = SkillSpec(
            name="doubler",
            description="Double a number",
            input_schema=Input,
            output_schema=Output,
            executor=executor
        )

        output_file = tmp_path / "doubler.md"
        metadata = {"language": "Python", "runtime": "sync"}
        skill.export_skill_md(str(output_file), metadata=metadata)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Python" in content
        assert "runtime" in content.lower()

    def test_export_skill_md_with_examples(self, tmp_path):
        """Test exporting skill with examples."""
        from pydantic import BaseModel

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            doubled: int

        def executor(input: Input) -> Output:
            return Output(doubled=input.value * 2)

        skill = SkillSpec(
            name="doubler",
            description="Double",
            input_schema=Input,
            output_schema=Output,
            executor=executor
        )

        examples = [
            {"input": {"value": 5}, "output": {"doubled": 10}},
            {"input": {"value": 10}, "output": {"doubled": 20}}
        ]

        output_file = tmp_path / "skill.md"
        skill.export_skill_md(str(output_file), examples=examples)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Example" in content


class TestSkillConstraints:
    """Test skill constraints handling."""

    def test_skill_with_constraints(self):
        """Test creating skill with constraints."""
        from pydantic import BaseModel

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            result: int

        def executor(input: Input) -> Output:
            return Output(result=input.value * 2)

        constraints = {
            "max_tokens": 1000,
            "timeout": 30,
            "rate_limit": 10
        }

        skill = SkillSpec(
            name="constrained",
            description="Constrained skill",
            input_schema=Input,
            output_schema=Output,
            executor=executor,
            constraints=constraints
        )

        assert skill.constraints["max_tokens"] == 1000
        assert skill.constraints["timeout"] == 30
        assert skill.constraints["rate_limit"] == 10

    def test_export_with_constraints(self, tmp_path):
        """Test exporting skill with constraints."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def executor(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=executor,
            constraints={"max_tokens": 500}
        )

        output_file = tmp_path / "skill.md"
        skill.export_skill_md(str(output_file))

        content = output_file.read_text()
        assert "Constraints" in content
        assert "Max Tokens" in content


class TestSkillInvalidFormat:
    """Test invalid skill format handling."""

    def test_from_file_invalid_format(self, tmp_path):
        """Test loading invalid skill.md format."""
        invalid_file = tmp_path / "invalid.md"
        invalid_file.write_text("This is not a valid skill.md format")

        # Should still create a skill but with default values
        skill = SkillSpec.from_file(str(invalid_file))
        assert skill.name is not None  # Will have a default name


class TestSkillFromUrl:
    """Test SkillSpec.from_url() method."""

    @patch('urllib.request.urlopen')
    def test_from_url_success(self, mock_urlopen):
        """Test loading skill from URL successfully."""
        # Mock URL response
        skill_content = """# Test Skill

## Description
A test skill for testing

## Input Schema
```json
{"type": "object", "properties": {"query": {"type": "string"}}}
```

## Output Schema
```json
{"type": "object", "properties": {"result": {"type": "string"}}}
```
"""
        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        skill = SkillSpec.from_url("https://example.com/skill.md", cache=False)

        assert skill.name == "Test Skill"
        assert "test skill" in skill.description.lower()
        assert skill.input_schema is not None
        assert skill.output_schema is not None

    @patch('urllib.request.urlopen')
    def test_from_url_with_cache(self, mock_urlopen, tmp_path):
        """Test loading skill from URL with caching."""
        skill_content = """# Cached Skill

## Description
A cached skill

## Input Schema
```json
{"type": "object"}
```

## Output Schema
```json
{"type": "object"}
```
"""
        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        with patch('pathlib.Path.home', return_value=tmp_path):
            skill = SkillSpec.from_url("https://example.com/skill.md", cache=True)

            assert skill.name == "Cached Skill"
            # Cache directory should be created
            cache_dir = tmp_path / ".houyi" / "skill_cache"
            assert cache_dir.exists()

    @patch('urllib.request.urlopen')
    def test_from_url_network_error(self, mock_urlopen):
        """Test handling network errors when loading from URL."""
        mock_urlopen.side_effect = urllib.error.URLError("Network error")

        with pytest.raises(urllib.error.URLError) as exc_info:
            SkillSpec.from_url("https://example.com/skill.md")

        assert "Network error" in str(exc_info.value)

    @patch('urllib.request.urlopen')
    def test_from_url_invalid_content(self, mock_urlopen):
        """Test handling invalid content from URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"Invalid JSON content {{{{"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        # Should handle parsing errors gracefully
        skill = SkillSpec.from_url("https://example.com/skill.md", cache=False)
        assert skill.name is not None  # Will have default values


class TestSkillFromRegistry:
    """Test SkillSpec.from_registry() method."""

    @patch('urllib.request.urlopen')
    def test_from_registry_basic(self, mock_urlopen):
        """Test loading skill from AgentSkills.io registry."""
        skill_content = """# Web Search

## Description
Search the web

## Input Schema
```json
{"type": "object", "properties": {"query": {"type": "string"}}}
```

## Output Schema
```json
{"type": "object", "properties": {"results": {"type": "array"}}}
```
"""
        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        skill = SkillSpec.from_registry("web_search", cache=False)

        assert skill.name == "Web Search"
        assert "search" in skill.description.lower()
        # Verify URL was called
        mock_urlopen.assert_called_once()


class TestSkillBindExecutor:
    """Test SkillSpec.bind_executor() method."""

    def test_bind_executor_basic(self):
        """Test binding executor to skill."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x * 2)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=None,
        )

        assert skill.executor is None

        skill.bind_executor(func)

        assert skill.executor is not None
        assert skill.executor == func

        # Test execution after binding
        result = skill.executor(Input(x=5))
        assert result.y == 10

    def test_bind_executor_replace(self):
        """Test replacing existing executor."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func1(input: Input) -> Output:
            return Output(y=input.x * 2)

        def func2(input: Input) -> Output:
            return Output(y=input.x * 3)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func1,
        )

        assert skill.executor == func1

        skill.bind_executor(func2)

        assert skill.executor == func2

        # Test new executor works
        result = skill.executor(Input(x=5))
        assert result.y == 15


class TestSkillConstraints:
    """Test skill constraints."""

    def test_constraints_empty_default(self):
        """Test default empty constraints."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        assert skill.constraints == {}

    def test_constraints_with_values(self):
        """Test constraints with values."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
            constraints={"timeout": 10, "max_retries": 3, "cost_limit": 0.5}
        )

        assert skill.constraints["timeout"] == 10
        assert skill.constraints["max_retries"] == 3
        assert skill.constraints["cost_limit"] == 0.5

    def test_constraints_modification(self):
        """Test modifying constraints."""
        from pydantic import BaseModel

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
            constraints={"timeout": 10}
        )

        # Modify constraints
        skill.constraints["timeout"] = 20
        skill.constraints["max_cost"] = 1.0

        assert skill.constraints["timeout"] == 20
        assert skill.constraints["max_cost"] == 1.0

    @patch('urllib.request.urlopen')
    def test_from_registry_with_version(self, mock_urlopen):
        """Test loading specific version from registry."""
        skill_content = """# Test Skill

## Description
Test

## Input Schema
```json
{"type": "object"}
```

## Output Schema
```json
{"type": "object"}
```
"""
        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        skill = SkillSpec.from_registry("test_skill", version="v1.0.0", cache=False)

        assert skill.name == "Test Skill"
        # Verify version in URL
        called_url = mock_urlopen.call_args[0][0]
        assert "v1.0.0" in called_url


class TestSkillParsing:
    """Test skill.md parsing functionality."""

    def test_parse_skill_md_complete(self):
        """Test parsing complete skill.md format."""
        content = """# Complete Skill

## Description
A complete skill with all fields

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
        parsed = SkillSpec._parse_skill_md(content)

        assert parsed["name"] == "Complete Skill"
        assert "complete skill" in parsed["description"].lower()
        assert "input_schema" in parsed
        assert "output_schema" in parsed

    def test_parse_skill_md_minimal(self):
        """Test parsing minimal skill.md format."""
        content = """# Minimal Skill

## Description
Minimal description
"""
        parsed = SkillSpec._parse_skill_md(content)

        assert parsed["name"] == "Minimal Skill"
        assert parsed["description"] == "Minimal description"
        # Should handle missing schemas gracefully
        assert "input_schema" not in parsed or parsed.get("input_schema") is None
