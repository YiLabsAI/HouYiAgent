"""Tests for Skill loading functionality (from_file, from_url, from_registry)."""

import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from houyi.core.skill import SkillSpec


class TestSkillFromFile:
    """Test SkillSpec.from_file() method."""

    def test_from_file_basic(self):
        """Test loading a skill from a local SKILL.md file."""
        skill = SkillSpec.from_file("skills/planning-with-files/SKILL.md")

        assert skill.name == "planning-with-files"
        assert skill.description is not None
        assert skill.input_schema is not None
        assert skill.output_schema is not None
        assert skill.executor is None  # Executor not bound yet
        assert skill.skill_md_path == "skills/planning-with-files/SKILL.md"

    def test_from_file_calculator(self):
        """Test loading community skill."""
        skill = SkillSpec.from_file("skills/skill-creator/SKILL.md")

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
            constraints={"timeout": 5000, "max_cost": 0.01},
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
            executor=executor,
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
            executor=executor,
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
            executor=executor,
        )

        examples = [
            {"input": {"value": 5}, "output": {"doubled": 10}},
            {"input": {"value": 10}, "output": {"doubled": 20}},
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

        constraints = {"max_tokens": 1000, "timeout": 30, "rate_limit": 10}

        skill = SkillSpec(
            name="constrained",
            description="Constrained skill",
            input_schema=Input,
            output_schema=Output,
            executor=executor,
            constraints=constraints,
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
            constraints={"max_tokens": 500},
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

    @patch("urllib.request.urlopen")
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
        mock_response.read.return_value = skill_content.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        skill = SkillSpec.from_url("https://example.com/skill.md", cache=False)

        assert skill.name == "Test Skill"
        assert "test skill" in skill.description.lower()
        assert skill.input_schema is not None
        assert skill.output_schema is not None

    @patch("urllib.request.urlopen")
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
        mock_response.read.return_value = skill_content.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        with patch("pathlib.Path.home", return_value=tmp_path):
            skill = SkillSpec.from_url("https://example.com/skill.md", cache=True)

            assert skill.name == "Cached Skill"
            # Cache directory should be created
            cache_dir = tmp_path / ".houyi" / "skill_cache"
            assert cache_dir.exists()

    @patch("urllib.request.urlopen")
    def test_from_url_network_error(self, mock_urlopen):
        """Test handling network errors when loading from URL."""
        mock_urlopen.side_effect = urllib.error.URLError("Network error")

        with pytest.raises(urllib.error.URLError) as exc_info:
            SkillSpec.from_url("https://example.com/skill.md")

        assert "Network error" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_from_url_invalid_content(self, mock_urlopen):
        """Test handling invalid content from URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"Invalid JSON content {{{{"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        # Should handle parsing errors gracefully
        skill = SkillSpec.from_url("https://example.com/skill.md", cache=False)
        assert skill.name is not None  # Will have a default name


class TestSkillFromRegistry:
    """Test SkillSpec.from_registry() method."""

    @patch("urllib.request.urlopen")
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
``}
"""
        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        skill = SkillSpec.from_registry(
            "web_search",
            cache=False,
            base_url="https://skills.example.com",
        )

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
            constraints={"timeout": 10, "max_retries": 3, "cost_limit": 0.5},
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
            constraints={"timeout": 10},
        )

        # Modify constraints
        skill.constraints["timeout"] = 20
        skill.constraints["max_cost"] = 1.0

        assert skill.constraints["timeout"] == 20
        assert skill.constraints["max_cost"] == 1.0

    @patch("urllib.request.urlopen")
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
``}
"""
        mock_response = MagicMock()
        mock_response.read.return_value = skill_content.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        skill = SkillSpec.from_registry(
            "test_skill",
            version="v1.0.0",
            cache=False,
            base_url="https://skills.example.com",
        )

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


class AdvancedTestSkillSpecToolSchema:
    """Advanced SkillSpec tool schema generation tests."""

    def test_to_tool_schema_basic(self):
        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="found")

        skill = SkillSpec(
            name="search",
            description="Search function",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        schema = skill.to_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search function"
        assert "parameters" in schema["function"]

    def test_to_tool_schema_with_optional_fields(self):
        class Input(BaseModel):
            query: str
            limit: int = 10
            offset: int = 0

        class Output(BaseModel):
            results: list[str]

        def search(input: Input) -> Output:
            return Output(results=[])

        skill = SkillSpec(
            name="search",
            description="Advanced search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        schema = skill.to_tool_schema()

        assert "parameters" in schema["function"]
        assert schema["function"]["name"] == "search"

    def test_to_tool_schema_complex_types(self):
        class Address(BaseModel):
            street: str
            city: str
            country: str

        class Input(BaseModel):
            name: str
            addresses: list[Address]
            metadata: dict

        class Output(BaseModel):
            success: bool
            message: str

        def process(input: Input) -> Output:
            return Output(success=True, message="Done")

        skill = SkillSpec(
            name="process_data",
            description="Process complex data",
            input_schema=Input,
            output_schema=Output,
            executor=process,
        )

        schema = skill.to_tool_schema()
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]


class AdvancedTestSkillSpecConstraints:
    """Advanced SkillSpec constraints tests."""

    def test_constraints_timeout(self):
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
            constraints={"timeout_ms": 5000},
        )

        assert skill.constraints["timeout_ms"] == 5000

    def test_constraints_max_retries(self):
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
            constraints={"max_retries": 3},
        )

        assert skill.constraints["max_retries"] == 3

    def test_constraints_max_cost(self):
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
            constraints={"max_cost": 0.01},
        )

        assert skill.constraints["max_cost"] == 0.01

    def test_constraints_multiple(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        constraints = {
            "timeout_ms": 5000,
            "max_retries": 3,
            "max_cost": 0.01,
            "cache_enabled": True,
        }

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
            constraints=constraints,
        )

        assert len(skill.constraints) == 4
        assert skill.constraints["cache_enabled"] is True


class AdvancedTestSkillSpecExecution:
    """Advanced SkillSpec execution scenario tests."""

    def test_execute_simple_function(self):
        class Input(BaseModel):
            x: int
            y: int

        class Output(BaseModel):
            sum: int

        def add(input: Input) -> Output:
            return Output(sum=input.x + input.y)

        skill = SkillSpec(
            name="add",
            description="Add two numbers",
            input_schema=Input,
            output_schema=Output,
            executor=add,
        )

        result = skill.executor(Input(x=5, y=3))
        assert result.sum == 8

    def test_execute_with_string_processing(self):
        class Input(BaseModel):
            text: str

        class Output(BaseModel):
            upper: str
            lower: str
            length: int

        def process_text(input: Input) -> Output:
            return Output(
                upper=input.text.upper(),
                lower=input.text.lower(),
                length=len(input.text),
            )

        skill = SkillSpec(
            name="process",
            description="Process text",
            input_schema=Input,
            output_schema=Output,
            executor=process_text,
        )

        result = skill.executor(Input(text="Hello"))
        assert result.upper == "HELLO"
        assert result.lower == "hello"
        assert result.length == 5

    def test_execute_with_list_processing(self):
        class Input(BaseModel):
            numbers: list[int]

        class Output(BaseModel):
            total: int
            average: float
            max_val: int
            min_val: int

        def analyze_numbers(input: Input) -> Output:
            nums = input.numbers
            return Output(
                total=sum(nums),
                average=sum(nums) / len(nums),
                max_val=max(nums),
                min_val=min(nums),
            )

        skill = SkillSpec(
            name="analyze",
            description="Analyze numbers",
            input_schema=Input,
            output_schema=Output,
            executor=analyze_numbers,
        )

        result = skill.executor(Input(numbers=[1, 2, 3, 4, 5]))
        assert result.total == 15
        assert result.average == 3.0
        assert result.max_val == 5
        assert result.min_val == 1

    def test_execute_with_dict_processing(self):
        class Input(BaseModel):
            data: dict

        class Output(BaseModel):
            keys: list[str]
            values: list
            count: int

        def process_dict(input: Input) -> Output:
            return Output(
                keys=list(input.data.keys()),
                values=list(input.data.values()),
                count=len(input.data),
            )

        skill = SkillSpec(
            name="process",
            description="Process dict",
            input_schema=Input,
            output_schema=Output,
            executor=process_dict,
        )

        result = skill.executor(Input(data={"a": 1, "b": 2, "c": 3}))
        assert result.count == 3
        assert "a" in result.keys


class AdvancedTestSkillSpecValidation:
    """Advanced SkillSpec validation tests."""

    def test_input_validation_success(self):
        class Input(BaseModel):
            email: str
            age: int

        class Output(BaseModel):
            valid: bool

        def validate(input: Input) -> Output:
            return Output(valid=True)

        valid_input = Input(email="test@example.com", age=25)
        assert valid_input.email == "test@example.com"

        skill = SkillSpec(
            name="validate",
            description="Validate input",
            input_schema=Input,
            output_schema=Output,
            executor=validate,
        )
        assert skill is not None

    def test_input_validation_failure(self):
        class Input(BaseModel):
            age: int

        with pytest.raises(ValidationError):
            Input(age="not_a_number")

    def test_output_validation_success(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            result: int
            message: str

        def process(input: Input) -> Output:
            return Output(result=input.x * 2, message="success")

        skill = SkillSpec(
            name="process",
            description="Process",
            input_schema=Input,
            output_schema=Output,
            executor=process,
        )

        result = skill.executor(Input(x=5))
        assert result.result == 10
        assert result.message == "success"

    def test_output_validation_failure(self):
        class Output(BaseModel):
            result: int

        with pytest.raises(ValidationError):
            Output(result="not_a_number")


class AdvancedTestSkillSpecEdgeCases:
    """Advanced SkillSpec edge case tests."""

    def test_skill_with_empty_description(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        assert skill.description == ""

    def test_skill_with_long_description(self):
        long_desc = "A" * 1000

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description=long_desc,
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        assert len(skill.description) == 1000

    def test_skill_with_unicode_name(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test_skill_unicode",
            description="Test skill",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        assert skill.name == "test_skill_unicode"

    def test_skill_without_executor(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=None,
        )

        assert skill.executor is None
