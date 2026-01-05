"""Advanced tests for SkillSpec class."""

import pytest
from pydantic import BaseModel, ValidationError

from houyi.core.skill import SkillSpec


class TestSkillSpecToolSchema:
    """Test SkillSpec tool schema generation."""

    def test_to_tool_schema_basic(self):
        """Test basic tool schema generation."""
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
        """Test tool schema with optional input fields."""
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
        """Test tool schema with complex nested types."""
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


class TestSkillSpecConstraints:
    """Test SkillSpec constraints handling."""

    def test_constraints_timeout(self):
        """Test skill with timeout constraint."""
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
            constraints={"timeout_ms": 5000}
        )
        
        assert skill.constraints["timeout_ms"] == 5000

    def test_constraints_max_retries(self):
        """Test skill with max_retries constraint."""
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
            constraints={"max_retries": 3}
        )
        
        assert skill.constraints["max_retries"] == 3

    def test_constraints_max_cost(self):
        """Test skill with max_cost constraint."""
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
            constraints={"max_cost": 0.01}
        )
        
        assert skill.constraints["max_cost"] == 0.01

    def test_constraints_multiple(self):
        """Test skill with multiple constraints."""
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
            "cache_enabled": True
        }
        
        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
            constraints=constraints
        )
        
        assert len(skill.constraints) == 4
        assert skill.constraints["cache_enabled"] is True


class TestSkillSpecExecution:
    """Test SkillSpec execution scenarios."""

    def test_execute_simple_function(self):
        """Test executing simple skill function."""
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
        """Test executing skill with string processing."""
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
                length=len(input.text)
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
        """Test executing skill with list processing."""
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
                min_val=min(nums)
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
        """Test executing skill with dict processing."""
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
                count=len(input.data)
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


class TestSkillSpecValidation:
    """Test SkillSpec validation scenarios."""

    def test_input_validation_success(self):
        """Test successful input validation."""
        class Input(BaseModel):
            email: str
            age: int
        
        class Output(BaseModel):
            valid: bool
        
        def validate(input: Input) -> Output:
            return Output(valid=True)
        
        skill = SkillSpec(
            name="validate",
            description="Validate input",
            input_schema=Input,
            output_schema=Output,
            executor=validate,
        )
        
        # Valid input
        valid_input = Input(email="test@example.com", age=25)
        assert valid_input.email == "test@example.com"

    def test_input_validation_failure(self):
        """Test input validation failure."""
        class Input(BaseModel):
            age: int
        
        class Output(BaseModel):
            result: int
        
        def process(input: Input) -> Output:
            return Output(result=input.age)
        
        skill = SkillSpec(
            name="process",
            description="Process age",
            input_schema=Input,
            output_schema=Output,
            executor=process,
        )
        
        # Invalid input type
        with pytest.raises(ValidationError):
            Input(age="not_a_number")

    def test_output_validation_success(self):
        """Test successful output validation."""
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
        """Test output validation failure."""
        class Input(BaseModel):
            x: int
        
        class Output(BaseModel):
            result: int
        
        skill = SkillSpec(
            name="process",
            description="Process",
            input_schema=Input,
            output_schema=Output,
            executor=None,
        )
        
        # Invalid output
        with pytest.raises(ValidationError):
            Output(result="not_a_number")


class TestSkillSpecEdgeCases:
    """Test SkillSpec edge cases."""

    def test_skill_with_empty_description(self):
        """Test skill with empty description."""
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
        """Test skill with very long description."""
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
        """Test skill with unicode characters in name."""
        class Input(BaseModel):
            x: int
        
        class Output(BaseModel):
            y: int
        
        def func(input: Input) -> Output:
            return Output(y=input.x)
        
        skill = SkillSpec(
            name="测试技能",
            description="Test skill",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )
        
        assert skill.name == "测试技能"

    def test_skill_without_executor(self):
        """Test skill without executor function."""
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
