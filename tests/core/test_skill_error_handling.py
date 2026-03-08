"""Error-path tests for core/skill.py."""

import pytest
from pydantic import BaseModel, ValidationError

from houyi.domain.skill.spec import SkillSpec


class TestSkillErrorHandling:
    def test_skill_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SkillSpec.from_file("nonexistent_file.md")

    def test_skill_from_file_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            SkillSpec.from_file("/invalid/path/skill.md")

    def test_skill_missing_required_fields(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        with pytest.raises(ValidationError):
            SkillSpec(
                name="test",
                input_schema=Input,
                output_schema=Output,
            )

    def test_skill_executor_with_invalid_input(self):
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
            executor=func,
        )

        with pytest.raises(ValidationError):
            skill.input_schema(x="not_an_int")
