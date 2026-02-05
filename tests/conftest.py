"""Shared test fixtures and utilities."""

import pytest
import pytest_asyncio
from pydantic import BaseModel

from houyi.core.skill import SkillSpec


@pytest_asyncio.fixture
async def graph_store(tmp_path):
    """Loaded GraphStore backed by a temp directory.

    Ensures SQLite handles are closed even on Windows.
    """
    from houyi.rag.indexed.graph.store import GraphStore

    async with GraphStore(knowledge_dir=str(tmp_path)) as store:
        yield store


# Common test schemas
class NumberInput(BaseModel):
    """Input schema for number operations."""

    value: int


class NumberOutput(BaseModel):
    """Output schema for number operations."""

    result: int


class TwoNumberInput(BaseModel):
    """Input schema for two-number operations."""

    a: int
    b: int


class TwoNumberOutput(BaseModel):
    """Output schema for two-number operations."""

    result: int


@pytest.fixture
def doubler_skill():
    """Create a doubler skill for testing."""
    return SkillSpec(
        name="doubler",
        description="Double a number",
        input_schema=NumberInput,
        output_schema=NumberOutput,
    )


@pytest.fixture
def doubler_skill_simple():
    """Create a simple doubler skill for testing."""
    return SkillSpec(
        name="doubler",
        description="Double",
        input_schema=NumberInput,
        output_schema=NumberOutput,
    )


@pytest.fixture
def add_skill():
    """Create an add skill for testing."""
    return SkillSpec(
        name="add",
        description="Add two numbers",
        input_schema=TwoNumberInput,
        output_schema=TwoNumberOutput,
    )


@pytest.fixture
def multiply_skill():
    """Create a multiply skill for testing."""
    return SkillSpec(
        name="multiply",
        description="Multiply two numbers",
        input_schema=TwoNumberInput,
        output_schema=TwoNumberOutput,
    )
