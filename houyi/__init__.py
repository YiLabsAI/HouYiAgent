"""HouYi - Next-generation lightweight multi-agent framework with industry-leading capabilities."""

__version__ = "0.1.0"

# Core specifications
from houyi.core.agent import AgentSpec
from houyi.core.assertion import AssertionSpec
from houyi.core.skill import SkillSpec
from houyi.core.task import TaskSpec

# Decorators and utilities
from houyi.decorators import tool

# Evaluation
from houyi.evaluation import (
    EvaluationResult,
    EvaluationSummary,
    Evaluator,
    evaluate,
)

# Runtime classes
from houyi.runtime.agent import Agent
from houyi.runtime.task import Task
from houyi.runtime.team import Team, Workflow

# Alias for Skill (AgentSkills.io compatibility)
Skill = SkillSpec

__all__ = [
    # Specifications
    "AgentSpec",
    "SkillSpec",
    "TaskSpec",
    "AssertionSpec",
    # Runtime
    "Agent",
    "Task",
    "Team",
    "Workflow",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "EvaluationSummary",
    "evaluate",
    # Decorators
    "tool",
    # Aliases
    "Skill",
]
