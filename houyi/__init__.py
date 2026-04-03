"""HouYi - Next-generation lightweight multi-agent framework with industry-leading capabilities."""

__version__ = "0.3.0"

from houyi.application.runtime.agent import Agent
from houyi.application.runtime.task import Task
from houyi.application.runtime.team import Team, Workflow
from houyi.assurance.evaluation import (
    EvaluationResult,
    EvaluationSummary,
    Evaluator,
    evaluate,
)
from houyi.assurance.verification.assertion import AssertionSpec
from houyi.decorators import tool
from houyi.domain.agent import AgentSpec, AgentTeamConfig
from houyi.domain.skill.spec import SkillSpec
from houyi.domain.task import TaskSpec

Skill = SkillSpec

__all__ = [
    "Agent",
    "AgentSpec",
    "AgentTeamConfig",
    "AssertionSpec",
    "EvaluationResult",
    "EvaluationSummary",
    "Evaluator",
    "Skill",
    "SkillSpec",
    "Task",
    "TaskSpec",
    "Team",
    "Workflow",
    "__version__",
    "evaluate",
    "tool",
]
