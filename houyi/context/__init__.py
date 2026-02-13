"""HouYi Context Engine.

Manages context window lifecycle for chat and agent scenarios:
- Token estimation (multi-model tokenizer adaptation)
- Context planning (block assembly, token budget allocation)
- Context rendering (ContextPlan → LLMMessage[])
- Context compression (Phase 2)

"""

from houyi.context.context_planner import ContextPlanner
from houyi.context.context_renderer import ContextRenderer
from houyi.context.token_estimator import TokenEstimator
from houyi.context.types import ContextBlock, ContextBlockType, ContextPlan, ContextUsage

__all__ = [
    "ContextBlock",
    "ContextBlockType",
    "ContextPlan",
    "ContextPlanner",
    "ContextRenderer",
    "ContextUsage",
    "TokenEstimator",
]
