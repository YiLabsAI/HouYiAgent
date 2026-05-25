"""Tool-calling adapter hooks for HouYi SDK.

Adapter resolution follows a two-stage strategy:

1. **Custom hooks** — user-registered hooks checked first (chain-of-responsibility).
2. **Factory fallback** — if no custom hook matches, delegates to
   LLMAdapterFactory.create(adapter_name) which is the single source of
   truth for adapter construction.

The only built-in hook is the fake-adapter hook which returns the deterministic
FakeToolCallAdapter (from houyi.testkit) for E2E testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from houyi.adapters.llm.models import ADAPTER_FAKE, ADAPTER_REAL
from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallAdapterContext:
    """Context passed to tool-call adapter hooks."""

    skills: list[SkillSpec]
    tool_sequence: list[str]
    parallel_tool_calls: bool | None
    now: datetime
    adapter_name: str = ADAPTER_REAL
    tool_names: list[str] = field(default_factory=list)


class ToolCallAdapterHook(Protocol):
    """Adapter hook signature."""

    def __call__(self, context: ToolCallAdapterContext) -> Any | None: ...


_ADAPTER_HOOKS: list[ToolCallAdapterHook] = []


def register_tool_call_adapter_hook(hook: ToolCallAdapterHook) -> None:
    """Register a tool-call adapter hook."""

    _ADAPTER_HOOKS.append(hook)


def resolve_tool_call_adapter(context: ToolCallAdapterContext) -> Any | None:
    """Resolve a tool-call adapter via registered hooks, then factory fallback.

    Resolution order:
    1. Iterate custom hooks — first non-None result wins.
    2. If adapter_name is "real" (the default), return None to let
       the caller use its own fallback.
    3. Otherwise delegate to LLMAdapterFactory.create(adapter_name).
    """

    for hook in _ADAPTER_HOOKS:
        try:
            adapter = hook(context)
        except Exception as exc:
            logger.warning("Tool-call adapter hook failed: %s", exc)
            continue
        if adapter is not None:
            return adapter

    if context.adapter_name == ADAPTER_REAL:
        return None

    try:
        from houyi.adapters.llm.factory import LLMAdapterFactory

        return LLMAdapterFactory.create(context.adapter_name)
    except Exception as exc:
        logger.warning(
            "LLMAdapterFactory.create(%s) failed: %s — returning None",
            context.adapter_name,
            exc,
        )
        return None


def _fake_tool_call_adapter_hook(context: ToolCallAdapterContext) -> Any | None:
    if context.adapter_name != ADAPTER_FAKE:
        return None
    from houyi.testkit.fake_adapter import FakeToolCallAdapter

    return FakeToolCallAdapter(context.tool_sequence, now=context.now)


register_tool_call_adapter_hook(_fake_tool_call_adapter_hook)
