"""Runtime contract for ecosystem skill integration.

Defines the runtime declaration that ecosystem skills can include in their
SKILL.md frontmatter to enable automatic executor binding and capability
reporting:

- adapter field resolves to a Python callable for executor binding
- CapabilityTier reports discovery/validation/execution capability
- RuntimeStatus reports operational readiness (ready/degraded/unavailable)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RuntimeMode(str, Enum):
    """How the skill is dispatched at runtime."""

    TOOL = "tool"
    SCRIPT = "script"
    TEMPLATE = "template"


class CapabilityTier(int, Enum):
    """Capability tier for a loaded skill."""

    METADATA = 1
    SCHEMA = 2
    EXECUTABLE = 3


class RuntimeStatus(str, Enum):
    """Runtime health for a loaded skill."""

    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class RuntimeContract(BaseModel):
    """Parsed runtime block from SKILL.md frontmatter."""

    mode: RuntimeMode = Field(
        default=RuntimeMode.TOOL,
        description="Dispatch mode: tool | script | template",
    )
    entry: str | None = Field(
        default=None,
        description="Entry point name (for tool mode, the registered tool name)",
    )
    adapter: str | None = Field(
        default=None,
        description=(
            "Python dotted path to executor callable, e.g. "
            "'houyi.skills.planning.adapter:execute'. "
            "Format: 'module.path:function_name'."
        ),
    )
    hooks_root: str | None = Field(
        default=None,
        description="Relative path to skill assets (hooks, scripts, templates)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Unknown fields preserved for forward compatibility",
    )

    @classmethod
    def from_dict(cls, data: Any) -> RuntimeContract | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            logger.debug(
                "runtime field is not a dict (%s), ignoring",
                type(data).__name__,
            )
            return None

        known_keys = {"mode", "entry", "adapter", "hooks_root"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        raw_mode = data.get("mode", "tool")
        try:
            mode = RuntimeMode(raw_mode)
        except ValueError:
            logger.warning(
                "Unknown runtime mode '%s', falling back to 'tool'",
                raw_mode,
            )
            mode = RuntimeMode.TOOL
            extra["original_mode"] = str(raw_mode)

        return cls(
            mode=mode,
            entry=data.get("entry"),
            adapter=data.get("adapter"),
            hooks_root=data.get("hooks_root"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode.value}
        if self.entry is not None:
            result["entry"] = self.entry
        if self.adapter is not None:
            result["adapter"] = self.adapter
        if self.hooks_root is not None:
            result["hooks_root"] = self.hooks_root
        result.update(self.extra)
        return result
