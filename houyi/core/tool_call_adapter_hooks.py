"""Tool-calling adapter hooks for HouYi SDK."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from houyi.core.skill import SkillSpec
from houyi.llm.base import LLMResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallAdapterContext:
    """Context passed to tool-call adapter hooks."""

    skills: list[SkillSpec]
    tool_sequence: list[str]
    parallel_tool_calls: bool | None
    now: datetime
    adapter_name: str = "real"
    tool_names: list[str] = field(default_factory=list)


class ToolCallAdapterHook(Protocol):
    """Adapter hook signature."""

    def __call__(self, context: ToolCallAdapterContext) -> Any | None: ...


_ADAPTER_HOOKS: list[ToolCallAdapterHook] = []


def register_tool_call_adapter_hook(hook: ToolCallAdapterHook) -> None:
    """Register a tool-call adapter hook."""

    _ADAPTER_HOOKS.append(hook)


def resolve_tool_call_adapter(context: ToolCallAdapterContext) -> Any | None:
    """Resolve a tool-call adapter via registered hooks."""

    for hook in _ADAPTER_HOOKS:
        try:
            adapter = hook(context)
        except Exception as exc:
            logger.warning("Tool-call adapter hook failed: %s", exc)
            continue
        if adapter is not None:
            return adapter
    return None


class FakeToolCallAdapter:
    """Deterministic tool-calling adapter used by E2E tests."""

    def __init__(self, sequence: list[str], now: datetime | None = None) -> None:
        self._sequence = [name for name in sequence if name]
        self._index = 0
        self._now = now or datetime.now(timezone.utc)

    async def chat(
        self,
        _messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        if self._index < len(self._sequence):
            remaining = self._sequence[self._index :]
            allow_parallel = bool(_kwargs.get("parallel_tool_calls")) and len(remaining) > 1
            tool_batch = remaining if allow_parallel else [remaining[0]]
            tool_calls: list[dict[str, Any]] = []
            tomorrow = (self._now.date() + timedelta(days=1)).isoformat()

            for offset, tool_name in enumerate(tool_batch, start=1):
                call_index = self._index + offset
                args: dict[str, Any] = {}
                if tool_name == "get_date":
                    args = {"offset_days": "tomorrow"}
                elif tool_name in {"get_weather", "get_weather_live"}:
                    args = {
                        "lat": 39.9042,
                        "lon": 116.4074,
                        "date": tomorrow,
                    }

                tool_calls.append(
                    {
                        "id": f"fake_call_{call_index}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args),
                        },
                    }
                )

            self._index += len(tool_batch)
            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage={},
                model="fake-toolcall",
            )

        return LLMResponse(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
            usage={},
            model="fake-toolcall",
        )


def _fake_tool_call_adapter_hook(context: ToolCallAdapterContext) -> Any | None:
    if context.adapter_name != "fake":
        return None
    return FakeToolCallAdapter(context.tool_sequence, now=context.now)


def _openai_compat_tool_call_adapter_hook(context: ToolCallAdapterContext) -> Any | None:
    if context.adapter_name != "openai_compat":
        return None
    from houyi.llm.openai_compat_adapter import OpenAICompatibleAdapter

    return OpenAICompatibleAdapter()


def _siliconflow_tool_call_adapter_hook(context: ToolCallAdapterContext) -> Any | None:
    if context.adapter_name != "siliconflow":
        return None
    from houyi.llm.openai_compat_adapter import OpenAICompatibleAdapter

    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = (
        os.getenv("SILICONFLOW_MODEL")
        or os.getenv("HOUYI_TOOLCALL_MODEL")
        or "deepseek-ai/DeepSeek-V3"
    )
    return OpenAICompatibleAdapter(api_key=api_key, base_url=base_url, model=model)


def _deepseek_tool_call_adapter_hook(context: ToolCallAdapterContext) -> Any | None:
    if context.adapter_name != "deepseek":
        return None
    from houyi.llm.openai_compat_adapter import OpenAICompatibleAdapter

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = (
        os.getenv("DEEPSEEK_MODEL")
        or os.getenv("HOUYI_TOOLCALL_MODEL")
        or "deepseek-ai/DeepSeek-V3"
    )
    return OpenAICompatibleAdapter(api_key=api_key, base_url=base_url, model=model)


def _vertex_gemini_tool_call_adapter_hook(context: ToolCallAdapterContext) -> Any | None:
    if context.adapter_name != "vertex_gemini":
        return None
    from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

    return GoogleVertexGeminiAdapter.from_env()


register_tool_call_adapter_hook(_fake_tool_call_adapter_hook)
register_tool_call_adapter_hook(_openai_compat_tool_call_adapter_hook)
register_tool_call_adapter_hook(_siliconflow_tool_call_adapter_hook)
register_tool_call_adapter_hook(_deepseek_tool_call_adapter_hook)
register_tool_call_adapter_hook(_vertex_gemini_tool_call_adapter_hook)
