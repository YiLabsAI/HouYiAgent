"""Helpers for LLM node execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_llm_node_inputs(
    *,
    config: dict[str, Any],
    run_settings: dict[str, Any] | None,
    resolve_tool_settings: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    prompt = config.get("prompt", "Hello, how can I help you?")
    system_prompt = config.get("system_prompt")
    user_prompt = config.get("user_prompt")
    model = config.get("model")

    max_tokens: Any = config.get("max_tokens")
    if isinstance(max_tokens, str):
        try:
            max_tokens = int(max_tokens)
        except ValueError:
            max_tokens = None

    enable_reasoning = config.get("enable_reasoning", False)
    thinking_budget = config.get("thinking_budget", 1024)

    effective_run_settings = run_settings or {}
    if effective_run_settings:
        tool_settings = effective_run_settings
    else:
        tool_settings = resolve_tool_settings(config)

    tool_names = tool_settings.get("tool_names") or []
    tool_choice = tool_settings.get("tool_choice")
    max_tool_calls = tool_settings.get("max_tool_calls", 6)
    temperature = tool_settings.get("temperature")
    parallel_tool_calls = tool_settings.get("parallel_tool_calls")
    enable_tool_calls = bool(tool_settings.get("enable_tool_calls") or tool_names)
    prompt_cache_key = tool_settings.get("prompt_cache_key")

    return {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model": model,
        "max_tokens": max_tokens,
        "enable_reasoning": enable_reasoning,
        "thinking_budget": thinking_budget,
        "enable_tool_calls": enable_tool_calls,
        "tool_names": tool_names,
        "tool_choice": tool_choice,
        "max_tool_calls": max_tool_calls,
        "temperature": temperature,
        "parallel_tool_calls": parallel_tool_calls,
        "prompt_cache_key": prompt_cache_key,
    }
