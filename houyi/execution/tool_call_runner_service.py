from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY
from houyi.execution.skill_executor import SkillExecutor


@dataclass(frozen=True)
class ToolCallRunnerResult:
    response: Any
    trace: list[dict[str, Any]]


class ToolCallRunnerService:
    """Run a simple tool-call loop against a chat adapter.

    This module exists primarily to satisfy imports for integration tests.
    The integration tests are disabled by default, but the import must succeed.
    """

    def select_skills(self, tool_names: list[str]) -> list[Any]:
        selected: list[Any] = []
        for name in tool_names:
            skill = DEFAULT_SKILL_REGISTRY.get(name)
            if skill is not None:
                selected.append(skill)
        return selected

    async def run_tool_calls(
        self,
        *,
        adapter: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        skills: list[Any],
        executor: SkillExecutor,
        max_rounds: int = 6,
    ) -> tuple[Any, list[dict[str, Any]]]:
        skill_by_name: dict[str, Any] = {getattr(skill, "name", ""): skill for skill in skills}
        trace: list[dict[str, Any]] = []

        current_messages = list(messages)
        last_response: Any = None

        for _round in range(max_rounds):
            last_response = await adapter.chat(current_messages, tools=tools)

            tool_calls = getattr(last_response, "tool_calls", None)
            finish_reason = getattr(last_response, "finish_reason", None)

            if not tool_calls:
                break

            for call in tool_calls:
                function_payload = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function_payload, dict):
                    continue

                tool_name = function_payload.get("name")
                raw_args = function_payload.get("arguments")
                if not isinstance(tool_name, str) or not tool_name:
                    continue

                args: dict[str, Any]
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                skill = skill_by_name.get(tool_name) or DEFAULT_SKILL_REGISTRY.get(tool_name)
                if skill is None:
                    trace.append(
                        {
                            "tool_name": tool_name,
                            "tool_call_id": call.get("id") if isinstance(call, dict) else None,
                            "args": args,
                            "result": {
                                "raw": {"error": f"unknown tool: {tool_name}"},
                                "is_error": True,
                            },
                        }
                    )
                    continue

                try:
                    result = await executor.execute(skill=skill, input_data=args)
                    trace.append(
                        {
                            "tool_name": tool_name,
                            "tool_call_id": call.get("id") if isinstance(call, dict) else None,
                            "args": args,
                            "result": {"raw": {"result": result}, "is_error": False},
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    trace.append(
                        {
                            "tool_name": tool_name,
                            "tool_call_id": call.get("id") if isinstance(call, dict) else None,
                            "args": args,
                            "result": {"raw": {"error": str(exc)}, "is_error": True},
                        }
                    )

            if finish_reason == "stop":
                break

        return last_response, trace
