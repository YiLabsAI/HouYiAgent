"""Deterministic tool-calling adapter for E2E tests.

This adapter replays a fixed tool-call sequence without making real LLM calls.
It is selected by the hook system when HOUYI_TOOLCALL_ADAPTER=fake.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from houyi.adapters.llm.base import LLMResponse


class FakeToolCallAdapter:
    """Deterministic tool-calling adapter used by E2E tests."""

    def __init__(self, sequence: list[str], now: datetime | None = None) -> None:
        self._sequence = [name for name in sequence if name]
        self._index = 0
        self._now = now or datetime.now(UTC)

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
                elif tool_name == "get_weather":
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
