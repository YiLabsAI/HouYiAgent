"""Integration test for LLM + Tool scenario (weather + web search)."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from dotenv import load_dotenv

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY
from houyi.execution.skill_executor import SkillExecutor
from houyi.execution.tool_call_runner import ToolCallRunner
from houyi.llm.base import LLMResponse
from houyi.web_search.skill import build_web_search_skill


def _load_console_tools() -> None:
    script_path = Path(__file__).resolve().parent / "fixtures" / "console_tools.py"
    spec = spec_from_file_location("console_tools", script_path)
    if spec and spec.loader:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)


class ScenarioAdapter:
    def __init__(self, tomorrow: str) -> None:
        self._calls = [
            ("get_date", {"offset_days": "tomorrow"}),
            ("get_location", {"city": "Hangzhou Binjiang"}),
            (
                "get_weather",
                {"lat": 39.9042, "lon": 116.4074, "date": tomorrow},
            ),
            (
                "web_search",
                {
                    "query": "Agricultural Bank of China Binjiang branch phone and hours",
                    "max_results": 3,
                },
            ),
        ]
        self._index = 0

    async def chat(self, _messages, tools=None, **_kwargs):
        if self._index < len(self._calls):
            tool_name, args = self._calls[self._index]
            self._index += 1
            return LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{self._index}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(args)},
                    }
                ],
                finish_reason="tool_calls",
                usage={},
                model="scenario-adapter",
            )

        return LLMResponse(
            content="Done.", tool_calls=[], finish_reason="stop", usage={}, model="scenario-adapter"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "env_key", "requires_key", "base_url_key"),
    [
        ("ddg", "DDG_INTEGRATION_TEST", False, None),
        ("searxng", None, False, "SEARXNG_BASE_URL"),
        ("tavily", "TAVILY_API_KEY", True, None),
        ("serper", "SERPER_API_KEY", True, None),
    ],
)
async def test_llm_tool_scenario_weather_and_web_search(
    provider: str,
    env_key: str | None,
    requires_key: bool,
    base_url_key: str | None,
) -> None:
    """Run tool-call loop: weather + web search for bank inquiry."""

    load_dotenv()
    if os.getenv("HOUYI_RUN_LLM_TOOL_SCENARIO_INTEGRATION_TESTS") != "1":
        pytest.skip(
            "Network integration test disabled by default; set "
            "HOUYI_RUN_LLM_TOOL_SCENARIO_INTEGRATION_TESTS=1 to enable"
        )
    os.environ.setdefault("HOUYI_DISABLE_LIVE_WEATHER", "1")
    if requires_key:
        if not os.getenv(env_key):
            pytest.skip(f"{env_key} not set; web_search requires network access")
    else:
        if env_key and os.getenv(env_key) != "1":
            pytest.skip(f"{env_key} not enabled; set to 1 to run DDG integration test")
    if base_url_key:
        if not os.getenv(base_url_key):
            pytest.skip(f"{base_url_key} not set; searxng requires a running instance")
    os.environ["WEB_SEARCH_PROVIDER"] = provider

    _load_console_tools()
    DEFAULT_SKILL_REGISTRY.register(build_web_search_skill(), overwrite=True)

    tool_names = ["get_date", "get_location", "get_weather", "web_search"]
    skills = [s for name in tool_names if (s := DEFAULT_SKILL_REGISTRY.get(name)) is not None]
    tools = DEFAULT_SKILL_REGISTRY.to_tool_schemas()

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    adapter = ScenarioAdapter(tomorrow)
    executor = SkillExecutor(max_retries=1, timeout=30)
    runner = ToolCallRunner()

    response, trace = await runner.run(
        adapter=adapter,
        messages=[
            {
                "role": "user",
                "content": "I plan to visit the Agricultural Bank of China tomorrow to handle savings. "
                "What will the weather be like, and can you check if the branch will be open?",
            }
        ],
        tools=tools,
        skills=skills,
        executor=executor,
        max_rounds=6,
    )

    assert response.finish_reason in {"stop", "tool_calls"}
    assert len(trace) == 4

    weather_entry = trace[2]
    weather_raw = weather_entry["result"]["raw"]
    assert "result" in weather_raw

    search_entry = trace[3]
    search_raw = search_entry["result"]["raw"]
    search_payload = search_raw.get("result", search_raw)
    assert search_payload.get("provider") == provider, f"unexpected provider payload: {search_raw}"
    assert search_payload.get("results")
