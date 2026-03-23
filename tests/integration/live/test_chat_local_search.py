"""ChatService local search scenarios."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from dotenv import load_dotenv
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import (
    CreateConversationRequest,
    SendMessageRequest,
)

from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.skills.builtin.local_tools import register_builtin_local_tools

_TYPED_LANE_SKILLS = [
    "houyi_list_dir",
    "houyi_find_files",
    "houyi_grep",
    "houyi_read_file",
]

_CLI_LANE_SKILLS = ["houyi_local_cli"]

_SYSTEM_INSTRUCTIONS = (
    "You are evaluating local workspace search behavior. "
    "You must use the available local tools before answering. "
    "Do not guess paths, file contents, or workspace structure. "
    "A response without at least one tool call is invalid."
)


@dataclass(frozen=True)
class _Case:
    name: str
    prompt: str
    expected_terms: tuple[str, ...] = ()
    require_tool_error: bool = False


@dataclass(frozen=True)
class _Lane:
    name: str
    skills: list[str]


@dataclass(frozen=True)
class _Run:
    completion_metadata: dict[str, object]
    tool_start_events: list[dict[str, object]]
    tool_result_events: list[dict[str, object]]
    tool_error_events: list[dict[str, object]]
    iteration_events: list[dict[str, object]]
    final_text: str


_CASES = [
    _Case(
        name="search",
        prompt=(
            "Use local tools to inspect the current workspace and find the most relevant skill definition file for web search. "
            "You must make at least one local tool call before answering. "
            "Do not guess or invent paths or file contents. "
            "Return the relative file path, one short reason, and a preview of the first 20 lines. "
            "If the first path or parameters are wrong, correct them and continue."
        ),
        expected_terms=("web_search", "skill.md"),
    ),
    _Case(
        name="repair",
        prompt=(
            "Use local tools to open houyi/skills/websearch/SKILL.md and preview the first 20 lines. "
            "You must make at least one local tool call before answering. "
            "Do not invent a correction without checking the workspace. "
            "If that path does not exist, locate the correct file in the workspace, then continue and explain the correction."
        ),
        expected_terms=("web_search", "skill.md"),
    ),
    _Case(
        name="compact",
        prompt=(
            "Use local tools to find skill definition files related to search in the current workspace. "
            "You must make at least one local tool call before answering. "
            "First narrow the candidates, then return the single best match with a short reason and a preview of the first 20 lines. "
            "Do not guess candidate files without checking the workspace."
        ),
        expected_terms=("search", "skill"),
    ),
]

_LANES = [
    _Lane(name="typed", skills=_TYPED_LANE_SKILLS),
    _Lane(name="cli", skills=_CLI_LANE_SKILLS),
]


def _parse_sse_events(raw_chunks: list[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for chunk in raw_chunks:
        lines = chunk.strip().split("\n")
        event: dict[str, object] = {}
        for line in lines:
            if line.startswith("id: "):
                event["id"] = line[4:]
            elif line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = json.loads(line[6:])
        if "event" in event:
            events.append(event)
    return events


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_local_tool_registry() -> None:
    required = set(_TYPED_LANE_SKILLS + _CLI_LANE_SKILLS)
    missing = [name for name in required if DEFAULT_SKILL_REGISTRY.get(name) is None]
    if not missing:
        return
    register_builtin_local_tools(DEFAULT_SKILL_REGISTRY)


def _live_model() -> str | None:
    load_dotenv()
    if os.getenv("HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS") != "1":
        pytest.skip(
            "Live local-search test disabled by default; set HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 to enable"
        )
    model = (os.getenv("HOUYI_LIVE_E2E_MODEL") or "").strip()
    return model or None


async def _run_live_case(
    *,
    tmp_path: Path,
    lane: str,
    enabled_skills: list[str],
    model: str | None,
    prompt: str,
) -> _Run:
    store = JsonStore(data_dir=tmp_path / f"chat-store-{lane}")
    service = ChatService(json_store=store, default_model=model or "")
    conversation = service.create_conversation(
        CreateConversationRequest(
            title=f"live-{lane}",
            model=model or "",
            system_instructions=_SYSTEM_INSTRUCTIONS,
        )
    )

    raw_chunks: list[str] = []
    async for chunk in service.send_message(
        conversation["conversation_id"],
        SendMessageRequest(
            content=prompt,
            model=model,
            enable_tool_calls=True,
            tool_call_strategy="aggressive",
            enable_skills=enabled_skills,
            max_tool_iterations=8,
        ),
    ):
        raw_chunks.append(chunk)

    events = _parse_sse_events(raw_chunks)
    complete_event = next(
        event for event in reversed(events) if event.get("event") == "message.complete"
    )
    completion_metadata = dict(complete_event.get("data", {}).get("metadata", {}))
    tool_start_events = [event for event in events if event.get("event") == "tool_call.start"]
    tool_result_events = [event for event in events if event.get("event") == "tool_call.result"]
    tool_error_events = [event for event in events if event.get("event") == "tool_call.error"]
    iteration_events = [event for event in events if event.get("event") == "agent.iteration"]
    final_text = "".join(
        str(event.get("data", {}).get("content") or "")
        for event in events
        if event.get("event") == "message.delta"
    ).strip()

    return _Run(
        completion_metadata=completion_metadata,
        tool_start_events=tool_start_events,
        tool_result_events=tool_result_events,
        tool_error_events=tool_error_events,
        iteration_events=iteration_events,
        final_text=final_text,
    )


def _assert_lane(run: _Run, lane: _Lane) -> None:
    seen_tool_names = {
        str(event.get("data", {}).get("tool_name") or "") for event in run.tool_start_events
    }
    if not seen_tool_names:
        return
    if lane.name == "typed":
        assert seen_tool_names <= set(_TYPED_LANE_SKILLS)
    else:
        assert seen_tool_names == {"houyi_local_cli"}
    for event in run.tool_error_events:
        assert event.get("data", {}).get("tool_name") in seen_tool_names


def _classify(run: _Run, case: _Case) -> str:
    finish_reason = str(run.completion_metadata.get("finish_reason") or "")
    if finish_reason == "error":
        return "request_error"
    if not run.tool_start_events:
        return "tool_free_answer"
    final_text = run.final_text.lower()
    if case.expected_terms and not any(term in final_text for term in case.expected_terms):
        return "tool_answer_miss"
    return "tool_answer_ok"


def _assert_case(run: _Run) -> None:
    assert run.completion_metadata.get("trace_id")
    assert run.final_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "lane"),
    [(case, lane) for case in _CASES for lane in _LANES],
    ids=lambda value: value.name,
)
async def test_local_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
    lane: _Lane,
) -> None:
    live_model = _live_model()
    workspace_root = _workspace_root()
    monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(workspace_root))
    _load_local_tool_registry()

    run = await _run_live_case(
        tmp_path=tmp_path,
        lane=lane.name,
        enabled_skills=lane.skills,
        model=live_model,
        prompt=case.prompt,
    )

    _assert_case(run)
    _assert_lane(run, lane)
    outcome = _classify(run, case)
    terminal_tool_calls = run.completion_metadata.get("tool_loop_terminal_tool_call_count")
    print(
        f"LIVE_RESULT case={case.name} lane={lane.name} outcome={outcome} "
        f"tool_calls={len(run.tool_start_events)} terminal_tool_calls={terminal_tool_calls} "
        f"finish_reason={run.completion_metadata.get('finish_reason')}"
    )
