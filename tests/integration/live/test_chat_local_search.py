"""ChatService local search scenarios."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pytest
from dotenv import load_dotenv
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.provider_service import _is_vertex_provider
from houyi_studio.server.chat.settings_store import ProviderConfig, SettingsStore
from houyi_studio.server.chat.types import (
    CreateConversationRequest,
    SendMessageRequest,
)

from houyi.adapters.llm.models import normalize_model_id
from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.infrastructure.config.env_config import EnvConfig
from houyi.skills.builtin.local_tools import register_builtin_local_tools

_TYPED_LANE_SKILLS = [
    "houyi_list_dir",
    "houyi_find_files",
    "houyi_grep",
    "houyi_read_file",
]

_CLI_LANE_SKILLS = ["houyi_local_cli"]
_CLI_PROJECTED_SKILLS = ["houyi_local_cli"]
_CLI_CHAIN_SKILLS = ["houyi_local_cli_chain"]
_CLI_PROJECTED_TOOL_NAMES = {
    "houyi_local_cli_read",
    "houyi_local_cli_list",
    "houyi_local_cli_find",
    "houyi_local_cli_grep",
}

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
    schema_exposure: str = "full"


@dataclass(frozen=True)
class _ModelProfile:
    key: str
    label: str
    model: str | None
    provider_id: str = ""
    provider_name: str = ""
    base_url: str = ""


@dataclass(frozen=True)
class _Run:
    completion_metadata: dict[str, object]
    tool_start_events: list[dict[str, object]]
    tool_result_events: list[dict[str, object]]
    tool_error_events: list[dict[str, object]]
    message_error_events: list[dict[str, object]]
    soft_tool_failures: list[dict[str, object]]
    iteration_events: list[dict[str, object]]
    final_text: str


@dataclass(frozen=True)
class _Aggregate:
    model_key: str
    model_label: str
    model: str | None
    lane: str
    case: str
    repeats: int
    outcome_counts: dict[str, int]
    tool_use_rate: float
    request_error_rate: float
    tool_free_answer_rate: float
    avg_tool_calls: float
    avg_soft_failures: float
    avg_iterations: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    avg_generation_time_ms: float
    avg_first_token_ms: float
    avg_tool_duration_ms: float
    post_error_recovery_success_rate: float
    avg_extra_tool_calls_after_first_failure: float
    avg_extra_iterations_after_first_failure: float
    avg_extra_tokens_after_first_failure: float
    first_soft_failure_counts: dict[str, int]
    invalid_chain_argument_rate: float
    unsupported_chain_command_rate: float
    projection_failed_rate: float
    provider_request_error_counts: dict[str, int]


@dataclass(frozen=True)
class _ProfilePreflight:
    profile: _ModelProfile
    request_error_code: str | None = None


def _metadata_number(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _usage_number(run: _Run, key: str) -> float:
    usage = run.completion_metadata.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    value = usage.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _generation_number(run: _Run, key: str) -> float:
    return _metadata_number(run.completion_metadata, key) or 0.0


def _tool_duration_values(run: _Run) -> list[float]:
    durations: list[float] = []
    for event in run.tool_result_events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        duration = _metadata_number(data, "duration_ms")
        if duration is not None:
            durations.append(duration)
            continue
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            continue
        nested_duration = _metadata_number(metadata, "duration_ms")
        if nested_duration is not None:
            durations.append(nested_duration)
    for event in run.tool_error_events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        duration = _metadata_number(data, "duration_ms")
        if duration is not None:
            durations.append(duration)
    return durations


def _request_error_code(run: _Run) -> str:
    for event in run.message_error_events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        error_code = str(data.get("error_code") or "").strip().lower()
        if error_code:
            return error_code
    error_code = str(run.completion_metadata.get("final_stream_error_code") or "").strip().lower()
    return error_code or "request_error"


def _is_provider_request_error(run: _Run) -> bool:
    if str(run.completion_metadata.get("finish_reason") or "") != "error":
        return False
    return _request_error_code(run).startswith("provider_")


def _average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _model_family_key(model: str) -> str:
    normalized = normalize_model_id(model).lower()
    if "deepseek" in normalized:
        return "deepseek"
    if "kimi" in normalized:
        return "kimi"
    if "minimax" in normalized:
        return "minimax"
    if "glm" in normalized:
        return "glm"
    if "gemini" in normalized:
        return "gemini"
    return normalized or "unknown"


def _provider_label(provider: ProviderConfig) -> str:
    return str(provider.name or provider.id or "provider").strip() or "provider"


def _settings_store() -> SettingsStore:
    settings_path = os.getenv("HOUYI_CHAT_SETTINGS_PATH", "data/settings.json")
    return SettingsStore(settings_path=settings_path)


def _workspace_root() -> Path:
    configured = (os.getenv("HOUYI_WORKSPACE_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def _load_local_tool_registry() -> None:
    required_tool_names = {
        "houyi_read_file",
        "houyi_find_files",
        "houyi_grep",
        "houyi_list_dir",
        "houyi_local_cli",
        "houyi_local_cli_chain",
    }
    if required_tool_names <= set(DEFAULT_SKILL_REGISTRY.list_names()):
        return
    register_builtin_local_tools(DEFAULT_SKILL_REGISTRY)


def _live_model() -> str | None:
    selected_profiles = _live_profiles()
    model = (os.getenv("HOUYI_LIVE_E2E_MODEL") or "").strip()
    return model or selected_profiles[0].model or None


def _tool_result_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    return result if isinstance(result, dict) else None


def _is_soft_tool_failure(payload: dict[str, Any] | None) -> bool:
    return isinstance(payload, dict) and payload.get("success") is False


def _has_recovery_signal(run: _Run) -> bool:
    return bool(run.tool_error_events or run.soft_tool_failures)


def _soft_failure_kind(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "none"
    message = str(payload.get("message") or "").strip().lower()
    if not message:
        return "soft_failure"
    if "invalid chain argument" in message:
        return "invalid_chain_argument"
    if "projection_failed" in message:
        return "projection_failed"
    if "not found" in message:
        return "not_found"
    if "invalid regex" in message:
        return "invalid_regex"
    return message.split(":", 1)[0].replace(" ", "_")[:64] or "soft_failure"


def _has_projection_failed(run: _Run) -> bool:
    for event in run.tool_result_events:
        payload = _tool_result_payload(event)
        if not isinstance(payload, dict):
            continue
        message = str(payload.get("message") or "").lower()
        if "projection_failed" in message:
            return True
        data = payload.get("data")
        if isinstance(data, dict):
            error_code = str(data.get("error_code") or "").lower()
            if error_code == "projection_failed":
                return True
    return False


def _has_invalid_chain_argument(run: _Run) -> bool:
    for event in run.soft_tool_failures:
        payload = _tool_result_payload(event)
        message = str(payload.get("message") or "").lower() if isinstance(payload, dict) else ""
        if "invalid chain argument" in message:
            return True
    return False


def _has_unsupported_chain_command(run: _Run) -> bool:
    for event in run.soft_tool_failures:
        payload = _tool_result_payload(event)
        message = str(payload.get("message") or "").lower() if isinstance(payload, dict) else ""
        if "unsupported chain command" in message:
            return True
    return False


def _event_tool_call_id(event: dict[str, Any]) -> str:
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("tool_call_id") or "")


def _event_round_index(event: dict[str, Any]) -> int | None:
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    round_index = data.get("round_index")
    return round_index if isinstance(round_index, int) else None


def _failure_events(run: _Run) -> list[dict[str, Any]]:
    return [*run.tool_error_events, *run.soft_tool_failures]


def _first_failure_event(run: _Run) -> dict[str, Any] | None:
    start_indexes = {
        _event_tool_call_id(event): index
        for index, event in enumerate(run.tool_start_events)
        if _event_tool_call_id(event)
    }
    ranked_failures: list[tuple[int, int, dict[str, Any]]] = []
    for order, event in enumerate(_failure_events(run)):
        tool_call_id = _event_tool_call_id(event)
        if not tool_call_id or tool_call_id not in start_indexes:
            continue
        ranked_failures.append((start_indexes[tool_call_id], order, event))
    if not ranked_failures:
        return None
    ranked_failures.sort(key=lambda item: (item[0], item[1]))
    return ranked_failures[0][2]


def _extra_tool_calls_after_first_failure(run: _Run) -> float:
    first_failure = _first_failure_event(run)
    if first_failure is None:
        return 0.0
    start_indexes = {
        _event_tool_call_id(event): index
        for index, event in enumerate(run.tool_start_events)
        if _event_tool_call_id(event)
    }
    tool_call_id = _event_tool_call_id(first_failure)
    start_index = start_indexes.get(tool_call_id)
    if start_index is None:
        return 0.0
    return float(max(0, len(run.tool_start_events) - start_index - 1))


def _extra_iterations_after_first_failure(run: _Run) -> float:
    first_failure = _first_failure_event(run)
    if first_failure is None:
        return 0.0
    failure_round = _event_round_index(first_failure)
    if failure_round is None:
        return 0.0
    later_rounds = {
        round_index
        for round_index in (_event_round_index(event) for event in run.iteration_events)
        if isinstance(round_index, int) and round_index > failure_round
    }
    return float(len(later_rounds))


def _iteration_usage(run: _Run, round_index: int, key: str) -> float:
    for event in run.iteration_events:
        if _event_round_index(event) != round_index:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        usage = data.get("usage")
        if not isinstance(usage, dict):
            continue
        value = usage.get(key)
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _extra_tokens_after_first_failure(run: _Run) -> float:
    first_failure = _first_failure_event(run)
    if first_failure is None:
        return 0.0
    failure_round = _event_round_index(first_failure)
    if failure_round is None:
        return 0.0
    later_rounds = {
        round_index
        for round_index in (_event_round_index(event) for event in run.iteration_events)
        if isinstance(round_index, int) and round_index > failure_round
    }
    return float(
        sum(_iteration_usage(run, round_index, "total_tokens") for round_index in later_rounds)
    )


def _build_env_deepseek_profile(env: EnvConfig) -> _ModelProfile | None:
    model = (os.getenv("HOUYI_LIVE_E2E_MODEL") or env.deepseek_model or "").strip()
    if env.default_llm_provider != "siliconflow" or not env.siliconflow_api_key or not model:
        return None
    return _ModelProfile(
        key="deepseek",
        label=f"DeepSeek::{model}",
        model=model,
        provider_id="_env_siliconflow",
        provider_name="SiliconFlow (env)",
        base_url=env.siliconflow_base_url,
    )


def _candidate_profiles_from_settings() -> list[_ModelProfile]:
    settings = _settings_store().get()
    candidates: list[_ModelProfile] = []
    for provider in settings.providers:
        if not provider.enabled:
            continue
        for model in provider.models:
            family = _model_family_key(model)
            if family not in {"deepseek", "kimi", "minimax", "glm", "gemini"}:
                continue
            provider_id = str(provider.id or "").strip()
            provider_name = _provider_label(provider)
            base_url = str(provider.base_url or "").strip()
            if family == "gemini" and not (
                _is_vertex_provider(provider_id, base_url)
                or _is_vertex_provider(provider_name.lower(), base_url)
            ):
                continue
            key = "vertex_gemini" if family == "gemini" else family
            candidates.append(
                _ModelProfile(
                    key=key,
                    label=f"{provider_name}::{model}",
                    model=model,
                    provider_id=provider_id,
                    provider_name=provider_name,
                    base_url=base_url,
                )
            )
    return candidates


def _live_profiles() -> list[_ModelProfile]:
    load_dotenv()
    if os.getenv("HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS") != "1":
        pytest.skip(
            "Live local-search test disabled by default; set HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 to enable"
        )
    env = EnvConfig.get()
    settings_profiles = _candidate_profiles_from_settings()
    chosen: dict[str, _ModelProfile] = {}
    for profile in settings_profiles:
        chosen.setdefault(profile.key, profile)
    env_profile = _build_env_deepseek_profile(env)
    if env_profile is not None:
        chosen.setdefault(env_profile.key, env_profile)
    profiles = [
        profile
        for key in ("deepseek", "kimi", "minimax", "glm", "vertex_gemini")
        if (profile := chosen.get(key)) is not None
    ]
    raw_filter = str(os.getenv("HOUYI_LIVE_E2E_PROFILE_KEYS") or "").strip()
    if raw_filter:
        allowed_keys = {key.strip().lower() for key in raw_filter.split(",") if key.strip()}
        profiles = [profile for profile in profiles if profile.key in allowed_keys]
    if not profiles:
        pytest.skip(
            "No enabled live model profiles found in settings/env for Phase 4.2 multi-model testing"
        )
    return profiles


_CASES = [
    _Case(
        name="search",
        prompt=(
            "Use local tools to inspect the current workspace and find the most relevant skill definition file for web search. "
            "You must make at least one local tool call before answering. "
            "Do not guess paths, file contents, or workspace structure. "
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
        require_tool_error=True,
    ),
    _Case(
        name="validation_repair",
        prompt=(
            "Use local tools to read the first 20 lines of tests/integration/live/test_chat_local_search.py. "
            "You must make at least one local tool call before answering. "
            "First intentionally make an invalid read attempt with start_line set to 0. "
            "If the tool rejects that request, correct the arguments and retry with a valid line range, then summarize the file purpose."
        ),
        expected_terms=("local search", "scenario"),
        require_tool_error=True,
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
    _Case(
        name="workflow_selection",
        prompt=(
            "Use local tools to inspect the current workspace and find the most relevant skill definition file for web search. "
            "You must make at least one local tool call before answering. "
            "If a chain-style workflow is available, you may combine multiple actions into one tool call, but you must still decide the sequence yourself. "
            "Do not guess paths, file contents, or workspace structure. "
            "Return the relative file path, one short reason, and a preview of the first 20 lines."
        ),
        expected_terms=("web_search", "skill.md"),
    ),
    _Case(
        name="chain_fallback",
        prompt=(
            "Use local tools to open houyi/skills/websearch/SKILL.md and preview the first 20 lines. "
            "You must make at least one local tool call before answering. "
            "If that path does not exist, use the available local actions to recover, locate the correct file, and continue. "
            "If a chain-style workflow is available, you may use fallback within that workflow, but you must decide when to fallback and when to stop."
        ),
        expected_terms=("web_search", "skill.md"),
        require_tool_error=True,
    ),
]

_LANES = [
    _Lane(name="typed", skills=_TYPED_LANE_SKILLS),
    _Lane(name="cli_full", skills=_CLI_LANE_SKILLS),
    _Lane(name="cli_projected", skills=_CLI_PROJECTED_SKILLS, schema_exposure="projected"),
    _Lane(
        name="cli_projected_minimal",
        skills=_CLI_PROJECTED_SKILLS,
        schema_exposure="projected_minimal",
    ),
    _Lane(name="cli_chain", skills=_CLI_CHAIN_SKILLS),
    _Lane(name="cli_chain_minimal", skills=_CLI_CHAIN_SKILLS, schema_exposure="minimal"),
]


def _parse_sse_events(raw_chunks: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for chunk in raw_chunks:
        text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
        lines = re.split(r"\r?\n", text)
        for line in lines:
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("event: "):
                current["event"] = line[7:]
            elif line.startswith("data: "):
                current["data"] = json.loads(line[6:])
            elif line == "":
                if "event" in current:
                    events.append(current)
                current = {}
    if "event" in current:
        events.append(current)
    return events


def _delta_text(events: list[dict[str, Any]]) -> str:
    return "".join(
        str(event.get("data", {}).get("content") or "")
        for event in events
        if event.get("event") == "message.delta"
    )


async def _run_live_case(
    *,
    tmp_path: Path,
    lane: str,
    enabled_skills: list[str],
    schema_exposure: str,
    model: str | None,
    settings_store: SettingsStore | None,
    prompt: str,
) -> _Run:
    store = JsonStore(data_dir=tmp_path / f"chat-store-{lane}")
    service = ChatService(
        json_store=store, default_model=model or "", settings_store=settings_store
    )
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
            schema_exposure=schema_exposure,
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
    message_error_events = [event for event in events if event.get("event") == "message.error"]
    soft_tool_failures = [
        event for event in tool_result_events if _is_soft_tool_failure(_tool_result_payload(event))
    ]
    iteration_events = [event for event in events if event.get("event") == "agent.iteration"]
    final_text = _delta_text(events)

    return _Run(
        completion_metadata=completion_metadata,
        tool_start_events=tool_start_events,
        tool_result_events=tool_result_events,
        tool_error_events=tool_error_events,
        message_error_events=message_error_events,
        soft_tool_failures=soft_tool_failures,
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
    elif lane.name == "cli_full":
        assert seen_tool_names == {"houyi_local_cli"}
    elif lane.name.startswith("cli_chain"):
        assert seen_tool_names == {"houyi_local_cli_chain"}
    else:
        assert seen_tool_names <= _CLI_PROJECTED_TOOL_NAMES
    for event in run.tool_error_events:
        assert event.get("data", {}).get("tool_name") in seen_tool_names


def _classify(run: _Run, case: _Case) -> str:
    finish_reason = str(run.completion_metadata.get("finish_reason") or "")
    if finish_reason == "error":
        return "request_error"
    if not run.final_text:
        return "empty_answer"
    if not run.tool_start_events:
        return "tool_free_answer"
    final_text = run.final_text.lower()
    if case.expected_terms and not any(term in final_text for term in case.expected_terms):
        return "tool_answer_miss"
    return "tool_answer_ok"


def _assert_case(run: _Run) -> None:
    assert run.completion_metadata.get("trace_id")


def _assert_recovery_case(run: _Run, case: _Case) -> None:
    if not case.require_tool_error:
        return
    if _is_provider_request_error(run):
        return
    if not run.tool_start_events:
        return
    if case.name == "validation_repair":
        assert run.tool_error_events or run.soft_tool_failures
        assert len(run.tool_start_events) >= 2
        assert run.tool_error_events
        first_error = run.tool_error_events[0].get("data", {}).get("error")
        assert isinstance(first_error, dict)
        assert isinstance(first_error.get("recovery_guidance"), dict)
        return
    assert run.tool_start_events
    if case.name == "chain_fallback":
        assert _has_recovery_signal(run)
        return
    if _has_recovery_signal(run):
        if case.name == "chain_fallback":
            return
        assert len(run.tool_start_events) >= 2


def _record_recovery_shape(run: _Run, case: _Case) -> None:
    if not case.require_tool_error:
        return
    if _is_provider_request_error(run):
        return
    if not run.tool_start_events:
        return
    if case.name == "validation_repair":
        assert run.tool_error_events or run.soft_tool_failures
        assert run.tool_error_events
        first_error = run.tool_error_events[0].get("data", {}).get("error")
        assert isinstance(first_error, dict)
        assert isinstance(first_error.get("recovery_guidance"), dict)
        return
    assert run.tool_start_events


def _print_recovery_debug(run: _Run) -> None:
    if run.message_error_events:
        first_message_error = run.message_error_events[0].get("data")
        print(
            "RECOVERY_DEBUG "
            f"message_error={json.dumps(first_message_error, ensure_ascii=False, sort_keys=True)}"
        )
    if run.tool_error_events:
        first_error = run.tool_error_events[0].get("data", {}).get("error")
        print(
            f"RECOVERY_DEBUG tool_error={json.dumps(first_error, ensure_ascii=False, sort_keys=True)}"
        )
    if run.soft_tool_failures:
        first_soft_failure = _tool_result_payload(run.soft_tool_failures[0])
        print(
            "RECOVERY_DEBUG "
            f"soft_failure={json.dumps(first_soft_failure, ensure_ascii=False, sort_keys=True)}"
        )


async def _run_profile_preflight(
    *,
    tmp_path: Path,
    profile: _ModelProfile,
    settings_store: SettingsStore | None,
) -> _ProfilePreflight:
    store = JsonStore(data_dir=tmp_path / f"preflight-store-{profile.key}")
    service = ChatService(
        json_store=store,
        default_model=profile.model or "",
        settings_store=settings_store,
    )
    conversation = service.create_conversation(
        CreateConversationRequest(
            title=f"preflight-{profile.key}",
            model=profile.model or "",
            system_instructions="Return READY.",
        )
    )

    raw_chunks: list[str] = []
    async for chunk in service.send_message(
        conversation["conversation_id"],
        SendMessageRequest(
            content="Return READY.",
            model=profile.model,
            enable_tool_calls=False,
            enable_skills=[],
        ),
    ):
        raw_chunks.append(chunk)

    events = _parse_sse_events(raw_chunks)
    complete_event = next(
        event for event in reversed(events) if event.get("event") == "message.complete"
    )
    completion_metadata = dict(complete_event.get("data", {}).get("metadata", {}))
    run = _Run(
        completion_metadata=completion_metadata,
        tool_start_events=[],
        tool_result_events=[],
        tool_error_events=[],
        message_error_events=[event for event in events if event.get("event") == "message.error"],
        soft_tool_failures=[],
        iteration_events=[event for event in events if event.get("event") == "tool_loop.iteration"],
        final_text=_delta_text(events),
    )
    if _is_provider_request_error(run):
        return _ProfilePreflight(profile=profile, request_error_code=_request_error_code(run))
    return _ProfilePreflight(profile=profile)


def _blocked_profile_aggregate(
    *,
    preflight: _ProfilePreflight,
    lane: _Lane,
    case: _Case,
    repeats: int,
) -> _Aggregate:
    request_error_code = preflight.request_error_code or "provider_request_failed"
    print(
        f"LIVE_PROFILE_PREFLIGHT model={preflight.profile.label} status=blocked error_code={request_error_code}"
    )
    return _Aggregate(
        model_key=preflight.profile.key,
        model_label=preflight.profile.label,
        model=preflight.profile.model,
        lane=lane.name,
        case=case.name,
        repeats=repeats,
        outcome_counts={"request_error": repeats},
        tool_use_rate=0.0,
        request_error_rate=1.0,
        tool_free_answer_rate=0.0,
        avg_tool_calls=0.0,
        avg_soft_failures=0.0,
        avg_iterations=0.0,
        avg_prompt_tokens=0.0,
        avg_completion_tokens=0.0,
        avg_total_tokens=0.0,
        avg_generation_time_ms=0.0,
        avg_first_token_ms=0.0,
        avg_tool_duration_ms=0.0,
        post_error_recovery_success_rate=0.0,
        avg_extra_tool_calls_after_first_failure=0.0,
        avg_extra_iterations_after_first_failure=0.0,
        avg_extra_tokens_after_first_failure=0.0,
        first_soft_failure_counts={},
        invalid_chain_argument_rate=0.0,
        unsupported_chain_command_rate=0.0,
        projection_failed_rate=0.0,
        provider_request_error_counts={request_error_code: repeats},
    )


async def _run_repeated_case(
    *,
    tmp_path: Path,
    profile: _ModelProfile,
    lane: _Lane,
    case: _Case,
    repeats: int,
) -> _Aggregate:
    outcome_counts: Counter[str] = Counter()
    tool_calls: list[float] = []
    soft_failures: list[float] = []
    iteration_counts: list[float] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    total_tokens: list[float] = []
    generation_time_ms: list[float] = []
    first_token_ms: list[float] = []
    tool_durations: list[float] = []
    recovery_successes = 0
    recovery_required_runs = 0
    extra_tool_calls_after_failure: list[float] = []
    extra_iterations_after_failure: list[float] = []
    extra_tokens_after_failure: list[float] = []
    first_soft_failure_counts: Counter[str] = Counter()
    provider_request_error_counts: Counter[str] = Counter()
    invalid_chain_argument_runs = 0
    unsupported_chain_command_runs = 0
    projection_failed_runs = 0
    workspace_root = _workspace_root()
    os.environ["HOUYI_WORKSPACE_ROOT"] = str(workspace_root)
    settings_store = _settings_store()
    preflight = await _run_profile_preflight(
        tmp_path=tmp_path,
        profile=profile,
        settings_store=settings_store,
    )
    if preflight.request_error_code:
        return _blocked_profile_aggregate(
            preflight=preflight,
            lane=lane,
            case=case,
            repeats=repeats,
        )
    for index in range(repeats):
        run = await _run_live_case(
            tmp_path=tmp_path / f"batch-{case.name}-{lane.name}-{index}",
            lane=f"{profile.key}-{lane.name}-{index}",
            enabled_skills=lane.skills,
            schema_exposure=lane.schema_exposure,
            model=profile.model,
            settings_store=settings_store,
            prompt=case.prompt,
        )
        _assert_case(run)
        _record_recovery_shape(run, case)
        if case.require_tool_error:
            _print_recovery_debug(run)
        _assert_lane(run, lane)
        outcome = _classify(run, case)
        outcome_counts[outcome] += 1
        tool_calls.append(float(len(run.tool_start_events)))
        soft_failures.append(float(len(run.soft_tool_failures)))
        iteration_counts.append(float(len(run.iteration_events)))
        prompt_tokens.append(_usage_number(run, "prompt_tokens"))
        completion_tokens.append(_usage_number(run, "completion_tokens"))
        total_tokens.append(_usage_number(run, "total_tokens"))
        generation_time_ms.append(_generation_number(run, "generation_time_ms"))
        first_token_ms.append(_generation_number(run, "first_token_ms"))
        tool_durations.extend(_tool_duration_values(run))
        if _is_provider_request_error(run):
            provider_request_error_counts[_request_error_code(run)] += 1
        if _has_recovery_signal(run):
            recovery_required_runs += 1
            if outcome == "tool_answer_ok":
                recovery_successes += 1
            extra_tool_calls_after_failure.append(_extra_tool_calls_after_first_failure(run))
            extra_iterations_after_failure.append(_extra_iterations_after_first_failure(run))
            extra_tokens_after_failure.append(_extra_tokens_after_first_failure(run))
        if run.soft_tool_failures:
            first_soft_failure_counts[
                _soft_failure_kind(_tool_result_payload(run.soft_tool_failures[0]))
            ] += 1
        if _has_invalid_chain_argument(run):
            invalid_chain_argument_runs += 1
        if _has_unsupported_chain_command(run):
            unsupported_chain_command_runs += 1
        if _has_projection_failed(run):
            projection_failed_runs += 1
        terminal_tool_calls = run.completion_metadata.get("tool_loop_terminal_tool_call_count")
        print(
            f"LIVE_BATCH_RESULT model={profile.label} case={case.name} lane={lane.name} repeat={index + 1}/{repeats} "
            f"outcome={outcome} tool_calls={len(run.tool_start_events)} "
            f"tool_errors={len(run.tool_error_events)} soft_failures={len(run.soft_tool_failures)} "
            f"terminal_tool_calls={terminal_tool_calls} finish_reason={run.completion_metadata.get('finish_reason')}"
        )
    repeats_float = float(repeats)
    return _Aggregate(
        model_key=profile.key,
        model_label=profile.label,
        model=profile.model,
        lane=lane.name,
        case=case.name,
        repeats=repeats,
        outcome_counts=dict(outcome_counts),
        tool_use_rate=(
            repeats
            - outcome_counts.get("tool_free_answer", 0)
            - outcome_counts.get("request_error", 0)
        )
        / repeats_float,
        request_error_rate=outcome_counts.get("request_error", 0) / repeats_float,
        tool_free_answer_rate=outcome_counts.get("tool_free_answer", 0) / repeats_float,
        avg_tool_calls=_average(tool_calls),
        avg_soft_failures=_average(soft_failures),
        avg_iterations=_average(iteration_counts),
        avg_prompt_tokens=_average(prompt_tokens),
        avg_completion_tokens=_average(completion_tokens),
        avg_total_tokens=_average(total_tokens),
        avg_generation_time_ms=_average(generation_time_ms),
        avg_first_token_ms=_average(first_token_ms),
        avg_tool_duration_ms=_average(tool_durations),
        post_error_recovery_success_rate=(
            recovery_successes / float(recovery_required_runs) if recovery_required_runs else 0.0
        ),
        avg_extra_tool_calls_after_first_failure=_average(extra_tool_calls_after_failure),
        avg_extra_iterations_after_first_failure=_average(extra_iterations_after_failure),
        avg_extra_tokens_after_first_failure=_average(extra_tokens_after_failure),
        first_soft_failure_counts=dict(first_soft_failure_counts),
        invalid_chain_argument_rate=invalid_chain_argument_runs / repeats_float,
        unsupported_chain_command_rate=unsupported_chain_command_runs / repeats_float,
        projection_failed_rate=projection_failed_runs / repeats_float,
        provider_request_error_counts=dict(provider_request_error_counts),
    )


@pytest.mark.asyncio
async def test_blocked_profile_short_circuit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules[__name__]
    profile = _ModelProfile(key="deepseek", label="DeepSeek::blocked", model="deepseek-test")
    lane = _Lane(name="cli_chain", skills=_CLI_CHAIN_SKILLS)
    case = next(item for item in _CASES if item.name == "workflow_selection")

    async def _fake_preflight(**_: object) -> _ProfilePreflight:
        return _ProfilePreflight(
            profile=profile,
            request_error_code="provider_quota_exhausted",
        )

    async def _unexpected_live_case(**_: object) -> _Run:
        raise AssertionError("blocked profiles should not execute per-repeat live cases")

    monkeypatch.setattr(module, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_settings_store", lambda: None)
    monkeypatch.setattr(module, "_run_profile_preflight", _fake_preflight)
    monkeypatch.setattr(module, "_run_live_case", _unexpected_live_case)

    aggregate = await _run_repeated_case(
        tmp_path=tmp_path,
        profile=profile,
        lane=lane,
        case=case,
        repeats=2,
    )

    assert aggregate.outcome_counts == {"request_error": 2}
    assert aggregate.request_error_rate == 1.0
    assert aggregate.avg_tool_calls == 0.0
    assert aggregate.provider_request_error_counts == {"provider_quota_exhausted": 2}


def test_profile_key_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    module = sys.modules[__name__]
    monkeypatch.setenv("HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS", "1")
    monkeypatch.setenv("HOUYI_LIVE_E2E_PROFILE_KEYS", "deepseek,kimi")
    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        module,
        "_candidate_profiles_from_settings",
        lambda: [
            _ModelProfile(key="deepseek", label="DeepSeek::x", model="deepseek-x"),
            _ModelProfile(key="kimi", label="Kimi::x", model="kimi-x"),
            _ModelProfile(key="vertex_gemini", label="Vertex::x", model="gemini-x"),
        ],
    )
    monkeypatch.setattr(module, "_build_env_deepseek_profile", lambda _env: None)
    monkeypatch.setattr(
        module, "EnvConfig", type("_StubEnvConfig", (), {"get": staticmethod(lambda: object())})
    )

    selected_profiles = _live_profiles()

    assert [profile.key for profile in selected_profiles] == ["deepseek", "kimi"]


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
    settings_store = _settings_store()

    run = await _run_live_case(
        tmp_path=tmp_path,
        lane=lane.name,
        enabled_skills=lane.skills,
        schema_exposure=lane.schema_exposure,
        model=live_model,
        settings_store=settings_store,
        prompt=case.prompt,
    )

    _assert_case(run)
    _assert_recovery_case(run, case)
    if case.require_tool_error:
        _print_recovery_debug(run)
    _assert_lane(run, lane)
    outcome = _classify(run, case)
    terminal_tool_calls = run.completion_metadata.get("tool_loop_terminal_tool_call_count")
    print(
        f"LIVE_RESULT case={case.name} lane={lane.name} outcome={outcome} "
        f"tool_calls={len(run.tool_start_events)} tool_errors={len(run.tool_error_events)} "
        f"soft_failures={len(run.soft_tool_failures)} terminal_tool_calls={terminal_tool_calls} "
        f"finish_reason={run.completion_metadata.get('finish_reason')}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda value: value.name)
async def test_local_search_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
) -> None:
    live_model = _live_model()
    workspace_root = _workspace_root()
    monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(workspace_root))
    _load_local_tool_registry()
    repeats = int(os.getenv("HOUYI_LIVE_E2E_REPEATS") or "2")
    profile = _ModelProfile(
        key="default", label=f"default::{live_model or 'env'}", model=live_model
    )

    aggregates = [
        await _run_repeated_case(
            tmp_path=tmp_path,
            profile=profile,
            lane=lane,
            case=case,
            repeats=repeats,
        )
        for lane in _LANES
    ]
    by_lane = {aggregate.lane: aggregate for aggregate in aggregates}

    print(
        "LIVE_LANE_SUMMARY "
        f"case={case.name} "
        + " ".join(
            f"{aggregate.lane}={aggregate.outcome_counts} "
            f"{aggregate.lane}_tool_use_rate={aggregate.tool_use_rate:.2f} "
            f"{aggregate.lane}_request_error_rate={aggregate.request_error_rate:.2f} "
            f"{aggregate.lane}_tool_free_answer_rate={aggregate.tool_free_answer_rate:.2f} "
            f"{aggregate.lane}_avg_tool_calls={aggregate.avg_tool_calls:.2f} "
            f"{aggregate.lane}_avg_soft_failures={aggregate.avg_soft_failures:.2f} "
            f"{aggregate.lane}_avg_iterations={aggregate.avg_iterations:.2f} "
            f"{aggregate.lane}_avg_prompt_tokens={aggregate.avg_prompt_tokens:.1f} "
            f"{aggregate.lane}_avg_completion_tokens={aggregate.avg_completion_tokens:.1f} "
            f"{aggregate.lane}_avg_total_tokens={aggregate.avg_total_tokens:.1f} "
            f"{aggregate.lane}_avg_generation_time_ms={aggregate.avg_generation_time_ms:.1f} "
            f"{aggregate.lane}_avg_first_token_ms={aggregate.avg_first_token_ms:.1f} "
            f"{aggregate.lane}_avg_tool_duration_ms={aggregate.avg_tool_duration_ms:.1f} "
            f"{aggregate.lane}_invalid_chain_argument_rate={aggregate.invalid_chain_argument_rate:.2f} "
            f"{aggregate.lane}_unsupported_chain_command_rate={aggregate.unsupported_chain_command_rate:.2f} "
            f"{aggregate.lane}_projection_failed_rate={aggregate.projection_failed_rate:.2f}"
            for aggregate in aggregates
        )
    )

    for aggregate in aggregates:
        assert aggregate.request_error_rate == 0.0

    if case.name == "workflow_selection":
        assert by_lane["cli_chain"].avg_tool_calls <= by_lane["cli_projected"].avg_tool_calls
        assert (
            by_lane["cli_chain_minimal"].avg_tool_calls
            <= by_lane["cli_projected_minimal"].avg_tool_calls
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "lane"),
    [
        (case, lane)
        for case in _CASES
        if case.name in {"workflow_selection", "chain_fallback"}
        for lane in _LANES
        if lane.name in {"cli_projected", "cli_chain", "cli_chain_minimal"}
    ],
    ids=lambda value: value.name,
)
async def test_local_search_multi_model_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
    lane: _Lane,
) -> None:
    profiles = _live_profiles()
    workspace_root = _workspace_root()
    monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(workspace_root))
    _load_local_tool_registry()
    repeats = int(os.getenv("HOUYI_LIVE_E2E_REPEATS") or "2")

    aggregates = [
        await _run_repeated_case(
            tmp_path=tmp_path / profile.key,
            profile=profile,
            lane=lane,
            case=case,
            repeats=repeats,
        )
        for profile in profiles
    ]

    print(
        "LIVE_MODEL_SUMMARY "
        f"case={case.name} lane={lane.name} "
        + " ".join(
            f"{aggregate.model_key}={{outcomes:{aggregate.outcome_counts},"
            f"avg_tool_calls:{aggregate.avg_tool_calls:.2f},"
            f"avg_soft_failures:{aggregate.avg_soft_failures:.2f},"
            f"avg_total_tokens:{aggregate.avg_total_tokens:.1f},"
            f"avg_generation_time_ms:{aggregate.avg_generation_time_ms:.1f},"
            f"avg_first_token_ms:{aggregate.avg_first_token_ms:.1f},"
            f"post_error_recovery_success_rate:{aggregate.post_error_recovery_success_rate:.2f},"
            f"avg_extra_tool_calls_after_first_failure:{aggregate.avg_extra_tool_calls_after_first_failure:.2f},"
            f"avg_extra_iterations_after_first_failure:{aggregate.avg_extra_iterations_after_first_failure:.2f},"
            f"avg_extra_tokens_after_first_failure:{aggregate.avg_extra_tokens_after_first_failure:.1f},"
            f"invalid_chain_argument_rate:{aggregate.invalid_chain_argument_rate:.2f},"
            f"unsupported_chain_command_rate:{aggregate.unsupported_chain_command_rate:.2f},"
            f"projection_failed_rate:{aggregate.projection_failed_rate:.2f},"
            f"provider_request_errors:{aggregate.provider_request_error_counts},"
            f"first_soft_failures:{aggregate.first_soft_failure_counts}}}"
            for aggregate in aggregates
        )
    )

    for aggregate in aggregates:
        if aggregate.request_error_rate == 0.0:
            continue
        request_error_keys = set(aggregate.provider_request_error_counts)
        assert request_error_keys
        assert request_error_keys <= {
            "provider_quota_exhausted",
            "provider_auth_failed",
            "provider_permission_denied",
            "provider_rate_limited",
            "provider_timeout",
            "provider_network_error",
            "provider_request_failed",
        }


def test_recovery_efficiency_helpers_measure_post_failure_cost() -> None:
    run = _Run(
        completion_metadata={"usage": {"total_tokens": 60}},
        tool_start_events=[
            {"data": {"tool_call_id": "call-1", "round_index": 1}},
            {"data": {"tool_call_id": "call-2", "round_index": 2}},
            {"data": {"tool_call_id": "call-3", "round_index": 3}},
        ],
        tool_result_events=[
            {"data": {"tool_call_id": "call-1", "round_index": 1, "result": {"success": False}}},
            {"data": {"tool_call_id": "call-2", "round_index": 2, "result": {"success": True}}},
            {"data": {"tool_call_id": "call-3", "round_index": 3, "result": {"success": True}}},
        ],
        tool_error_events=[],
        message_error_events=[],
        soft_tool_failures=[
            {"data": {"tool_call_id": "call-1", "round_index": 1, "result": {"success": False}}}
        ],
        iteration_events=[
            {"data": {"round_index": 1, "usage": {"total_tokens": 10}}},
            {"data": {"round_index": 2, "usage": {"total_tokens": 15}}},
            {"data": {"round_index": 3, "usage": {"total_tokens": 20}}},
        ],
        final_text="done",
    )

    assert _extra_tool_calls_after_first_failure(run) == 2.0
    assert _extra_iterations_after_first_failure(run) == 2.0
    assert _extra_tokens_after_first_failure(run) == 35.0


@pytest.mark.asyncio
async def test_run_repeated_case_aggregates_recovery_efficiency_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules[__name__]
    profile = _ModelProfile(key="kimi", label="Kimi::test", model="kimi-test")
    lane = _Lane(name="cli_chain", skills=_CLI_CHAIN_SKILLS)
    case = next(item for item in _CASES if item.name == "chain_fallback")

    async def _fake_preflight(**_: object) -> _ProfilePreflight:
        return _ProfilePreflight(profile=profile)

    runs = [
        _Run(
            completion_metadata={
                "finish_reason": "stop",
                "trace_id": "t1",
                "usage": {"total_tokens": 40},
            },
            tool_start_events=[
                {
                    "data": {
                        "tool_call_id": "call-1",
                        "round_index": 1,
                        "tool_name": "houyi_local_cli_chain",
                    }
                },
                {
                    "data": {
                        "tool_call_id": "call-2",
                        "round_index": 2,
                        "tool_name": "houyi_local_cli_chain",
                    }
                },
            ],
            tool_result_events=[
                {
                    "data": {
                        "tool_call_id": "call-1",
                        "round_index": 1,
                        "result": {"success": False},
                    }
                },
                {"data": {"tool_call_id": "call-2", "round_index": 2, "result": {"success": True}}},
            ],
            tool_error_events=[],
            message_error_events=[],
            soft_tool_failures=[
                {"data": {"tool_call_id": "call-1", "round_index": 1, "result": {"success": False}}}
            ],
            iteration_events=[
                {"data": {"round_index": 1, "usage": {"total_tokens": 10}}},
                {"data": {"round_index": 2, "usage": {"total_tokens": 12}}},
            ],
            final_text="web_search skill.md answer ok",
        ),
        _Run(
            completion_metadata={
                "finish_reason": "stop",
                "trace_id": "t2",
                "usage": {"total_tokens": 30},
            },
            tool_start_events=[
                {
                    "data": {
                        "tool_call_id": "call-3",
                        "round_index": 1,
                        "tool_name": "houyi_local_cli_chain",
                    }
                }
            ],
            tool_result_events=[
                {"data": {"tool_call_id": "call-3", "round_index": 1, "result": {"success": True}}}
            ],
            tool_error_events=[],
            message_error_events=[],
            soft_tool_failures=[],
            iteration_events=[
                {"data": {"round_index": 1, "usage": {"total_tokens": 9}}},
            ],
            final_text="web_search skill.md answer ok",
        ),
    ]

    async def _fake_live_case(**_: object) -> _Run:
        return runs.pop(0)

    monkeypatch.setattr(module, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_settings_store", lambda: None)
    monkeypatch.setattr(module, "_run_profile_preflight", _fake_preflight)
    monkeypatch.setattr(module, "_run_live_case", _fake_live_case)

    aggregate = await _run_repeated_case(
        tmp_path=tmp_path,
        profile=profile,
        lane=lane,
        case=case,
        repeats=2,
    )

    assert aggregate.post_error_recovery_success_rate == 1.0
    assert aggregate.avg_extra_tool_calls_after_first_failure == 1.0
    assert aggregate.avg_extra_iterations_after_first_failure == 1.0
    assert aggregate.avg_extra_tokens_after_first_failure == 12.0
