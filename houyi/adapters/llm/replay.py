from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from houyi.interface.protocol.ir.checkpoint_ir import LLMCallLog
from houyi.interface.protocol.ir.tooling_ir import LLMToolCallOutputIR

logger = logging.getLogger(__name__)


def is_deterministic_replay(*, execution_metadata: Any) -> bool:
    if not isinstance(execution_metadata, dict):
        return False
    return execution_metadata.get("replay_mode") == "deterministic"


def build_prompt_cache_key(*, model: str | None, prompt_cache_key: str | None) -> str | None:
    if not prompt_cache_key:
        return None
    return f"{model or ''}:{prompt_cache_key}"


def get_cached_response(*, llm_cache: dict[str, str], cache_key: str | None) -> str | None:
    if not cache_key:
        return None
    return llm_cache.get(cache_key)


class ReplayDecisionKind(str, Enum):
    RECORDED_LLM_TEXT = "recorded_llm_text"
    RECORDED_TOOL_OUTPUT = "recorded_tool_output"
    CACHED_LLM_TEXT = "cached_llm_text"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    kind: ReplayDecisionKind
    llm_text: str | None = None
    tool_output: dict[str, Any] | None = None
    cache_key: str | None = None
    prompt_cache_key: str | None = None


def decide_replay(
    *,
    execution_metadata: Any,
    llm_call_logs: dict[str, list[LLMCallLog]],
    execution_id: str,
    node_id: str,
    llm_cache: dict[str, str] | None = None,
    model: str | None = None,
    prompt_cache_key: str | None = None,
) -> ReplayDecision:
    deterministic = is_deterministic_replay(execution_metadata=execution_metadata)
    if deterministic:
        recorded_text = get_recorded_llm_response(
            llm_call_logs=llm_call_logs,
            execution_id=execution_id,
            node_id=node_id,
        )
        if recorded_text:
            return ReplayDecision(
                kind=ReplayDecisionKind.RECORDED_LLM_TEXT,
                llm_text=recorded_text,
            )

        recorded_tool_output = get_recorded_tool_call_output(
            llm_call_logs=llm_call_logs,
            execution_id=execution_id,
            node_id=node_id,
        )
        if recorded_tool_output:
            return ReplayDecision(
                kind=ReplayDecisionKind.RECORDED_TOOL_OUTPUT,
                tool_output=recorded_tool_output,
            )

    if llm_cache is not None:
        cache_key = build_prompt_cache_key(model=model, prompt_cache_key=prompt_cache_key)
        cached_response = get_cached_response(llm_cache=llm_cache, cache_key=cache_key)
        if cached_response is not None:
            return ReplayDecision(
                kind=ReplayDecisionKind.CACHED_LLM_TEXT,
                llm_text=cached_response,
                cache_key=cache_key,
                prompt_cache_key=prompt_cache_key,
            )

    return ReplayDecision(kind=ReplayDecisionKind.NONE)


def record_llm_call(
    *,
    llm_call_logs: dict[str, list[LLMCallLog]],
    execution_id: str,
    node_id: str,
    model: str,
    prompt: str | list[dict[str, Any]],
    response: str,
    metadata: dict | None = None,
    timestamp: datetime | None = None,
) -> LLMCallLog:
    call_id = f"llm_{len(llm_call_logs.get(execution_id, []))}_{node_id}"

    llm_log = LLMCallLog(
        call_id=call_id,
        node_id=node_id,
        timestamp=timestamp or datetime.now(),
        model=model,
        prompt=prompt,
        response=response,
        metadata=metadata or {},
    )

    llm_call_logs.setdefault(execution_id, []).append(llm_log)
    logger.debug("Recorded LLM call: %s for node %s", call_id, node_id)
    return llm_log


def get_recorded_llm_response(
    *,
    llm_call_logs: dict[str, list[LLMCallLog]],
    execution_id: str,
    node_id: str,
) -> str | None:
    logs = llm_call_logs.get(execution_id, [])
    for log in logs:
        if log.node_id != node_id:
            continue
        metadata = log.metadata or {}
        if isinstance(metadata, dict) and (
            metadata.get("tool_call_output") is not None or metadata.get("tool_calls") is not None
        ):
            continue
        logger.info(
            "Found recorded response for node %s (call_id: %s)",
            node_id,
            log.call_id,
        )
        return log.response
    return None


def get_recorded_tool_call_output(
    *,
    llm_call_logs: dict[str, list[LLMCallLog]],
    execution_id: str,
    node_id: str,
) -> dict[str, Any] | None:
    logs = llm_call_logs.get(execution_id, [])
    for log in logs:
        if log.node_id != node_id:
            continue
        metadata = log.metadata or {}
        if not isinstance(metadata, dict):
            continue

        tool_output = metadata.get("tool_call_output")
        if tool_output is not None:
            return tool_output

        tool_calls = metadata.get("tool_calls")
        if tool_calls is None:
            continue

        messages: list[dict[str, Any]] = []
        if isinstance(log.prompt, list):
            messages = list(log.prompt)
        elif log.prompt:
            messages = [{"role": "user", "content": str(log.prompt)}]
        if log.response:
            messages.append({"role": "assistant", "content": log.response})

        output_payload = LLMToolCallOutputIR(
            content=log.response or "",
            tool_calls=tool_calls,
            finish_reason=metadata.get("finish_reason"),
            tool_finish_reason=metadata.get("tool_finish_reason"),
            tool_call_rounds=metadata.get("tool_call_rounds", 0),
            max_rounds_reached=metadata.get("max_rounds_reached", False),
            tool_errors=metadata.get("tool_errors", []),
            messages=messages,
            error=None,
        )
        return output_payload.model_dump(by_alias=True)

    return None
