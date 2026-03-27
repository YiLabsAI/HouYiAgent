"""Built-in local tools for workspace development workflows.

This module defines core local tools:
- houyi_read_file
- houyi_write_file
- houyi_find_files
- houyi_list_dir
- houyi_grep
- houyi_shell_exec
- houyi_local_cli
- houyi_local_cli_chain
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import os
import re
import shlex
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator

from houyi.domain.skill.policy import (
    ExecPerm,
    FilesystemPerm,
    InvocationPolicy,
    ModelAutoInvoke,
    Permissions,
    SideEffect,
)
from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec

DEFAULT_ENCODING = "utf-8"
DEFAULT_MAX_READ_CHARS = 200_000
DEFAULT_MAX_OUTPUT_CHARS = 50_000
DEFAULT_MAX_RESULTS = 200
DEFAULT_SHELL_TIMEOUT_SECONDS = 30
MAX_SHELL_TIMEOUT_SECONDS = 120

DANGEROUS_COMMAND_PATTERNS = (
    r"\brm\s+-rf\s+/$",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\|:&\s*;\s*\}",
)


class ToolResponse(BaseModel):
    success: bool
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ReadFileInput(BaseModel):
    path: str
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class WriteFileInput(BaseModel):
    path: str
    content: str
    create_parents: bool = True


class FindFilesInput(BaseModel):
    root_path: str = "."
    pattern: str = "*"
    search_mode: Literal["glob", "contains", "exact"] = "contains"
    iterative_subdirs: bool = False
    max_depth: int = Field(default=16, ge=0, le=64)
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class ListDirInput(BaseModel):
    path: str = "."
    recursive: bool = False
    max_entries: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class GrepInput(BaseModel):
    path: str = "."
    query: str
    case_sensitive: bool = False
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class ShellExecInput(BaseModel):
    command: str
    cwd: str | None = None
    timeout_seconds: int = Field(
        default=DEFAULT_SHELL_TIMEOUT_SECONDS, ge=1, le=MAX_SHELL_TIMEOUT_SECONDS
    )


class LocalCliInput(BaseModel):
    command: Literal["read", "list", "find", "grep"]
    path: str = "."
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    pattern: str | None = None
    query: str | None = None
    search_mode: Literal["glob", "contains", "exact"] = "contains"
    recursive: bool = False
    iterative_subdirs: bool = False
    case_sensitive: bool = False
    max_depth: int = Field(default=16, ge=0, le=64)
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)
    max_entries: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class LocalCliChainInput(BaseModel):
    mode: Literal["plan", "continue", "repair"] = Field(
        default="plan",
        description="Chain mode: initial plan, continue, or local repair.",
    )
    workflow_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional workflow identifier.",
    )
    continuation_token: str | None = Field(
        default=None,
        min_length=1,
        description="Opaque token from a previous chain run.",
    )
    resume_from_step_index: int | None = Field(
        default=None,
        ge=0,
        description="Continue from this step index.",
    )
    failed_step_index: int | None = Field(
        default=None,
        ge=0,
        description="Failed step index for repair.",
    )
    repair_action: (
        Literal[
            "replace_failed_step",
            "append_fallback_after_failed_step",
            "narrow_failed_step_arguments",
        ]
        | None
    ) = Field(
        default=None,
        description="Repair action for the failed step.",
    )
    replan_reason: str | None = Field(
        default=None,
        min_length=1,
        description="Reason for a full replan when repair is not enough.",
    )
    workflow: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Legacy chain workflow string for an already-decided multi-step local workflow. Prefer structured steps for staged narrowing, fallback, or repair. For unclear workflow selection or unverified paths, narrow with find/list/grep before read. Example: find path=houyi/skills pattern=SKILL.md | grep query=web | read start_line=1 end_line=20"
        ),
    )
    steps: list[LocalCliChainStepInput] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Preferred structured chain steps for a known multi-step workflow. Use this after deciding staged find/list/grep/read actions, instead of generic file probing. If candidates or paths are uncertain, verify with find/list/grep before the first read."
        ),
    )

    @model_validator(mode="after")
    def _validate_input_mode(self) -> LocalCliChainInput:
        if not (self.workflow or self.steps):
            raise ValueError("Provide workflow or steps")
        if self.mode in {"continue", "repair"} and not self.continuation_token:
            raise ValueError("continuation_token is required for continue or repair mode")
        if self.mode == "continue" and self.resume_from_step_index is None:
            raise ValueError("resume_from_step_index is required for continue mode")
        if self.mode == "repair":
            if self.failed_step_index is None:
                raise ValueError("failed_step_index is required for repair mode")
            if self.repair_action is None:
                raise ValueError("repair_action is required for repair mode")
        return self


class LocalCliChainStepInput(BaseModel):
    operator: Literal["&&", "||", "|", ";"] | None = Field(
        default=None,
        description="Operator applied before this step. Omit on the first step. If omitted on later steps, pipe semantics are used.",
    )
    command: Literal["read", "list", "find", "grep"] = Field(
        description="Atomic local action for this step."
    )
    path: str | None = Field(
        default=None,
        description="Workspace-relative path. Omit to use the projected path from a previous step when available.",
    )
    pattern: str | None = Field(default=None, description="Pattern for find steps.")
    query: str | None = Field(default=None, description="Search query for grep steps.")
    start_line: int | None = Field(default=None, ge=1, description="Start line for read steps.")
    end_line: int | None = Field(default=None, ge=1, description="End line for read steps.")
    search_mode: Literal["glob", "contains", "exact"] | None = Field(
        default=None,
        description="Search mode for find steps.",
    )
    recursive: bool | None = Field(default=None, description="Recursive listing for list steps.")
    iterative_subdirs: bool | None = Field(
        default=None,
        description="Whether find should iterate subdirectories breadth-first.",
    )
    case_sensitive: bool | None = Field(
        default=None,
        description="Case sensitivity for grep steps.",
    )
    max_depth: int | None = Field(default=None, ge=0, le=64, description="Maximum search depth.")
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum matches returned by find or grep.",
    )
    max_entries: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum entries returned by list.",
    )

    def to_chain_link(self, *, index: int) -> _ChainLink:
        args = self.model_dump(exclude_none=True, exclude={"operator", "command"})
        operator = None if index == 0 else (self.operator or "|")
        return _ChainLink(
            operator=operator,
            step=_ChainStep(command=self.command, args=args),
        )


class LocalCliReadInput(BaseModel):
    path: str = "."
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class LocalCliListInput(BaseModel):
    path: str = "."
    recursive: bool = False
    max_entries: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class LocalCliFindInput(BaseModel):
    path: str = "."
    pattern: str = "*"
    search_mode: Literal["glob", "contains", "exact"] = "contains"
    iterative_subdirs: bool = False
    max_depth: int = Field(default=16, ge=0, le=64)
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class LocalCliGrepInput(BaseModel):
    path: str = "."
    query: str
    case_sensitive: bool = False
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


_LOCAL_CLI_PROJECTED_TOOL_NAMES = {
    "read": "houyi_local_cli_read",
    "list": "houyi_local_cli_list",
    "find": "houyi_local_cli_find",
    "grep": "houyi_local_cli_grep",
}

_LOCAL_CLI_PROJECTED_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "read": LocalCliReadInput,
    "list": LocalCliListInput,
    "find": LocalCliFindInput,
    "grep": LocalCliGrepInput,
}

_LOCAL_CLI_PROJECTED_DESCRIPTIONS = {
    "read": "Read local file content with optional line range through the unified local CLI backend. Use this when the atomic action is already clear, not for deciding a broader workflow or strategy.",
    "list": "List local directory entries through the unified local CLI backend. Use this when the next atomic step is already clear, not as a substitute for workflow or skill selection.",
    "find": "Find local files by pattern through the unified local CLI backend. Use this when file discovery is the clear next atomic action, not when the model still needs to decide a broader workflow.",
    "grep": "Search local file contents by query through the unified local CLI backend. Use this when content search is the clear next atomic action, not for unresolved workflow-selection tasks.",
}

_LOCAL_CLI_CHAIN_OPERATORS = ("&&", "||", "|", ";")
_LOCAL_CLI_ATOMIC_COMMANDS = {"read", "list", "find", "grep"}
_LOCAL_CLI_CHAIN_POSITIONAL_ARG_KEYS = {
    "read": ("path",),
    "list": ("path",),
    "find": ("path", "pattern"),
    "grep": ("query",),
}


@dataclass(frozen=True)
class _ChainStep:
    command: Literal["read", "list", "find", "grep"]
    args: dict[str, Any]


@dataclass(frozen=True)
class _ChainLink:
    operator: str | None
    step: _ChainStep


@dataclass(frozen=True)
class _ProjectionContext:
    values: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class _ContinuationState:
    workflow_id: str
    mode: str
    frozen_success_steps: list[dict[str, Any]]
    projection: dict[str, Any]
    projection_error: str | None = None
    failed_step_index: int | None = None
    repair_scope: str = "retry_failed_step_only"


_CONTINUATION_TOKEN_PREFIX = "local_cli_chain:v1:"


def _encode_continuation_state(state: _ContinuationState) -> str:
    payload = {
        "workflow_id": state.workflow_id,
        "mode": state.mode,
        "frozen_success_steps": state.frozen_success_steps,
        "projection": state.projection,
        "projection_error": state.projection_error,
        "failed_step_index": state.failed_step_index,
        "repair_scope": state.repair_scope,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"{_CONTINUATION_TOKEN_PREFIX}{encoded}"


def _decode_continuation_state(token: str | None) -> _ContinuationState | None:
    raw = str(token or "").strip()
    if not raw.startswith(_CONTINUATION_TOKEN_PREFIX):
        return None
    encoded = raw[len(_CONTINUATION_TOKEN_PREFIX) :]
    if not encoded:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    frozen_success_steps = payload.get("frozen_success_steps")
    projection = payload.get("projection")
    if not isinstance(frozen_success_steps, list) or not isinstance(projection, dict):
        return None
    return _ContinuationState(
        workflow_id=str(payload.get("workflow_id") or "local_cli_chain"),
        mode=str(payload.get("mode") or "plan"),
        frozen_success_steps=[item for item in frozen_success_steps if isinstance(item, dict)],
        projection=projection,
        projection_error=(
            str(payload.get("projection_error"))
            if isinstance(payload.get("projection_error"), str)
            else None
        ),
        failed_step_index=(
            int(payload["failed_step_index"])
            if isinstance(payload.get("failed_step_index"), int)
            else None
        ),
        repair_scope=str(payload.get("repair_scope") or "retry_failed_step_only"),
    )


def _reused_step_entry(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator": step.get("operator"),
        "command": step.get("command"),
        "args": dict(step.get("args") or {}),
        "success": True,
        "reused": True,
    }


def _continuation_replan_result(
    *,
    workflow_id: str,
    continuation_token: str,
    message: str,
    frozen_success_steps: list[dict[str, Any]],
    failed_step_index: int | None,
) -> dict[str, Any]:
    return ToolResponse(
        success=False,
        message=message,
        data={
            "workflow_id": workflow_id,
            "continuation_token": continuation_token,
            "failure_kind": "replan_required",
            "recovery_hint": "Do not rewrite already successful earlier steps. Continue from the failed step or explicitly declare replan_reason when a full replanning is necessary.",
            "repair_scope": "retry_failed_step_only",
            "replan_required": True,
            "frozen_success_steps": frozen_success_steps,
            "failed_step_index": failed_step_index,
        },
    ).model_dump()


def _build_continuation_state(
    *,
    workflow_id: str,
    mode: str,
    projection: _ProjectionContext,
    frozen_success_steps: list[dict[str, Any]],
    failed_step_index: int | None,
) -> _ContinuationState:
    return _ContinuationState(
        workflow_id=workflow_id,
        mode=mode,
        frozen_success_steps=frozen_success_steps,
        projection=dict(projection.values),
        projection_error=projection.error,
        failed_step_index=failed_step_index,
        repair_scope="retry_failed_step_only",
    )


def _prepare_chain_execution_context(
    *,
    normalized_mode: str,
    input_workflow_id: str,
    continuation_token: str,
    resume_from_step_index: int | None,
    failed_step_index: int | None,
) -> tuple[
    str,
    list[dict[str, Any]],
    _ProjectionContext,
    dict[str, Any] | None,
]:
    active_workflow_id = input_workflow_id or "local_cli_chain"
    decoded_state = _decode_continuation_state(continuation_token)
    if decoded_state is not None and not input_workflow_id:
        active_workflow_id = decoded_state.workflow_id

    if normalized_mode == "continue" and decoded_state is not None:
        prior_success_count = len(decoded_state.frozen_success_steps)
        if resume_from_step_index is not None and resume_from_step_index < prior_success_count:
            return (
                active_workflow_id,
                [],
                _ProjectionContext(values={}),
                _continuation_replan_result(
                    workflow_id=active_workflow_id,
                    continuation_token=continuation_token,
                    message="continue mode cannot resume before frozen successful steps; explicit replan is required",
                    frozen_success_steps=decoded_state.frozen_success_steps,
                    failed_step_index=decoded_state.failed_step_index,
                ),
            )

    if normalized_mode == "repair" and decoded_state is not None:
        prior_success_count = len(decoded_state.frozen_success_steps)
        if failed_step_index is not None and failed_step_index < prior_success_count:
            return (
                active_workflow_id,
                [],
                _ProjectionContext(values={}),
                _continuation_replan_result(
                    workflow_id=active_workflow_id,
                    continuation_token=continuation_token,
                    message="repair mode cannot modify frozen successful steps; explicit replan is required",
                    frozen_success_steps=decoded_state.frozen_success_steps,
                    failed_step_index=failed_step_index,
                ),
            )

    reused_steps = (
        list(decoded_state.frozen_success_steps)
        if decoded_state is not None and normalized_mode in {"continue", "repair"}
        else []
    )
    projection = (
        _ProjectionContext(
            values=dict(decoded_state.projection),
            error=decoded_state.projection_error,
        )
        if decoded_state is not None and normalized_mode in {"continue", "repair"}
        else _ProjectionContext(values={})
    )
    return active_workflow_id, reused_steps, projection, None


async def _execute_chain_links(
    *,
    links: list[_ChainLink],
    reused_steps: list[dict[str, Any]],
    projection: _ProjectionContext,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    _ProjectionContext,
    dict[str, Any] | None,
    dict[str, Any] | None,
    bool,
]:
    executed_steps: list[dict[str, Any]] = [_reused_step_entry(step) for step in reused_steps]
    last_result: dict[str, Any] | None = None
    last_success = True
    failure_summary: dict[str, Any] | None = None
    frozen_success_steps: list[dict[str, Any]] = list(reused_steps)
    active_projection = projection

    for link in links:
        step_index = len(frozen_success_steps) if last_success else len(executed_steps)
        operator = link.operator
        if operator == "&&" and not last_success:
            executed_steps.append(
                {
                    "operator": operator,
                    "command": link.step.command,
                    "skipped": True,
                    "step_index": step_index,
                }
            )
            continue
        if operator == "||" and last_success:
            executed_steps.append(
                {
                    "operator": operator,
                    "command": link.step.command,
                    "skipped": True,
                    "step_index": step_index,
                }
            )
            continue

        result = await _execute_chain_step(link.step, active_projection)
        last_result = result
        last_success = _is_chain_success(result)
        if last_success:
            active_projection = _extract_projection(result)
            failure_summary = None
            frozen_success_steps.append(
                {
                    "step_index": step_index,
                    "operator": operator,
                    "command": link.step.command,
                    "args": dict(link.step.args),
                }
            )
        else:
            failure_summary = _chain_failure_summary(
                step_index=step_index,
                step=link.step,
                result=result,
            )
        executed_steps.append(
            {
                "step_index": step_index,
                "operator": operator,
                "command": link.step.command,
                "args": link.step.args,
                "success": last_success,
                "projection": dict(active_projection.values),
                "projection_error": active_projection.error,
                "result": result,
            }
        )

    return (
        executed_steps,
        frozen_success_steps,
        active_projection,
        last_result,
        failure_summary,
        last_success,
    )


def _coerce_chain_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", raw):
        with_value = int(raw)
        return with_value
    return raw


def _flush_chain_buffer(tokens: list[str], buffer: list[str]) -> None:
    text = "".join(buffer).strip()
    if text:
        tokens.append(text)
    buffer.clear()


def _match_chain_operator(workflow: str, index: int) -> tuple[str | None, int]:
    two_chars = workflow[index : index + 2]
    if two_chars in {"&&", "||"}:
        return two_chars, 2
    char = workflow[index]
    if char in {"|", ";"}:
        return char, 1
    return None, 0


def _tokenize_chain_workflow(workflow: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    quote_char: str | None = None
    index = 0
    while index < len(workflow):
        char = workflow[index]
        if quote_char is not None:
            buffer.append(char)
            if char == quote_char:
                quote_char = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote_char = char
            buffer.append(char)
            index += 1
            continue
        operator, width = _match_chain_operator(workflow, index)
        if operator is not None:
            _flush_chain_buffer(tokens, buffer)
            tokens.append(operator)
            index += width
            continue
        buffer.append(char)
        index += 1
    if quote_char is not None:
        raise ValueError("workflow has an unterminated quoted value")
    _flush_chain_buffer(tokens, buffer)
    return tokens


def _split_chain_step_tokens(step_text: str) -> list[str]:
    stripped = step_text.strip()
    function_match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)", stripped)
    if function_match is None:
        return shlex.split(step_text)
    command, raw_args = function_match.groups()
    lexer = shlex.shlex(raw_args, posix=True)
    lexer.whitespace_split = True
    lexer.whitespace += ","
    lexer.commenters = ""
    return [command, *list(lexer)]


def _parse_chain_step(step_text: str) -> _ChainStep:
    tokens = _split_chain_step_tokens(step_text)
    if not tokens:
        raise ValueError("workflow step is empty")
    command = tokens[0]
    if command not in _LOCAL_CLI_ATOMIC_COMMANDS:
        raise ValueError(f"unsupported chain command: {command}")
    args: dict[str, Any] = {}
    positional_keys = _LOCAL_CLI_CHAIN_POSITIONAL_ARG_KEYS.get(command, ())
    positional_index = 0
    for token in tokens[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            if not key:
                raise ValueError(f"invalid chain argument: {token}")
            args[key] = _coerce_chain_value(value)
            continue
        while positional_index < len(positional_keys) and positional_keys[positional_index] in args:
            positional_index += 1
        if positional_index >= len(positional_keys):
            raise ValueError(f"invalid chain argument: {token}")
        args[positional_keys[positional_index]] = _coerce_chain_value(token)
        positional_index += 1
    return _ChainStep(
        command=cast(Literal["read", "list", "find", "grep"], command),
        args=args,
    )


def _parse_chain_workflow(workflow: str) -> list[_ChainLink]:
    tokens = _tokenize_chain_workflow(workflow)
    if not tokens:
        raise ValueError("workflow is empty")
    links: list[_ChainLink] = []
    pending_operator: str | None = None
    expecting_step = True
    for token in tokens:
        if token in _LOCAL_CLI_CHAIN_OPERATORS:
            if expecting_step:
                raise ValueError(f"unexpected operator: {token}")
            pending_operator = token
            expecting_step = True
            continue
        if not expecting_step:
            raise ValueError(f"missing operator before step: {token}")
        links.append(_ChainLink(operator=pending_operator, step=_parse_chain_step(token)))
        pending_operator = None
        expecting_step = False
    if expecting_step:
        raise ValueError("workflow cannot end with an operator")
    return links


def _chain_links_from_structured_steps(steps: list[LocalCliChainStepInput]) -> list[_ChainLink]:
    return [step.to_chain_link(index=index) for index, step in enumerate(steps)]


def _normalize_structured_steps(
    steps: Sequence[LocalCliChainStepInput | dict[str, Any]],
) -> list[LocalCliChainStepInput]:
    normalized_steps: list[LocalCliChainStepInput] = []
    for step in steps:
        if isinstance(step, LocalCliChainStepInput):
            normalized_steps.append(step)
            continue
        normalized_steps.append(LocalCliChainStepInput.model_validate(step))
    return normalized_steps


def _collect_projection_candidate_paths(matches: Any) -> list[str]:
    if not isinstance(matches, list):
        return []
    candidate_paths: list[str] = []
    for item in matches:
        if isinstance(item, str) and item and item not in candidate_paths:
            candidate_paths.append(item)
            continue
        if not isinstance(item, dict):
            continue
        match_path = item.get("path")
        if isinstance(match_path, str) and match_path and match_path not in candidate_paths:
            candidate_paths.append(match_path)
    return candidate_paths


def _projection_context_from_candidates(
    *, projection: dict[str, Any], candidate_paths: list[str]
) -> _ProjectionContext:
    if len(candidate_paths) == 1:
        projection.setdefault("path", candidate_paths[0])
        return _ProjectionContext(values=projection)
    if len(candidate_paths) > 1:
        return _ProjectionContext(
            values=projection,
            error=(
                "projection requires a unique path candidate; add a narrower step or explicit path"
            ),
        )
    return _ProjectionContext(values=projection)


def _extract_projection(result: dict[str, Any]) -> _ProjectionContext:
    data = result.get("data")
    if not isinstance(data, dict):
        return _ProjectionContext(values={})
    projection: dict[str, Any] = {}
    for key in ("path", "root_path"):
        value = data.get(key)
        if isinstance(value, str) and value:
            projection[key] = value
    candidate_paths = _collect_projection_candidate_paths(data.get("matches"))
    return _projection_context_from_candidates(
        projection=projection,
        candidate_paths=candidate_paths,
    )


def _is_chain_success(result: dict[str, Any]) -> bool:
    if result.get("success") is False:
        return False
    data = result.get("data")
    if not isinstance(data, dict):
        return True
    matches = data.get("matches")
    if isinstance(matches, list) and not matches:
        return False
    entries = data.get("entries")
    if isinstance(entries, list) and not entries:
        return False
    content = data.get("content")
    return not (isinstance(content, str) and not content and data.get("line_count") == 0)


def _apply_projection(
    command: str, args: dict[str, Any], projection: _ProjectionContext
) -> dict[str, Any]:
    merged = dict(args)
    if (
        projection.error
        and command == "read"
        and "path" not in merged
        and "root_path" not in merged
    ):
        raise ValueError(projection.error)
    if command == "find" and "path" not in merged and "root_path" not in merged:
        if isinstance(projection.values.get("path"), str):
            merged["path"] = projection.values["path"]
        elif isinstance(projection.values.get("root_path"), str):
            merged["path"] = projection.values["root_path"]
    if command in {"read", "list", "grep"} and "path" not in merged:
        projected_path = projection.values.get("path") or projection.values.get("root_path")
        if isinstance(projected_path, str):
            merged["path"] = projected_path
    return merged


def _projection_failure_result(
    *, step: _ChainStep, projection: _ProjectionContext, message: str
) -> dict[str, Any]:
    return ToolResponse(
        success=False,
        message=message,
        data={
            "failure_kind": "projection_failed",
            "recovery_hint": "Retry only the failing step. Add an explicit path=... or insert a narrower find/grep step before piping, while keeping already successful earlier steps unchanged.",
            "command": step.command,
            "args": step.args,
            "projection": dict(projection.values),
        },
    ).model_dump()


def _chain_parse_failure_result(message: str) -> dict[str, Any]:
    lowered = message.lower()
    failure_kind = "invalid_chain_workflow"
    recovery_hint = "Rewrite only the malformed step. Use read/list/find/grep steps joined by |, &&, ||, or ; with key=value arguments, or switch to structured steps when the workflow is already known."
    if lowered.startswith("unsupported chain command:"):
        failure_kind = "unsupported_chain_command"
        recovery_hint = "Replace only the unsupported step with read, list, find, or grep. Example: find path=houyi/skills pattern=SKILL.md | read start_line=1 end_line=20"
    elif lowered.startswith("invalid chain argument:"):
        failure_kind = "invalid_chain_argument"
        recovery_hint = "Correct only the failing step arguments. Use key=value arguments or function-style steps like read(path=..., start_line=1). A single positional value is only accepted for read/list as path, find as path then pattern, and grep as query. Prefer structured steps when the workflow is already decided."
    return ToolResponse(
        success=False,
        message=message,
        data={
            "failure_kind": failure_kind,
            "recovery_hint": recovery_hint,
            "repair_scope": "retry_failed_step_only",
            "accepted_commands": sorted(_LOCAL_CLI_ATOMIC_COMMANDS),
        },
    ).model_dump()


def _chain_failure_summary(
    *,
    step_index: int,
    step: _ChainStep,
    result: dict[str, Any],
) -> dict[str, Any]:
    def _missing_read_recovery_template(failed_args: dict[str, Any]) -> dict[str, Any] | None:
        raw_path = failed_args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        normalized_path = raw_path.strip().replace("\\", "/")
        path_parts = [part for part in normalized_path.split("/") if part and part != "."]
        if not path_parts:
            return None
        filename = path_parts[-1]
        parent_parts = path_parts[:-1]
        search_root_parts = parent_parts[:-1] if len(parent_parts) >= 2 else parent_parts
        search_root = "/".join(search_root_parts) if search_root_parts else "."

        read_step: dict[str, Any] = {
            "command": "read",
            "path_from_previous_find": True,
        }
        start_line = failed_args.get("start_line")
        end_line = failed_args.get("end_line")
        if isinstance(start_line, int) and start_line >= 1:
            read_step["start_line"] = start_line
        if isinstance(end_line, int) and end_line >= 1:
            read_step["end_line"] = end_line

        return {
            "reason": "read_file_not_found",
            "steps": [
                {
                    "command": "find",
                    "path": search_root,
                    "pattern": filename,
                    "search_mode": "exact",
                },
                read_step,
            ],
        }

    data = result.get("data")
    message = str(result.get("message") or "")
    failure_kind = "step_execution_failed"
    recovery_hint = "Retry only the failing step with corrected arguments or a narrower input. Keep already successful earlier steps unchanged."
    recovery_template: dict[str, Any] | None = None
    if isinstance(data, dict):
        raw_failure_kind = data.get("failure_kind")
        if isinstance(raw_failure_kind, str) and raw_failure_kind:
            failure_kind = raw_failure_kind
        raw_recovery_hint = data.get("recovery_hint")
        if isinstance(raw_recovery_hint, str) and raw_recovery_hint:
            recovery_hint = raw_recovery_hint
    if step.command == "read" and message.startswith("File not found:"):
        recovery_hint = "Retry only the failing read step. Do not guess another path directly. Prefer the provided recovery_step_template as the default repair path: first run the suggested find step, then read from its exact match. Only add extra narrowing if that first find still returns multiple candidates. Keep already successful earlier steps unchanged."
        recovery_template = _missing_read_recovery_template(step.args)
    return {
        "failure_kind": failure_kind,
        "recovery_hint": recovery_hint,
        "recovery_step_template": recovery_template,
        "repair_scope": "retry_failed_step_only",
        "failed_step_index": step_index,
        "failed_command": step.command,
        "failed_args": dict(step.args),
    }


async def _execute_chain_step(step: _ChainStep, projection: _ProjectionContext) -> dict[str, Any]:
    try:
        args = _apply_projection(step.command, step.args, projection)
    except ValueError as exc:
        return _projection_failure_result(step=step, projection=projection, message=str(exc))
    if step.command == "read":
        return await _local_cli_executor(
            command="read",
            path=str(args.get("path", ".")),
            start_line=args.get("start_line"),
            end_line=args.get("end_line"),
        )
    if step.command == "list":
        return await _local_cli_executor(
            command="list",
            path=str(args.get("path", ".")),
            recursive=bool(args.get("recursive", False)),
            max_entries=int(args.get("max_entries", DEFAULT_MAX_RESULTS)),
        )
    if step.command == "find":
        path = args.pop("root_path", args.get("path", "."))
        search_mode = cast(
            Literal["glob", "contains", "exact"],
            str(args.get("search_mode", "contains")),
        )
        return await _local_cli_executor(
            command="find",
            path=str(path),
            pattern=str(args.get("pattern") or "*"),
            search_mode=search_mode,
            iterative_subdirs=bool(args.get("iterative_subdirs", False)),
            max_depth=int(args.get("max_depth", 16)),
            max_results=int(args.get("max_results", DEFAULT_MAX_RESULTS)),
        )
    return await _local_cli_executor(
        command="grep",
        path=str(args.get("path", ".")),
        query=str(args.get("query") or ""),
        case_sensitive=bool(args.get("case_sensitive", False)),
        max_results=int(args.get("max_results", DEFAULT_MAX_RESULTS)),
    )


async def _local_cli_chain_executor(
    *,
    mode: str = "plan",
    workflow_id: str | None = None,
    continuation_token: str | None = None,
    resume_from_step_index: int | None = None,
    failed_step_index: int | None = None,
    repair_action: str | None = None,
    replan_reason: str | None = None,
    workflow: str | None = None,
    steps: Sequence[LocalCliChainStepInput | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    input_mode = "workflow"
    try:
        if steps:
            links = _chain_links_from_structured_steps(_normalize_structured_steps(steps))
            input_mode = "steps"
        elif workflow:
            links = _parse_chain_workflow(workflow)
        else:
            return _chain_parse_failure_result("Provide workflow or steps")
    except ValueError as exc:
        return _chain_parse_failure_result(str(exc))

    normalized_mode = str(mode or "plan").strip().lower() or "plan"
    input_workflow_id = str(workflow_id or "").strip()
    incoming_continuation_token = str(continuation_token or "").strip()
    (
        active_workflow_id,
        reused_steps,
        projection,
        early_result,
    ) = _prepare_chain_execution_context(
        normalized_mode=normalized_mode,
        input_workflow_id=input_workflow_id,
        continuation_token=incoming_continuation_token,
        resume_from_step_index=resume_from_step_index,
        failed_step_index=failed_step_index,
    )
    if early_result is not None:
        return early_result

    (
        executed_steps,
        frozen_success_steps,
        projection,
        last_result,
        failure_summary,
        last_success,
    ) = await _execute_chain_links(
        links=links,
        reused_steps=reused_steps,
        projection=projection,
    )
    if last_result is None:
        return ToolResponse(
            success=False, message="workflow produced no executable steps"
        ).model_dump()

    continuation_state = _build_continuation_state(
        workflow_id=active_workflow_id,
        mode=normalized_mode,
        projection=projection,
        frozen_success_steps=frozen_success_steps,
        failed_step_index=(failure_summary or {}).get("failed_step_index"),
    )
    active_continuation_token = _encode_continuation_state(continuation_state)
    return ToolResponse(
        success=last_success,
        data={
            "mode": normalized_mode,
            "workflow_id": active_workflow_id,
            "continuation_token": active_continuation_token,
            "input_continuation_token": incoming_continuation_token,
            "resume_from_step_index": resume_from_step_index,
            "failed_step_index": failed_step_index,
            "repair_action": repair_action,
            "replan_reason": replan_reason,
            "workflow": workflow or "",
            "input_mode": input_mode,
            "steps": executed_steps,
            "frozen_success_steps": frozen_success_steps,
            "reused_step_count": len(reused_steps),
            "replan_required": False,
            "repair_scope": "retry_failed_step_only",
            "final": last_result,
            **(failure_summary or {}),
            "truncated": False,
        },
    ).model_dump()


def _workspace_root() -> Path:
    configured = (os.getenv("HOUYI_WORKSPACE_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def _resolve_workspace_path(path_value: str, *, for_creation: bool = False) -> Path:
    base = _workspace_root()
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()

    target = candidate if not for_creation else candidate.parent
    if target != base and base not in target.parents:
        raise ValueError(f"Path must stay within workspace root: {base}")
    return candidate


def _trim_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _iter_bfs_entries(root: Path, max_depth: int) -> list[Path]:
    discovered: list[Path] = []
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    while queue:
        current, depth = queue.popleft()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            discovered.append(entry)
            if entry.is_dir() and depth < max_depth:
                queue.append((entry, depth + 1))
    return discovered


def _iter_files_under(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for item in root.rglob("*"):
        if item.is_file():
            yield item


def _iter_file_lines(file_path: Path) -> Iterator[tuple[int, str]]:
    with file_path.open("r", encoding=DEFAULT_ENCODING, errors="ignore") as handle:
        for index, line in enumerate(handle, start=1):
            yield index, line.rstrip("\r\n")


async def _read_file_executor(
    *,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    try:
        file_path = _resolve_workspace_path(path)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if not file_path.is_file():
        return ToolResponse(success=False, message=f"File not found: {file_path}").model_dump()

    text = file_path.read_text(encoding=DEFAULT_ENCODING)
    lines = text.splitlines()
    if start_line is not None or end_line is not None:
        start = max((start_line or 1) - 1, 0)
        end = end_line or len(lines)
        if end < start + 1:
            return ToolResponse(
                success=False, message="end_line must be greater than or equal to start_line"
            ).model_dump()
        selected = lines[start:end]
    else:
        selected = lines

    content = _trim_text("\n".join(selected), DEFAULT_MAX_READ_CHARS)
    return ToolResponse(
        success=True,
        data={
            "path": str(file_path),
            "content": content,
            "line_count": len(selected),
            "truncated": len(content) >= DEFAULT_MAX_READ_CHARS,
        },
    ).model_dump()


async def _write_file_executor(
    *,
    path: str,
    content: str,
    create_parents: bool = True,
) -> dict[str, Any]:
    try:
        file_path = _resolve_workspace_path(path, for_creation=True)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if create_parents:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(content, encoding=DEFAULT_ENCODING)
    return ToolResponse(
        success=True,
        data={
            "path": str(file_path),
            "bytes_written": len(content.encode(DEFAULT_ENCODING)),
        },
    ).model_dump()


def _resolve_search_mode(*, pattern: str, search_mode: str) -> str:
    has_glob_meta = any(ch in pattern for ch in "*?[")
    return "glob" if (search_mode == "contains" and has_glob_meta) else search_mode


def _collect_search_matches(
    *,
    iterator: Any,
    matcher: Any,
    root: Path,
    max_results: int,
) -> tuple[list[str], set[str]]:
    matches: list[str] = []
    searched_dirs: set[str] = {str(root)}
    for candidate in iterator:
        if candidate.is_dir():
            searched_dirs.add(str(candidate))
        if matcher(candidate.name):
            matches.append(str(candidate))
            if len(matches) >= max_results:
                break
    return matches, searched_dirs


async def _find_files_executor(
    *,
    root_path: str = ".",
    pattern: str = "*",
    search_mode: Literal["glob", "contains", "exact"] = "contains",
    iterative_subdirs: bool = False,
    max_depth: int = 16,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    effective_mode = _resolve_search_mode(pattern=pattern, search_mode=search_mode)

    def _matches(candidate_name: str) -> bool:
        if effective_mode == "exact":
            return candidate_name.lower() == pattern.lower()
        if effective_mode == "contains":
            return pattern.lower() in candidate_name.lower()
        return fnmatch.fnmatch(candidate_name, pattern)

    def _iter_candidates() -> Any:
        if not iterative_subdirs:
            return root.rglob("*")
        return _iter_bfs_entries(root, max_depth)

    try:
        root = _resolve_workspace_path(root_path)
    except ValueError as exc:
        return ToolResponse(
            success=False,
            message=f"{exc}. Use a relative path under workspace root (e.g. '.' or 'subdir').",
        ).model_dump()
    if not root.exists() or not root.is_dir():
        return ToolResponse(success=False, message=f"Directory not found: {root}").model_dump()

    matches, searched_dirs = _collect_search_matches(
        iterator=_iter_candidates(),
        matcher=_matches,
        root=root,
        max_results=max_results,
    )

    return ToolResponse(
        success=True,
        data={
            "root_path": str(root),
            "pattern": pattern,
            "search_mode": search_mode,
            "effective_mode": effective_mode,
            "iterative_subdirs": iterative_subdirs,
            "max_depth": max_depth,
            "searched_dirs": sorted(searched_dirs),
            "matches": matches,
            "truncated": len(matches) >= max_results,
        },
    ).model_dump()


async def _list_dir_executor(
    *,
    path: str = ".",
    recursive: bool = False,
    max_entries: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    try:
        directory = _resolve_workspace_path(path)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if not directory.exists() or not directory.is_dir():
        return ToolResponse(success=False, message=f"Directory not found: {directory}").model_dump()

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    entries: list[dict[str, Any]] = []
    for entry in iterator:
        entries.append({"path": str(entry), "is_dir": entry.is_dir()})
        if len(entries) >= max_entries:
            break

    return ToolResponse(
        success=True,
        data={
            "path": str(directory),
            "entries": entries,
            "recursive": recursive,
            "truncated": len(entries) >= max_entries,
        },
    ).model_dump()


async def _grep_executor(
    *,
    path: str,
    query: str,
    case_sensitive: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    try:
        root = _resolve_workspace_path(path)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if not root.exists():
        return ToolResponse(success=False, message=f"Path not found: {root}").model_dump()

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error as exc:
        return ToolResponse(success=False, message=f"Invalid regex: {exc}").model_dump()

    files = _iter_files_under(root)
    results: list[dict[str, Any]] = []
    for file_path in files:
        for index, line in _iter_file_lines(file_path):
            if pattern.search(line):
                results.append({"path": str(file_path), "line": index, "text": line})
                if len(results) >= max_results:
                    return ToolResponse(
                        success=True,
                        data={"matches": results, "truncated": True},
                    ).model_dump()

    return ToolResponse(success=True, data={"matches": results, "truncated": False}).model_dump()


def _validate_command(command: str) -> None:
    normalized = command.strip()
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, normalized):
            raise ValueError(f"Blocked dangerous command pattern: {pattern}")


async def _shell_exec_executor(
    *,
    command: str,
    cwd: str | None = None,
    timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    effective_timeout_seconds = int(min(timeout_seconds, MAX_SHELL_TIMEOUT_SECONDS))
    try:
        _validate_command(command)
        working_dir = _resolve_workspace_path(cwd or ".")
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()

    started_at = asyncio.get_running_loop().time()
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(working_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=float(effective_timeout_seconds),
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        duration_ms = round((asyncio.get_running_loop().time() - started_at) * 1000, 2)
        return ToolResponse(
            success=False,
            message=f"Command timed out after {effective_timeout_seconds}s",
            data={
                "command": command,
                "cwd": str(working_dir),
                "timed_out": True,
                "timeout_seconds": effective_timeout_seconds,
                "duration_ms": duration_ms,
                "retry_count": 0,
            },
        ).model_dump()

    duration_ms = round((asyncio.get_running_loop().time() - started_at) * 1000, 2)
    stdout_text = _trim_text(
        (stdout_bytes or b"").decode(DEFAULT_ENCODING, errors="ignore"), DEFAULT_MAX_OUTPUT_CHARS
    )
    stderr_text = _trim_text(
        (stderr_bytes or b"").decode(DEFAULT_ENCODING, errors="ignore"), DEFAULT_MAX_OUTPUT_CHARS
    )
    success = process.returncode == 0
    return ToolResponse(
        success=success,
        message="" if success else f"Command exited with code {process.returncode}",
        data={
            "command": command,
            "cwd": str(working_dir),
            "returncode": process.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_truncated": len(stdout_text) >= DEFAULT_MAX_OUTPUT_CHARS,
            "stderr_truncated": len(stderr_text) >= DEFAULT_MAX_OUTPUT_CHARS,
            "timed_out": False,
            "timeout_seconds": effective_timeout_seconds,
            "duration_ms": duration_ms,
            "retry_count": 0,
        },
    ).model_dump()


async def _local_cli_executor(
    *,
    command: Literal["read", "list", "find", "grep"],
    path: str = ".",
    start_line: int | None = None,
    end_line: int | None = None,
    pattern: str | None = None,
    query: str | None = None,
    search_mode: Literal["glob", "contains", "exact"] = "contains",
    recursive: bool = False,
    iterative_subdirs: bool = False,
    case_sensitive: bool = False,
    max_depth: int = 16,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_entries: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    if command == "read":
        return await _read_file_executor(path=path, start_line=start_line, end_line=end_line)
    if command == "list":
        return await _list_dir_executor(path=path, recursive=recursive, max_entries=max_entries)
    if command == "find":
        effective_pattern = pattern or "*"
        return await _find_files_executor(
            root_path=path,
            pattern=effective_pattern,
            search_mode=search_mode,
            iterative_subdirs=iterative_subdirs,
            max_depth=max_depth,
            max_results=max_results,
        )
    effective_query = query or pattern
    if not effective_query:
        return ToolResponse(
            success=False,
            message="query is required for grep command",
        ).model_dump()
    return await _grep_executor(
        path=path,
        query=effective_query,
        case_sensitive=case_sensitive,
        max_results=max_results,
    )


def make_local_cli_projected_executor(
    command: Literal["read", "list", "find", "grep"],
) -> Any:
    """Build a projected executor that binds one command onto the shared CLI backend."""

    async def _executor(**kwargs: Any) -> dict[str, Any]:
        return await _local_cli_executor(command=command, **kwargs)

    return _executor


def build_local_cli_projected_skills() -> list[SkillSpec]:
    """Build command-aware Local CLI facades over the unified backend."""

    projected_skills: list[SkillSpec] = []
    for command in ("read", "list", "find", "grep"):
        projected_skills.append(
            SkillSpec(
                name=_LOCAL_CLI_PROJECTED_TOOL_NAMES[command],
                description=_LOCAL_CLI_PROJECTED_DESCRIPTIONS[command],
                input_schema=_LOCAL_CLI_PROJECTED_INPUT_MODELS[command],
                output_schema=ToolResponse,
                executor=make_local_cli_projected_executor(command),
                is_core=True,
                invocation_policy=_policy_allow_filesystem_read(),
                permissions=Permissions(filesystem=FilesystemPerm(read=True)),
                metadata={
                    "tags": ["cli", "projected", command, "workspace"],
                    "local_cli_projected": True,
                    "local_cli_command": command,
                    "local_cli_backend": "houyi_local_cli",
                },
            )
        )
    return projected_skills


def _policy_allow_filesystem_read() -> InvocationPolicy:
    return InvocationPolicy(
        model_auto_invoke=ModelAutoInvoke.ALLOW, side_effect=SideEffect.FILESYSTEM
    )


def _policy_allow_with_consent(side_effect: SideEffect) -> InvocationPolicy:
    return InvocationPolicy(
        model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
        side_effect=side_effect,
    )


def build_builtin_local_tools() -> list[SkillSpec]:
    """Build built-in local tool specs."""
    return [
        SkillSpec(
            name="houyi_read_file",
            description="Read local file content with optional line range. Use this when the target path is already verified. If the path or file choice is still uncertain, narrow with find, list, or grep before read.",
            input_schema=ReadFileInput,
            output_schema=ToolResponse,
            executor=_read_file_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["file", "read", "workspace"]},
        ),
        SkillSpec(
            name="houyi_write_file",
            description="Write content to a local file under workspace.",
            input_schema=WriteFileInput,
            output_schema=ToolResponse,
            executor=_write_file_executor,
            is_core=True,
            invocation_policy=_policy_allow_with_consent(SideEffect.FILESYSTEM),
            permissions=Permissions(filesystem=FilesystemPerm(write=True)),
            metadata={"tags": ["file", "write", "workspace"]},
        ),
        SkillSpec(
            name="houyi_find_files",
            description="Find files by pattern under a workspace directory. Prefer this when the path, skill location, or file choice is still unclear before reading a file directly.",
            input_schema=FindFilesInput,
            output_schema=ToolResponse,
            executor=_find_files_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["file", "search", "glob"]},
        ),
        SkillSpec(
            name="houyi_list_dir",
            description="List directory entries under workspace. Use this to verify workspace structure or candidate folders before guessing a read path.",
            input_schema=ListDirInput,
            output_schema=ToolResponse,
            executor=_list_dir_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["directory", "list", "workspace"]},
        ),
        SkillSpec(
            name="houyi_grep",
            description="Search text patterns in files under workspace. Prefer this to narrow multiple candidates before choosing which file to read.",
            input_schema=GrepInput,
            output_schema=ToolResponse,
            executor=_grep_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["search", "grep", "regex"]},
        ),
        SkillSpec(
            name="houyi_shell_exec",
            description="Execute a shell command in workspace with timeout and output limits.",
            input_schema=ShellExecInput,
            output_schema=ToolResponse,
            executor=_shell_exec_executor,
            is_core=True,
            invocation_policy=_policy_allow_with_consent(SideEffect.EXEC),
            permissions=Permissions(exec=ExecPerm(enabled=True)),
            metadata={"tags": ["shell", "command", "execution"]},
        ),
        SkillSpec(
            name="houyi_local_cli",
            description="Run a read-only local CLI-style command for read, list, find, or grep workflows. If the path or target file is unverified, prefer find, list, or grep before read, and keep narrowing when multiple candidates remain.",
            input_schema=LocalCliInput,
            output_schema=ToolResponse,
            executor=_local_cli_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["cli", "read", "list", "find", "grep", "workspace"]},
        ),
        SkillSpec(
            name="houyi_local_cli_chain",
            description="Run a controlled read-only local CLI workflow chain over read, list, find, and grep actions. Prefer this for a staged local workflow, fallback, or repair. If a fallback path is unverified, do not begin with read; verify with find, list, or grep first.",
            input_schema=LocalCliChainInput,
            output_schema=ToolResponse,
            executor=_local_cli_chain_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={
                "tags": ["cli", "chain", "workflow", "read", "list", "find", "grep", "workspace"]
            },
        ),
    ]


def register_builtin_local_tools(registry: SkillRegistry) -> list[str]:
    """Register built-in local tools as core skills."""
    registered: list[str] = []
    for skill in build_builtin_local_tools():
        registry.register(skill, overwrite=True)
        registered.append(skill.name)
    return registered


__all__ = [
    "build_builtin_local_tools",
    "register_builtin_local_tools",
]
