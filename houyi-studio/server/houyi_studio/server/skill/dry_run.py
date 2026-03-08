"""Dry-run validation for skill invocations.

Validates schema compliance, policy, side effects, and optionally verifies
with a real LLM call (live mode).  No I/O except the optional LLM probe.

Progressive Disclosure Phases (when live=True):
  1. Discovery  — load skill metadata (name, description, version, hooks)
  2. Activation — build OpenAI-format tool definitions from SkillSpec
  3. Negotiation — construct system prompt + natural user query
  4. Execution  — send to LLM, parse response, record latency + usage
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import re
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from houyi.application.tool_calling.tool_bridge import build_tool_definitions_for_skill
from houyi.domain.skill.registry import SkillRegistry

from .serializer import POLICY_ALLOW, POLICY_DENY, extract_side_effects

if TYPE_CHECKING:
    from houyi.domain.skill.policy import PolicyEnforcer
    from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


class DryRunValidator:
    """Validates a skill invocation without executing it.

    Responsibilities (SRP):
    - Schema compliance check
    - Policy evaluation
    - Side-effect detection
    - (opt.) Live LLM verification with phased timing
    """

    def __init__(
        self,
        registry: SkillRegistry,
        policy_enforcer: PolicyEnforcer | None = None,
    ) -> None:
        self._registry = registry
        self._policy_enforcer = policy_enforcer

    # ── Public API ────────────────────────────────────────────────

    async def validate(
        self,
        skill_name: str,
        tool_name: str,
        input_data: dict[str, Any],
        live: bool = False,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        """Run static (and optionally live) validation.

        Returns a result dict with keys: ``valid``, ``schema_errors``,
        ``policy_result``, ``capability_gaps``, ``estimated_side_effects``,
        and optionally ``llm_verification`` (with ``phases`` timeline).
        """
        result: dict[str, Any] = {
            "valid": True,
            "schema_errors": [],
            "policy_result": POLICY_ALLOW,
            "capability_gaps": [],
            "estimated_side_effects": [],
        }

        skill = self._registry.get(skill_name)
        if not skill:
            result["valid"] = False
            result["schema_errors"].append(f"Skill not found: {skill_name}")
            return result

        available_workflows = _collect_available_workflows(skill)
        if available_workflows:
            result["available_workflows"] = available_workflows

        self._check_schema(skill, tool_name, input_data, result)
        self._check_policy(skill_name, tool_name, result)
        self._check_side_effects(skill, result)

        if live and result["valid"]:
            result["llm_verification"] = await _live_verify(
                skill,
                skill_name,
                tool_name,
                input_data,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        elif live and not result["valid"]:
            result["llm_verification"] = {
                "success": False,
                "message": f"Skipped — static validation failed (policy: {result['policy_result']})",
                "phases": [],
            }

        return result

    # ── Private validation steps ──────────────────────────────────

    @staticmethod
    def _check_schema(
        skill: SkillSpec,
        tool_name: str,
        input_data: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if not input_data:
            return  # availability-check only
        validated = False
        if hasattr(skill, "tools") and skill.tools:
            for tool in skill.tools:
                if getattr(tool, "name", None) == tool_name:
                    if hasattr(tool, "input_schema") and tool.input_schema:
                        try:
                            tool.input_schema.model_validate(input_data)
                        except Exception as e:
                            result["valid"] = False
                            result["schema_errors"].extend(DryRunValidator._format_schema_errors(e))
                        validated = True
                    break
        if not validated and hasattr(skill, "input_schema") and skill.input_schema:
            try:
                skill.input_schema.model_validate(input_data)
            except Exception as e:
                result["valid"] = False
                result["schema_errors"].extend(DryRunValidator._format_schema_errors(e))

    @staticmethod
    def _format_schema_errors(error: Exception) -> list[str]:
        if not hasattr(error, "errors"):
            return [str(error)]

        errors_attr = error.errors
        if not callable(errors_attr):
            return [str(error)]

        try:
            raw_errors = errors_attr()
        except Exception:
            return [str(error)]

        messages: list[str] = []
        for item in raw_errors:
            if not isinstance(item, dict):
                continue
            loc_raw = item.get("loc", ())
            loc = ".".join(str(part) for part in loc_raw if part not in {"__root__", "body"})
            type_code = str(item.get("type", ""))
            ctx = item.get("ctx") if isinstance(item.get("ctx"), dict) else {}
            message = DryRunValidator._format_single_schema_error(
                type_code, ctx, str(item.get("msg", "invalid value"))
            )
            if loc:
                messages.append(f"{loc}: {message}")
            else:
                messages.append(message)

        return messages or [str(error)]

    @staticmethod
    def _format_single_schema_error(type_code: str, ctx: dict[str, Any], fallback: str) -> str:
        if type_code == "missing":
            return "is required"
        if type_code in {"int_parsing", "int_type"}:
            return "must be an integer"
        if type_code in {"float_parsing", "float_type"}:
            return "must be a number"
        if type_code == "greater_than_equal":
            return f"must be >= {ctx.get('ge')}"
        if type_code == "greater_than":
            return f"must be > {ctx.get('gt')}"
        if type_code == "less_than_equal":
            return f"must be <= {ctx.get('le')}"
        if type_code == "less_than":
            return f"must be < {ctx.get('lt')}"
        if type_code == "string_too_short":
            return f"is too short (min {ctx.get('min_length')})"
        if type_code == "string_too_long":
            return f"is too long (max {ctx.get('max_length')})"
        return fallback

    def _check_policy(self, skill_name: str, tool_name: str, result: dict[str, Any]) -> None:
        skill = self._registry.get(skill_name)
        if skill:
            ip = getattr(skill, "invocation_policy", None)
            if ip is not None:
                mai = getattr(ip, "model_auto_invoke", None)
                if mai is not None:
                    action = mai.value if hasattr(mai, "value") else str(mai)
                    result["policy_result"] = action
                    if action == POLICY_DENY:
                        result["valid"] = False
                    return

        if not self._policy_enforcer:
            return
        try:
            pr = self._policy_enforcer.check_invocation(skill_name, is_model_initiated=False)
            result["policy_result"] = POLICY_DENY if not pr.allowed else POLICY_ALLOW
            if not pr.allowed:
                result["valid"] = False
        except AttributeError:
            pass

    @staticmethod
    def _check_side_effects(skill: SkillSpec, result: dict[str, Any]) -> None:
        if hasattr(skill, "permissions") and skill.permissions:
            result["estimated_side_effects"] = extract_side_effects(skill.permissions)


# ── Live LLM verification with phased progressive disclosure ─────


def _assess_tool_result(exec_result: object, exec_result_str: str) -> str:
    """Determine the phase status based on actual tool execution output.

    Returns "pass", "warn", or "fail".
    Priority: non-empty results → pass (even with fallback errors);
    empty results + errors → warn; empty results only → warn.
    """
    if isinstance(exec_result, dict):
        if exec_result.get("success") is False:
            return "fail"

        results = exec_result.get("results")
        has_results = isinstance(results, list) and len(results) > 0

        if has_results:
            return "pass"

        metadata = exec_result.get("metadata")
        has_errors = bool(exec_result.get("errors"))
        if isinstance(metadata, dict):
            has_errors = (
                has_errors or metadata.get("error_count", 0) > 0 or bool(metadata.get("errors"))
            )

        if isinstance(results, list) and len(results) == 0:
            return "warn"
        if has_errors:
            return "warn"

    lower = exec_result_str[:500].lower()
    if "'results': []" in lower or "results': []" in lower:
        return "warn"

    return "pass"


def _build_natural_query(
    skill_name: str,
    tool_name: str,
    input_data: dict[str, Any],
) -> str:
    """Build a precise query instructing the LLM to call the tool with exact arguments."""
    if not input_data:
        return f"Please call the '{tool_name}' tool with no arguments."

    params_str = json.dumps(input_data, ensure_ascii=False, indent=2)
    return (
        f"Please call the '{tool_name}' tool exactly with these arguments:\n"
        f"```json\n{params_str}\n```\n"
        f"Do not modify, guess, or omit any values. Just pass them directly to the tool."
    )


async def _live_verify(
    skill: SkillSpec,
    skill_name: str,
    tool_name: str,
    input_data: dict[str, Any],
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Send a probe to the LLM and check the tool call.

    Returns an enriched result containing **phases** — a timeline of
    progressive disclosure stages with wall-clock timestamps (ms), making
    the trigger timing of each stage visible in the UI.

    Phases:
      1. discovery   — skill metadata loaded from registry
      2. activation  — tool definitions built from SkillSpec
      3. negotiation — system prompt + user query constructed
      4. execution   — real LLM API call, response parsed
    """
    requested_input: dict[str, Any] = dict(input_data)

    try:
        from houyi.adapters.llm import LLMAdapterFactory
    except ImportError:
        return {
            "success": False,
            "message": "LLM adapter not available — install houyi[model-adapters]",
            "requested_input": requested_input,
        }

    phases: list[dict[str, Any]] = []
    t0 = time.monotonic()

    def _elapsed_ms() -> int:
        return round((time.monotonic() - t0) * 1000)

    # ── Phase 1: Discovery ────────────────────────────────────────
    skill_desc = getattr(skill, "description", "") or ""
    skill_version = getattr(skill, "version", "") or ""
    skill_hooks = [h.event.value for h in getattr(skill, "hooks", [])]
    phases.append(
        {
            "name": "discovery",
            "label": "Skill Discovery",
            "timestamp_ms": _elapsed_ms(),
            "status": "pass",
            "data": {
                "skill_name": skill_name,
                "description": skill_desc[:200],
                "version": skill_version,
                "hooks": skill_hooks,
            },
        }
    )

    # ── Phase 2: Activation ───────────────────────────────────────
    tool_defs = _build_tool_definitions(skill)
    tool_names = [td.get("function", {}).get("name", "") for td in tool_defs]
    phases.append(
        {
            "name": "activation",
            "label": "Tool Activation",
            "timestamp_ms": _elapsed_ms(),
            "status": "pass" if tool_defs else "fail",
            "data": {
                "tool_count": len(tool_defs),
                "tool_names": tool_names,
            },
        }
    )

    if not tool_defs:
        return {
            "success": False,
            "message": "No tool definitions available for LLM probe",
            "phases": phases,
            "requested_input": requested_input,
        }

    # ── Phase 3: Negotiation ──────────────────────────────────────
    system_prompt = (
        f"You are a helpful assistant. You have access to the "
        f"'{skill_name}' skill ({skill_desc}). "
        f"When the user asks for help, use the available tools. "
        f"Always prefer calling a tool over answering from memory."
    )
    user_query = _build_natural_query(skill_name, tool_name, input_data)

    phases.append(
        {
            "name": "negotiation",
            "label": "LLM Negotiation",
            "timestamp_ms": _elapsed_ms(),
            "status": "pass",
            "data": {
                "system_prompt_length": len(system_prompt),
                "user_query": user_query,
            },
        }
    )

    # ── Phase 4: Execution ────────────────────────────────────────
    try:
        adapter = LLMAdapterFactory.create(llm_provider)
        if llm_model:
            if hasattr(adapter, "model"):
                adapter.model = llm_model
            if hasattr(adapter, "default_model"):
                adapter.default_model = llm_model
        model_name = (
            getattr(adapter, "model", None) or getattr(adapter, "default_model", None) or "unknown"
        )

        response = await adapter.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            tools=tool_defs,
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        exec_ms = _elapsed_ms()

        result = _parse_llm_response(response, tool_name)

        # Capture usage stats to prove a real API call happened
        usage: dict[str, Any] = {}
        if hasattr(response, "usage") and response.usage:
            raw_usage = response.usage
            if isinstance(raw_usage, dict):
                usage = raw_usage
            else:
                usage = {
                    k: getattr(raw_usage, k, 0)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if hasattr(raw_usage, k)
                }

        phases.append(
            {
                "name": "execution",
                "label": "LLM Execution",
                "timestamp_ms": exec_ms,
                "status": "pass" if result.get("success") else "fail",
                "data": {
                    "provider": llm_provider or "default",
                    "model": model_name,
                    "latency_ms": exec_ms - (phases[-1]["timestamp_ms"] if phases else 0),
                    "usage": usage,
                },
            }
        )

        # ── Phase 5: Tool Execution ────────────────────────────────
        tool_call = result.get("tool_call", {}) if isinstance(result.get("tool_call"), dict) else {}
        tool_call_args = tool_call.get("arguments")
        tool_exec_status: str | None = None
        tool_exec_payload: dict[str, Any] | None = None
        executor = skill.executor if callable(getattr(skill, "executor", None)) else None
        if result.get("success") and executor is None:
            executor = _derive_script_compat_executor(skill)

        if result.get("success") and executor:
            try:
                import asyncio
                import inspect

                if isinstance(tool_call_args, str):
                    with contextlib.suppress(Exception):
                        tool_call_args = json.loads(tool_call_args)

                argument_source = "observed_tool_call"
                if _is_missing_args(tool_call_args) and requested_input:
                    tool_call_args = dict(requested_input)
                    argument_source = "requested_input_fallback"
                    tool_call["arguments"] = tool_call_args
                    tool_call["arguments_source"] = argument_source
                    result["tool_call"] = tool_call

                if isinstance(tool_call_args, dict):
                    exec_fn = executor
                    if inspect.iscoroutinefunction(exec_fn):
                        exec_result = await exec_fn(**tool_call_args)
                    else:
                        exec_result = await asyncio.to_thread(exec_fn, **tool_call_args)

                    try:
                        exec_result_str = json.dumps(exec_result, ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        exec_result_str = str(exec_result)
                    if len(exec_result_str) > 2000:
                        exec_result_str = exec_result_str[:2000] + "... (truncated)"

                    result["execution_result"] = exec_result_str
                    exec_status = _assess_tool_result(exec_result, exec_result_str)
                    preview = (
                        exec_result
                        if isinstance(exec_result, (dict, list))
                        else exec_result_str[:500]
                    )
                    phases.append(
                        {
                            "name": "tool_execution",
                            "label": "Tool Execution",
                            "timestamp_ms": _elapsed_ms(),
                            "status": exec_status,
                            "data": {
                                "result_length": len(exec_result_str),
                                "result_preview": preview,
                                "argument_source": argument_source,
                            },
                        }
                    )
                    tool_exec_status = exec_status
                    tool_exec_payload = {
                        "result_length": len(exec_result_str),
                        "result_preview": preview,
                        "argument_source": argument_source,
                    }
                else:
                    result["execution_result"] = None
                    phases.append(
                        {
                            "name": "tool_execution",
                            "label": "Tool Execution",
                            "timestamp_ms": _elapsed_ms(),
                            "status": "skip",
                            "data": {
                                "reason": "arguments not a valid dict",
                                "argument_source": "none",
                            },
                        }
                    )
                    tool_exec_status = "skip"
                    tool_exec_payload = {
                        "reason": "arguments not a valid dict",
                        "argument_source": "none",
                    }
            except Exception as exec_err:
                logger.warning(
                    "Tool execution failed for '%s': %s",
                    tool_name,
                    exec_err,
                )
                result["execution_result"] = f"Error: {exec_err}"
                phases.append(
                    {
                        "name": "tool_execution",
                        "label": "Tool Execution",
                        "timestamp_ms": _elapsed_ms(),
                        "status": "fail",
                        "data": {"error": str(exec_err)[:300]},
                    }
                )
                tool_exec_status = "fail"
                tool_exec_payload = {"error": str(exec_err)[:300]}
        elif result.get("success"):
            phases.append(
                {
                    "name": "tool_execution",
                    "label": "Tool Execution",
                    "timestamp_ms": _elapsed_ms(),
                    "status": "skip",
                    "data": {"reason": "no executor available"},
                }
            )
            tool_exec_status = "skip"
            tool_exec_payload = {"reason": "no executor available"}

        # ── Phase 6: Final Response Synthesis ──────────────────────
        if result.get("success") and tool_exec_status not in {None, "skip"}:
            try:
                tool_result_summary = result.get("execution_result")
                followup = await adapter.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                        {
                            "role": "user",
                            "content": (
                                "The tool has been executed. Based on the following tool result, "
                                "provide the final answer for the original request.\n"
                                f"Tool result:\n{tool_result_summary}"
                            ),
                        },
                    ],
                    tools=[],
                )
                final_answer = _extract_response_content(followup)
                result["final_answer"] = final_answer
                final_status = "pass" if final_answer else "warn"
                phases.append(
                    {
                        "name": "final_response",
                        "label": "Final Response",
                        "timestamp_ms": _elapsed_ms(),
                        "status": final_status,
                        "data": {
                            "answer_preview": final_answer[:500] if final_answer else "",
                            "tool_execution_status": tool_exec_status,
                            "tool_execution": tool_exec_payload or {},
                        },
                    }
                )
            except Exception as followup_err:
                phases.append(
                    {
                        "name": "final_response",
                        "label": "Final Response",
                        "timestamp_ms": _elapsed_ms(),
                        "status": "fail",
                        "data": {
                            "error": str(followup_err)[:300],
                            "tool_execution_status": tool_exec_status,
                        },
                    }
                )

        result["probe_prompt"] = user_query
        result["system_prompt"] = system_prompt
        result["tool_definitions"] = tool_defs
        result["model_name"] = model_name
        result["provider"] = llm_provider or "default"
        result["usage"] = usage
        result["phases"] = phases
        result["requested_input"] = requested_input
        return result

    except Exception as e:
        logger.exception("LLM live verification failed for skill '%s'", skill_name)
        exec_ms = _elapsed_ms()
        model_name = "unknown"
        try:
            adapter_check = LLMAdapterFactory.create(llm_provider)
            if llm_model:
                if hasattr(adapter_check, "model"):
                    adapter_check.model = llm_model
                if hasattr(adapter_check, "default_model"):
                    adapter_check.default_model = llm_model
            model_name = (
                getattr(adapter_check, "model", None)
                or getattr(adapter_check, "default_model", None)
                or "unknown"
            )
        except Exception:
            pass

        phases.append(
            {
                "name": "execution",
                "label": "LLM Execution",
                "timestamp_ms": exec_ms,
                "status": "fail",
                "data": {"error": str(e)[:300]},
            }
        )

        return {
            "success": False,
            "message": f"LLM verification failed: {e}",
            "probe_prompt": user_query if "user_query" in dir() else "",
            "system_prompt": system_prompt if "system_prompt" in dir() else "",
            "tool_definitions": tool_defs if "tool_defs" in dir() else [],
            "model_name": model_name,
            "provider": llm_provider or "default",
            "phases": phases,
            "requested_input": requested_input,
        }


def _simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip Pydantic-specific metadata from a JSON Schema for LLM consumption.

    Many LLMs (especially non-OpenAI providers) get confused by Pydantic's
    ``model_json_schema()`` extras — top-level ``title``, per-property
    ``title``, and ``anyOf`` wrappers for optional fields.  Cleaning these
    produces a minimal schema that all providers handle reliably.
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop_name: _simplify_property(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        else:
            cleaned[key] = value
    return cleaned


def _derive_script_compat_executor(skill: SkillSpec) -> Any | None:
    """Best-effort runtime binding for instruction-driven script skills."""
    instructions = getattr(skill, "instructions", None)
    raw_skill_dir = getattr(skill, "skill_dir", None)
    if not isinstance(instructions, str) or not instructions.strip() or not raw_skill_dir:
        return None

    try:
        from .loader import SkillLoader

        skill_dir = Path(raw_skill_dir).resolve()
        templates = SkillLoader._extract_script_command_templates(instructions)
        if not templates:
            return None
    except Exception:
        return None

    async def _executor(**kwargs):
        import asyncio
        import json
        import sys

        def _normalize_command(command: list[str]) -> list[str]:
            normalized: list[str] = []
            for idx, token in enumerate(command):
                if idx == 0 and token == "python":
                    normalized.append(sys.executable)
                    continue
                if token.startswith("-"):
                    normalized.append(token)
                    continue

                path_token = Path(token)
                if not path_token.is_absolute():
                    candidate = (skill_dir / path_token).resolve()
                    if candidate.exists():
                        normalized.append(str(candidate))
                        continue
                normalized.append(token)
            return normalized

        def _dependency_state(command: list[str]) -> tuple[list[str], list[str]]:
            required = SkillLoader._infer_required_binaries(command)
            return required, SkillLoader._missing_binaries(required)

        explicit_command = isinstance(kwargs.get("command"), str) and bool(
            str(kwargs.get("command", "")).strip()
        )
        explicit_workflow = isinstance(kwargs.get("workflow_id"), str) and bool(
            str(kwargs.get("workflow_id", "")).strip()
        )

        command = SkillLoader._build_script_compat_command(kwargs, templates)
        if not command:
            return {
                "success": False,
                "message": "No executable script command could be derived from payload",
                "payload": kwargs,
            }

        normalized_cmd = _normalize_command(command)
        _, missing_bins = _dependency_state(normalized_cmd)
        if missing_bins and not explicit_command and not explicit_workflow:
            for idx in range(len(templates)):
                candidate_payload = dict(kwargs)
                candidate_payload["workflow_id"] = f"template_{idx + 1}"
                candidate_cmd = SkillLoader._build_script_compat_command(
                    candidate_payload, templates
                )
                if not candidate_cmd:
                    continue
                candidate_normalized = _normalize_command(candidate_cmd)
                _, candidate_missing = _dependency_state(candidate_normalized)
                if candidate_missing:
                    continue
                normalized_cmd = candidate_normalized
                _, missing_bins = _dependency_state(normalized_cmd)
                break

        if missing_bins:
            missing_msg = (
                "Missing required runtime dependency: "
                + ", ".join(missing_bins)
                + ". Please install it (e.g. LibreOffice provides 'soffice')."
            )
            return {
                "success": False,
                "exit_code": 127,
                "error_code": "missing_dependency",
                "missing_dependencies": missing_bins,
                "command": normalized_cmd,
                "output": "",
                "stderr": missing_msg,
            }

        proc = await asyncio.create_subprocess_exec(
            *normalized_cmd,
            cwd=str(skill_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "success": False,
                "message": "Script compatibility execution timed out",
                "command": normalized_cmd,
            }

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        parsed_output: object = stdout_text
        if stdout_text:
            with contextlib.suppress(Exception):
                parsed_output = json.loads(stdout_text)

        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "command": normalized_cmd,
            "output": parsed_output,
            "stderr": stderr_text,
        }

    return _executor


def _collect_available_workflows(skill: SkillSpec) -> list[dict[str, Any]]:
    extra = getattr(skill, "extra_frontmatter", None)
    if isinstance(extra, dict):
        workflows = _collect_frontmatter_workflows(extra)
        if workflows:
            return workflows

    instructions = getattr(skill, "instructions", None)
    if not isinstance(instructions, str) or not instructions.strip():
        return []

    try:
        from .loader import SkillLoader

        templates = SkillLoader._extract_script_command_templates(instructions)
    except Exception:
        return []

    workflows: list[dict[str, Any]] = []
    for idx, template in enumerate(templates, start=1):
        base_tokens = [str(t) for t in (template.get("base_tokens") or [])]
        raw = str(template.get("raw") or "").strip()
        if not _is_meaningful_workflow_template(base_tokens=base_tokens, raw=raw):
            continue
        command_text = raw or " ".join(base_tokens)

        wf_id = f"template_{idx}"
        flags = [str(flag) for flag in (template.get("flags") or [])]
        required_bins = SkillLoader._infer_required_binaries(base_tokens)
        missing_bins = SkillLoader._missing_binaries(required_bins)

        confidence_score, confidence_level = _workflow_confidence(
            source="instructions",
            base_tokens=base_tokens,
            command_text=command_text,
        )
        validation_status = "pass" if not missing_bins else "warn"
        validation_issues = []
        if missing_bins:
            validation_issues.append(
                "Missing dependency: "
                f"{', '.join(missing_bins)}"
                " (workflow is discoverable but execution will fail until installed)"
            )

        workflows.append(
            {
                "id": wf_id,
                "title": _workflow_title_from_template(base_tokens=base_tokens, workflow_id=wf_id),
                "command": command_text,
                "params": flags,
                "depends_on": required_bins,
                "source": "instructions",
                "evidence": raw or " ".join(base_tokens),
                "confidence": confidence_level,
                "confidence_score": confidence_score,
                "validation": {
                    "status": validation_status,
                    "issues": validation_issues,
                    "missing_dependencies": missing_bins,
                },
            }
        )

    return _dedupe_workflows(workflows)


def _collect_frontmatter_workflows(extra_frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    raw = extra_frontmatter.get("workflows")
    if not isinstance(raw, list):
        return []

    workflows: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue

        command = item.get("command")
        if not isinstance(command, str) or not command.strip():
            continue

        workflow_id = str(item.get("id") or f"workflow_{index}").strip()
        title = str(item.get("title") or workflow_id).strip()

        params: list[str] = []
        raw_params = item.get("params")
        if isinstance(raw_params, list):
            params = [str(p).strip() for p in raw_params if str(p).strip()]
        elif isinstance(raw_params, dict):
            params = [str(k).strip() for k in raw_params if str(k).strip()]

        depends_on: list[str] = []
        raw_depends_on = item.get("depends_on")
        if isinstance(raw_depends_on, list):
            depends_on = [str(d).strip() for d in raw_depends_on if str(d).strip()]

        with contextlib.suppress(Exception):
            from .loader import SkillLoader

            depends_on = depends_on or SkillLoader._infer_required_binaries(shlex.split(command))

        missing_dependencies: list[str] = []
        with contextlib.suppress(Exception):
            from .loader import SkillLoader

            missing_dependencies = SkillLoader._missing_binaries(depends_on)

        confidence_score, confidence_level = _workflow_confidence(
            source="frontmatter",
            base_tokens=shlex.split(command),
            command_text=command,
        )
        validation_status = "pass" if not missing_dependencies else "warn"
        validation_issues = []
        if missing_dependencies:
            validation_issues.append(
                "Missing dependency: "
                f"{', '.join(missing_dependencies)}"
                " (workflow is discoverable but execution will fail until installed)"
            )

        workflows.append(
            {
                "id": workflow_id,
                "title": title,
                "command": command.strip(),
                "params": params,
                "depends_on": depends_on,
                "source": "frontmatter",
                "evidence": f"frontmatter.workflows[{index - 1}]",
                "confidence": confidence_level,
                "confidence_score": confidence_score,
                "validation": {
                    "status": validation_status,
                    "issues": validation_issues,
                    "missing_dependencies": missing_dependencies,
                },
            }
        )

    return _dedupe_workflows(workflows)


def _is_meaningful_workflow_template(*, base_tokens: list[str], raw: str) -> bool:
    if not base_tokens:
        return False
    joined = " ".join(base_tokens).lower()
    if not joined.strip():
        return False
    if joined in {"python", "sh"}:
        return False
    if raw.strip().startswith("#"):
        return False
    if any(token.endswith(".py") for token in base_tokens):
        return True
    if base_tokens[0].startswith("./"):
        return True
    return base_tokens[0] in {"python", "sh"} and len(base_tokens) >= 2


def _workflow_title_from_template(*, base_tokens: list[str], workflow_id: str) -> str:
    if not base_tokens:
        return workflow_id

    script_index = -1
    script_name = ""
    for idx, token in enumerate(base_tokens):
        if token.endswith(".py"):
            script_index = idx
            script_name = Path(token).name
            break

    if not script_name:
        return workflow_id

    # Improve readability for script runner patterns like:
    # python scripts/run.py auth_manager.py status
    if script_name == "run.py" and script_index >= 0:
        script_token = (
            Path(base_tokens[script_index + 1]).name
            if script_index + 1 < len(base_tokens)
            and not base_tokens[script_index + 1].startswith("--")
            else ""
        )
        operation_token = (
            base_tokens[script_index + 2]
            if script_index + 2 < len(base_tokens)
            and not base_tokens[script_index + 2].startswith("--")
            else ""
        )
        if script_token and operation_token and not operation_token.endswith(".py"):
            return f"{script_token} · {operation_token}"
        if script_token:
            return script_token

    return script_name


def _workflow_confidence(
    *, source: str, base_tokens: list[str], command_text: str
) -> tuple[float, str]:
    score = 0.35
    if source == "frontmatter":
        score += 0.5
    else:
        score += 0.25
    if any(token.endswith(".py") for token in base_tokens):
        score += 0.1
    if " --" in f" {command_text}":
        score += 0.05

    bounded = max(0.0, min(1.0, round(score, 2)))
    if bounded >= 0.85:
        return bounded, "high"
    if bounded >= 0.65:
        return bounded, "medium"
    return bounded, "low"


def _dedupe_workflows(workflows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_commands: set[str] = set()
    for workflow in workflows:
        workflow_id = str(workflow.get("id") or "").strip()
        command = str(workflow.get("command") or "").strip().lower()
        if not workflow_id or not command:
            continue
        if workflow_id in seen_ids or command in seen_commands:
            continue
        seen_ids.add(workflow_id)
        seen_commands.add(command)
        deduped.append(workflow)
    return deduped


def _parse_tool_arguments(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw

    cleaned = _strip_deepseek_tokens(raw).strip()
    if not cleaned:
        return cleaned

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_ch)
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == start_ch:
                    depth += 1
                elif cleaned[i] == end_ch:
                    depth -= 1
                    if depth == 0:
                        cleaned = cleaned[start : i + 1]
                        break
            break

    with contextlib.suppress(Exception):
        return json.loads(cleaned)

    with contextlib.suppress(Exception):
        return ast.literal_eval(cleaned)

    return cleaned


def _extract_action(payload: Any, depth: int = 0) -> str:
    if depth > 6 or payload is None:
        return ""

    if isinstance(payload, str):
        parsed = _parse_tool_arguments(payload)
        if parsed is payload:
            return ""
        return _extract_action(parsed, depth + 1)

    if isinstance(payload, list):
        for item in payload:
            action = _extract_action(item, depth + 1)
            if action:
                return action
        return ""

    if not isinstance(payload, dict):
        return ""

    for key, value in payload.items():
        if key.lower() == "action" and isinstance(value, str):
            return value

    for key in ("arguments", "input", "params", "payload", "data", "args", "kwargs", "tool_input"):
        if key in payload:
            action = _extract_action(payload.get(key), depth + 1)
            if action:
                return action

    for value in payload.values():
        action = _extract_action(value, depth + 1)
        if action:
            return action

    return ""


def _simplify_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Simplify a single property schema: drop title, flatten anyOf nullables."""
    result: dict[str, Any] = {}
    for key, value in prop.items():
        if key == "title":
            continue
        if key == "anyOf" and isinstance(value, list):
            non_null = [t for t in value if t != {"type": "null"}]
            if len(non_null) == 1:
                result.update(non_null[0])
                continue
        result[key] = value
    return result


def _build_tool_definitions(skill: SkillSpec) -> list[dict[str, Any]]:
    """Build OpenAI-format tool definitions from a SkillSpec."""
    return build_tool_definitions_for_skill(skill)


_DEEPSEEK_TOKEN_RE = re.compile(
    r"<[｜\|].*?[｜\|]>",
    re.DOTALL,
)


def _strip_deepseek_tokens(text: str) -> str:
    """Remove DeepSeek internal markers (e.g. <｜tool▁calls▁begin｜>) and
    extract the first valid JSON object or array from the string."""
    cleaned = _DEEPSEEK_TOKEN_RE.sub("", text).strip()
    for start_ch, end_ch in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == start_ch:
                depth += 1
            elif cleaned[i] == end_ch:
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
    return cleaned


def _parse_llm_response(response: object, expected_tool: str) -> dict[str, Any]:
    """Extract tool call info from an LLMResponse or raw OpenAI response.

    Always captures ``raw_content`` (text content from the LLM, if any)
    alongside the parsed ``tool_call``, providing full transparency.
    """
    tool_calls = getattr(response, "tool_calls", None) or []

    # Capture raw text content from the response
    raw_content = ""
    if hasattr(response, "content") and response.content:
        raw_content = str(response.content)[:4000]
    elif hasattr(response, "choices") and response.choices:
        msg = getattr(response.choices[0], "message", None)
        if msg and getattr(msg, "content", None):
            raw_content = str(msg.content)[:4000]

    # Fallback: raw OpenAI response with choices[].message.tool_calls
    if not tool_calls and hasattr(response, "choices"):
        for choice in response.choices:
            msg = getattr(choice, "message", None)
            if msg and getattr(msg, "tool_calls", None):
                tool_calls = msg.tool_calls
                break

    if tool_calls:
        first = tool_calls[0]
        name: str | None = None
        args: Any = None

        def _pick_non_none(*values: Any) -> Any:
            for value in values:
                if value is not None:
                    return value
            return None

        if isinstance(first, dict):
            func = first.get("function", {})
            name = _pick_non_none(func.get("name"), first.get("name"))
            args = _pick_non_none(func.get("arguments"), first.get("arguments"))
        else:
            name = _pick_non_none(
                getattr(first, "name", None),
                getattr(first.function, "name", None) if hasattr(first, "function") else None,
            )
            args = _pick_non_none(
                getattr(first, "arguments", None),
                getattr(first.function, "arguments", None) if hasattr(first, "function") else None,
            )

        args = _parse_tool_arguments(args)

        recovered_name, recovered_args = _recover_tool_call_from_text(raw_content)
        if not name and recovered_name:
            name = recovered_name
        if _is_missing_args(args) and recovered_args is not None:
            args = recovered_args

        # Unpack nested LLM hallucinations e.g. arguments={"name": "foo", "arguments": {"action": ...}}
        if (
            isinstance(args, dict)
            and "arguments" in args
            and "name" in args
            and args["name"] == name
        ):
            args = args["arguments"]

        observed_action = _extract_action(args)

        matched = name == expected_tool
        return {
            "success": matched,
            "message": (
                f"LLM correctly called '{name}'"
                if matched
                else f"LLM called '{name}' instead of '{expected_tool}'"
            ),
            "tool_call": {
                "name": name,
                "arguments": args,
                **({"action": observed_action} if observed_action else {}),
            },
            "raw_content": raw_content or None,
        }

    recovered_name, recovered_args = _recover_tool_call_from_text(raw_content)
    if recovered_name:
        observed_action = _extract_action(recovered_args)
        matched = recovered_name == expected_tool
        return {
            "success": matched,
            "message": (
                f"LLM correctly called '{recovered_name}'"
                if matched
                else f"LLM called '{recovered_name}' instead of '{expected_tool}'"
            ),
            "tool_call": {
                "name": recovered_name,
                "arguments": recovered_args,
                **({"action": observed_action} if observed_action else {}),
            },
            "raw_content": raw_content or None,
        }

    return {
        "success": False,
        "message": f"LLM did not produce a tool call. Response: {raw_content[:200]}",
        "raw_content": raw_content or None,
    }


def _is_missing_args(args: Any) -> bool:
    if args is None:
        return True
    if isinstance(args, str):
        return not args.strip()
    if isinstance(args, dict):
        return len(args) == 0
    if isinstance(args, list):
        return len(args) == 0
    return False


def _recover_tool_call_from_text(raw_content: str) -> tuple[str | None, Any]:
    if not raw_content:
        return None, None

    parsed = _parse_tool_arguments(raw_content)
    if parsed is raw_content:
        return None, None
    return _find_tool_call_candidate(parsed)


def _extract_response_content(response: object) -> str:
    """Best-effort extraction of assistant text from adapter response."""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if content is not None:
        return str(content).strip()

    if hasattr(response, "choices") and response.choices:
        msg = getattr(response.choices[0], "message", None)
        text = getattr(msg, "content", None) if msg is not None else None
        if isinstance(text, str):
            return text.strip()
        if text is not None:
            return str(text).strip()

    return ""


def _find_tool_call_candidate(payload: Any, depth: int = 0) -> tuple[str | None, Any]:
    if payload is None or depth > 8:
        return None, None

    if isinstance(payload, str):
        parsed = _parse_tool_arguments(payload)
        if parsed is payload:
            return None, None
        return _find_tool_call_candidate(parsed, depth + 1)

    if isinstance(payload, list):
        for item in payload:
            name, args = _find_tool_call_candidate(item, depth + 1)
            if name:
                return name, args
        return None, None

    if not isinstance(payload, dict):
        return None, None

    function_block = payload.get("function")
    if isinstance(function_block, dict):
        name = function_block.get("name")
        if isinstance(name, str) and name:
            args = function_block.get("arguments")
            return name, _parse_tool_arguments(args)

    name = payload.get("name")
    if isinstance(name, str) and name:
        args = payload.get("arguments")
        if args is None:
            args = payload.get("args")
        return name, _parse_tool_arguments(args)

    for key in ("tool_calls", "tool_call", "calls", "choices", "message"):
        if key in payload:
            recovered_name, recovered_args = _find_tool_call_candidate(payload[key], depth + 1)
            if recovered_name:
                return recovered_name, recovered_args

    for value in payload.values():
        recovered_name, recovered_args = _find_tool_call_candidate(value, depth + 1)
        if recovered_name:
            return recovered_name, recovered_args

    return None, None
