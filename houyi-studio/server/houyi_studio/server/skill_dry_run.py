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

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from houyi.core.skill_registry import SkillRegistry

from .skill_serializer import POLICY_ALLOW, POLICY_DENY, extract_side_effects

if TYPE_CHECKING:
    from houyi.core.skill.policy import PolicyEnforcer
    from houyi.core.skill.spec import SkillSpec

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

        self._check_schema(skill, tool_name, input_data, result)
        self._check_policy(skill_name, tool_name, result)
        self._check_side_effects(skill, result)

        if live:
            result["llm_verification"] = await _live_verify(
                skill, skill_name, tool_name, input_data
            )

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
                            result["schema_errors"].append(str(e))
                        validated = True
                    break
        if not validated and hasattr(skill, "input_schema") and skill.input_schema:
            try:
                skill.input_schema.model_validate(input_data)
            except Exception as e:
                result["valid"] = False
                result["schema_errors"].append(str(e))

    def _check_policy(self, skill_name: str, tool_name: str, result: dict[str, Any]) -> None:
        if not self._policy_enforcer:
            return
        pr = self._policy_enforcer.evaluate(skill_name, tool_name, invoked_by_model=False)
        result["policy_result"] = pr.action.value
        if pr.action.value == POLICY_DENY:
            result["valid"] = False

    @staticmethod
    def _check_side_effects(skill: SkillSpec, result: dict[str, Any]) -> None:
        if hasattr(skill, "permissions") and skill.permissions:
            result["estimated_side_effects"] = extract_side_effects(skill.permissions)


# ── Live LLM verification with phased progressive disclosure ─────


def _build_natural_query(
    skill_name: str,
    tool_name: str,
    input_data: dict[str, Any],
) -> str:
    """Build a natural user query that lets the LLM decide how to use the tool.

    Instead of "call function X with args Y" (which causes the LLM to echo
    the input), we describe the *task* so the LLM has to reason about
    which tool to use and what arguments to supply.
    """
    if not input_data:
        return f"I need to use the '{tool_name}' capability. Please help me with this tool."

    # Build a task-oriented description from the input parameters
    parts: list[str] = []
    for key, value in input_data.items():
        parts.append(f"{key}={value}")
    params_str = ", ".join(parts)

    return (
        f"I need help with a task. Here is the context: {params_str}. "
        f"Use the '{tool_name}' tool to complete this request and return the result."
    )


async def _live_verify(
    skill: SkillSpec,
    skill_name: str,
    tool_name: str,
    input_data: dict[str, Any],
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
    try:
        from houyi.llm.factory import LLMAdapterFactory
    except ImportError:
        return {
            "success": False,
            "message": "LLM adapter not available — install houyi[model-adapters]",
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
        adapter = LLMAdapterFactory.create()
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
                    "model": model_name,
                    "latency_ms": exec_ms - (phases[-1]["timestamp_ms"] if phases else 0),
                    "usage": usage,
                },
            }
        )

        # ── Phase 5: Tool Execution ────────────────────────────────
        tool_call_args = result.get("tool_call", {}).get("arguments")
        if result.get("success") and tool_call_args and skill.executor:
            try:
                import asyncio
                import inspect

                if isinstance(tool_call_args, str):
                    try:
                        tool_call_args = json.loads(tool_call_args)
                    except Exception:
                        pass

                if isinstance(tool_call_args, dict):
                    exec_fn = skill.executor
                    if inspect.iscoroutinefunction(exec_fn):
                        exec_result = await exec_fn(**tool_call_args)
                    else:
                        exec_result = await asyncio.to_thread(exec_fn, **tool_call_args)

                    exec_result_str = str(exec_result)
                    if len(exec_result_str) > 2000:
                        exec_result_str = exec_result_str[:2000] + "... (truncated)"

                    result["execution_result"] = exec_result_str
                    phases.append(
                        {
                            "name": "tool_execution",
                            "label": "Tool Execution",
                            "timestamp_ms": _elapsed_ms(),
                            "status": "pass",
                            "data": {
                                "result_length": len(exec_result_str),
                                "result_preview": exec_result_str[:500],
                            },
                        }
                    )
                else:
                    result["execution_result"] = None
                    phases.append(
                        {
                            "name": "tool_execution",
                            "label": "Tool Execution",
                            "timestamp_ms": _elapsed_ms(),
                            "status": "skip",
                            "data": {"reason": "arguments not a valid dict"},
                        }
                    )
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

        result["probe_prompt"] = user_query
        result["system_prompt"] = system_prompt
        result["tool_definitions"] = tool_defs
        result["model_name"] = model_name
        result["usage"] = usage
        result["phases"] = phases
        return result

    except Exception as e:
        logger.exception("LLM live verification failed for skill '%s'", skill_name)
        exec_ms = _elapsed_ms()
        model_name = "unknown"
        try:
            adapter_check = LLMAdapterFactory.create()
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
            "phases": phases,
        }


def _build_tool_definitions(skill: SkillSpec) -> list[dict[str, Any]]:
    """Build OpenAI-format tool definitions from a SkillSpec.

    Handles two layouts:
    1. Skills with a ``tools`` list (multi-tool skills).
    2. Skills that *are* the tool (single-tool — name + input_schema on the skill itself).
    """
    defs: list[dict[str, Any]] = []

    if hasattr(skill, "tools") and skill.tools:
        for tool in skill.tools:
            schema: dict[str, Any] = {}
            if hasattr(tool, "input_schema") and tool.input_schema:
                try:
                    schema = tool.input_schema.model_json_schema()
                except Exception:
                    pass
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", "") or "",
                        "parameters": schema,
                    },
                }
            )
    else:
        # Single-tool skill: the skill itself is the tool
        schema = {}
        if hasattr(skill, "input_schema") and skill.input_schema:
            try:
                schema = skill.input_schema.model_json_schema()
            except Exception:
                pass
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": getattr(skill, "name", "unknown"),
                    "description": getattr(skill, "description", "") or "",
                    "parameters": schema,
                },
            }
        )

    return defs


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
        raw_content = str(response.content)[:500]
    elif hasattr(response, "choices") and response.choices:
        msg = getattr(response.choices[0], "message", None)
        if msg and getattr(msg, "content", None):
            raw_content = str(msg.content)[:500]

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

        if isinstance(first, dict):
            func = first.get("function", {})
            name = func.get("name") or first.get("name")
            args = func.get("arguments") or first.get("arguments")
        else:
            name = getattr(first, "name", None) or (
                getattr(first.function, "name", None) if hasattr(first, "function") else None
            )
            args = getattr(first, "arguments", None) or (
                getattr(first.function, "arguments", None) if hasattr(first, "function") else None
            )

        if isinstance(args, str):
            args = _strip_deepseek_tokens(args)
            try:
                args = json.loads(args)
            except Exception:
                pass

        matched = name == expected_tool
        return {
            "success": matched,
            "message": (
                f"LLM correctly called '{name}'"
                if matched
                else f"LLM called '{name}' instead of '{expected_tool}'"
            ),
            "tool_call": {"name": name, "arguments": args},
            "raw_content": raw_content or None,
        }

    return {
        "success": False,
        "message": f"LLM did not produce a tool call. Response: {raw_content[:200]}",
        "raw_content": raw_content or None,
    }
