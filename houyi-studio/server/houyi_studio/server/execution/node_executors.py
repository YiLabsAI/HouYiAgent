"""Node executors for the console execution engine."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from houyi.application.workflow.config_service import ConfigService
from houyi.application.workflow.llm_node_utils import build_llm_node_inputs
from houyi.application.workflow.route_node_utils import evaluate_route
from houyi.application.workflow.skill_executor import SkillExecutor
from houyi.application.workflow.tool_node_utils import (
    build_inputs_from_context_values,
    extract_schema_fields,
    normalize_tool_name,
)
from houyi.application.workflow.verify_node_utils import build_verification_rules
from houyi.assurance.verification import ConstraintChecker, PythonVerifier, SQLVerifier
from houyi.assurance.verification.verifier import VerificationResult
from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.infrastructure.observability import Span, SpanType, TraceContext
from houyi.interface.protocol.ir import ToolNodeOutputIR
from houyi.interface.protocol.ir.execution_ir import NodeExecutionIR
from houyi.interface.protocol.ir.plan_ir import NodeIR

from ..gateway.events import SpanUpdateEvent
from .context import ExecutionContext
from .node_executor_registry import NodeExecutor

logger = logging.getLogger(__name__)

LLMExecute = Callable[
    [
        str,
        Any,
        str,
        NodeExecutionIR,
        str,
        str | None,
        int | None,
        bool,
        int | None,
        str | None,
        str | None,
        bool,
        list[str] | None,
        Any | None,
        int,
        float | None,
        bool | None,
        str | None,
    ],
    Awaitable[None],
]
LLMMockExecute = Callable[[str, Any, str, NodeExecutionIR], Awaitable[None]]


class LLMNodeExecutor(NodeExecutor):
    """Executor for LLM nodes."""

    def __init__(
        self,
        config_service: ConfigService,
        execute_llm_real: LLMExecute,
        execute_llm_mock: LLMMockExecute,
        use_mock: bool = False,
    ) -> None:
        self.config_service = config_service
        self.execute_llm_real = execute_llm_real
        self.execute_llm_mock = execute_llm_mock
        self.use_mock = use_mock

    def build_inputs(self, context: ExecutionContext, node: NodeIR) -> dict[str, Any]:
        config = node.config if isinstance(node.config, dict) else {}
        return build_llm_node_inputs(
            config=config,
            run_settings=context.run_settings,
            resolve_tool_settings=self.config_service.resolve_tool_settings,
        )

    async def execute(
        self,
        context: ExecutionContext,
        node: NodeIR,
        node_exec: NodeExecutionIR,
    ) -> None:
        inputs = node_exec.inputs or self.build_inputs(context, node)
        node_exec.inputs = inputs
        prompt = inputs.get("prompt", "")
        system_prompt = inputs.get("system_prompt")
        user_prompt = inputs.get("user_prompt")
        model = inputs.get("model")
        max_tokens = inputs.get("max_tokens")
        enable_reasoning = bool(inputs.get("enable_reasoning"))
        thinking_budget = inputs.get("thinking_budget")
        enable_tool_calls = bool(inputs.get("enable_tool_calls"))
        tool_names = inputs.get("tool_names") or []
        tool_choice = inputs.get("tool_choice")
        max_tool_calls = inputs.get("max_tool_calls", 6)
        temperature = inputs.get("temperature")
        parallel_tool_calls = inputs.get("parallel_tool_calls")
        prompt_cache_key = inputs.get("prompt_cache_key")

        logger.debug("Executing LLM node %s with prompt: %s", node.node_id, str(prompt)[:50])
        logger.debug("Node config: %s", node.config)
        logger.debug("enable_reasoning=%s (type: %s)", enable_reasoning, type(enable_reasoning))
        if enable_reasoning:
            logger.debug(
                "Reasoning enabled for node %s (budget: %s)", node.node_id, thinking_budget
            )
        else:
            logger.debug("Reasoning disabled for node %s", node.node_id)

        if self.use_mock:
            await self.execute_llm_mock(
                context.session_id, context.execution, node.node_id, node_exec
            )
            return

        await self.execute_llm_real(
            context.session_id,
            context.execution,
            node.node_id,
            node_exec,
            prompt,
            model,
            max_tokens,
            enable_reasoning,
            thinking_budget,
            system_prompt,
            user_prompt,
            enable_tool_calls,
            tool_names,
            tool_choice,
            max_tool_calls,
            temperature,
            parallel_tool_calls,
            prompt_cache_key,
        )


class ToolNodeExecutor(NodeExecutor):
    """Executor for TOOL nodes."""

    @staticmethod
    def _normalize_tool_name(value: str) -> str:
        return normalize_tool_name(value)

    def build_inputs(self, context: ExecutionContext, node: NodeIR) -> dict[str, Any]:
        return node.inputs or {}

    @staticmethod
    def _extract_schema_fields(input_schema: type | None) -> set[str]:
        return extract_schema_fields(input_schema)

    @staticmethod
    def _create_tool_span(tool_name: str) -> Span | None:
        """Create tool span as child of current trace context.

        Returns None if no active trace context (instrumentation disabled).
        """
        parent = TraceContext.current()
        if parent is None:
            return None

        return Span(
            name=f"tool.{tool_name}",
            parent=parent,
            span_type=SpanType.TOOL,
            tool_name=tool_name,
            attributes={
                "tool.name": tool_name,
            },
        )

    @staticmethod
    async def _emit_child_spans(
        parent_span: Any,
        obs: Any,
        session_id: str,
        execution_id: str,
    ) -> None:
        """Recursively emit all child spans (internal sub-spans) to the observation service.

        Tool instrumentation (e.g. WebSearchService) creates child spans on the
        tool span. These need to be emitted separately so they appear in the
        frontend Timeline waterfall.
        """
        children = getattr(parent_span, "children", None)
        if not children:
            return
        for child in children:
            try:
                await obs.emit(
                    SpanUpdateEvent.from_span(
                        child,
                        session_id=session_id,
                        execution_id=execution_id,
                    )
                )
            except Exception:
                logger.debug("Failed to emit child span %s", getattr(child, "span_id", "?"))
            # Recurse into grandchildren
            await ToolNodeExecutor._emit_child_spans(child, obs, session_id, execution_id)

    @staticmethod
    def _build_tool_cache_key(tool_name: str, args: dict[str, Any]) -> str | None:
        """Build a cache key matching ToolCallRunner's format for cross-path consistency."""
        payload = {"tool": tool_name, "args": args, "version": None}
        try:
            return json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except TypeError:
            return None

    def _build_inputs_from_context(
        self,
        context: ExecutionContext,
        skill: Any,
    ) -> dict[str, Any]:
        schema_fields = self._extract_schema_fields(getattr(skill, "input_schema", None))
        return build_inputs_from_context_values(
            schema_fields=schema_fields,
            context_values=context.execution.context,
        )

    async def execute(
        self,
        context: ExecutionContext,
        node: NodeIR,
        node_exec: NodeExecutionIR,
    ) -> None:
        node_exec.inputs = node_exec.inputs or self.build_inputs(context, node)
        logger.debug(
            "[%s] Tool node config: %s inputs_declared=%s",
            node.node_id,
            node.config,
            node.inputs,
        )
        tool_name = None
        if isinstance(node.config, dict):
            tool_name = node.config.get("tool_name")
        if not tool_name and isinstance(node.metadata, dict):
            tool_name = node.metadata.get("tool_name") or node.metadata.get("skill_name")

        if not tool_name:
            node_exec.error = "Tool name missing"
            node_exec.outputs = ToolNodeOutputIR(
                output={"error": "tool_name_missing"},
                is_error=True,
                metadata={"tool_name": None},
            ).model_dump()
            return

        normalized_tool = self._normalize_tool_name(tool_name)

        if normalized_tool == "web_search":
            run_settings = context.run_settings or {}
            if not run_settings and isinstance(context.execution.metadata, dict):
                run_settings = context.execution.metadata.get("run_settings") or {}
            provider_override = None
            if isinstance(run_settings, dict):
                provider_override = run_settings.get("web_search_provider")
            if isinstance(provider_override, str) and provider_override.strip():
                if not isinstance(node_exec.inputs, dict):
                    node_exec.inputs = {}
                if not node_exec.inputs.get("provider"):
                    node_exec.inputs["provider"] = provider_override.strip()

            # NOTE: Fresh replay should not use caches by default. web_search has an internal
            # process-level cache (LRU) which would otherwise be reused across replays.
            # Allow overriding this behavior for debugging via env var.
            replay_mode = None
            if isinstance(context.execution.metadata, dict):
                replay_mode = context.execution.metadata.get("replay_mode")
            allow_fresh_web_search_cache = (
                os.getenv("HOUYI_FRESH_REPLAY_USE_WEB_SEARCH_CACHE") or ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if replay_mode == "fresh" and not allow_fresh_web_search_cache:
                if not isinstance(node_exec.inputs, dict):
                    node_exec.inputs = {}
                node_exec.inputs.setdefault("use_cache", False)

        skill = DEFAULT_SKILL_REGISTRY.get(normalized_tool)
        if not skill:
            node_exec.error = f"Tool not registered: {tool_name}"
            node_exec.outputs = ToolNodeOutputIR(
                output={"error": "tool_not_registered"},
                is_error=True,
                metadata={"tool_name": tool_name},
            ).model_dump()
            return

        timeout = node.config.get("timeout", 30) if isinstance(node.config, dict) else 30
        max_retries = node.config.get("max_retries", 3) if isinstance(node.config, dict) else 3
        executor = SkillExecutor(max_retries=max_retries, timeout=float(timeout))
        if not node_exec.inputs:
            node_exec.inputs = self._build_inputs_from_context(context, skill)

        logger.debug(
            "[%s] Tool inputs prepared: tool=%s inputs=%s context_keys=%s",
            node.node_id,
            normalized_tool,
            node_exec.inputs or {},
            sorted(context.execution.context.keys()),
        )

        # Create tool span as child of current trace context
        tool_span = self._create_tool_span(normalized_tool)

        # Emit start span event so the frontend Timeline can show it
        obs = context.observation_service
        if tool_span and obs:
            await obs.emit(
                SpanUpdateEvent.from_span(
                    tool_span,
                    session_id=context.session_id,
                    execution_id=context.execution.execution_id,
                )
            )

        # Push tool span onto TraceContext so internal sub-spans (e.g. from
        # WebSearchService instrumentation) attach as children of the tool span.
        _tc_token = None
        if tool_span:
            _tc_token = TraceContext.push(tool_span)

        # Tool cache lookup: reuse results from prior executions within the same session.
        # Uses the same key format as ToolCallRunner for consistency.
        tool_cache = context.tool_cache
        replay_mode = None
        if isinstance(context.execution.metadata, dict):
            replay_mode = context.execution.metadata.get("replay_mode")
        allow_fresh_tool_cache = (
            os.getenv("HOUYI_FRESH_REPLAY_USE_TOOL_CACHE") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if replay_mode == "fresh" and not allow_fresh_tool_cache:
            tool_cache = None

        cache_key = self._build_tool_cache_key(normalized_tool, node_exec.inputs or {})

        try:
            # Check tool cache before executing
            cached_result = None
            if tool_cache is not None and cache_key:
                cached_result = tool_cache.get(cache_key)

            if cached_result is not None:
                logger.info(
                    "[%s] Tool cache hit: tool=%s cache_key=%s",
                    node.node_id,
                    normalized_tool,
                    cache_key[:80] if cache_key else None,
                )
                result = dict(cached_result)
                result_meta = dict(result.get("metadata") or {})
                result_meta["cache_hit"] = True
                result_meta["cache_key"] = cache_key
                result["metadata"] = result_meta
            else:
                executor = SkillExecutor()
                result = await executor.execute(skill, node_exec.inputs)
                # Store in cache for future lookups
                if (
                    tool_cache is not None
                    and cache_key
                    and not (isinstance(result, dict) and result.get("error"))
                ):
                    tool_cache[cache_key] = result

            output_metadata: dict[str, Any] = {}
            if isinstance(result, dict):
                result_meta = result.get("metadata")
                if isinstance(result_meta, dict):
                    output_metadata.update(result_meta)
                raw_payload = result.get("raw")
                if isinstance(raw_payload, dict):
                    raw_meta = raw_payload.get("metadata")
                    if isinstance(raw_meta, dict):
                        for key in ("cache_hit", "cache_key", "cached"):
                            if key in raw_meta and key not in output_metadata:
                                output_metadata[key] = raw_meta[key]

                if output_metadata:
                    result_metadata = result.get("metadata")
                    if not isinstance(result_metadata, dict):
                        result_metadata = {}
                    for key in ("cache_hit", "cache_key", "cached"):
                        if key in output_metadata and key not in result_metadata:
                            result_metadata[key] = output_metadata[key]
                    result["metadata"] = result_metadata

                    raw_payload = result.get("raw")
                    if isinstance(raw_payload, dict):
                        raw_metadata = raw_payload.get("metadata")
                        if not isinstance(raw_metadata, dict):
                            raw_metadata = {}
                        for key in ("cache_hit", "cache_key", "cached"):
                            if key in output_metadata and key not in raw_metadata:
                                raw_metadata[key] = output_metadata[key]
                        raw_payload["metadata"] = raw_metadata
                        result["raw"] = raw_payload

            node_exec.outputs = ToolNodeOutputIR(
                output=result,
                is_error=False,
                metadata={"tool_name": normalized_tool, **output_metadata},
            ).model_dump()

            # Update tool span with cache hit info
            if tool_span and output_metadata.get("cache_hit") is True:
                tool_span.cache_hit = True
                tool_span.set_attribute("tool.cache_hit", True)
                logger.info(
                    "[%s] Tool cache hit: tool=%s cache_key=%s",
                    node.node_id,
                    normalized_tool,
                    output_metadata.get("cache_key"),
                )

            # End tool span on success and emit completion event
            if tool_span:
                tool_span.set_status("ok")
                tool_span.end()
                if obs:
                    await obs.emit(
                        SpanUpdateEvent.from_span(
                            tool_span,
                            session_id=context.session_id,
                            execution_id=context.execution.execution_id,
                        )
                    )
                    # Emit internal sub-spans created by tool instrumentation
                    await self._emit_child_spans(
                        tool_span, obs, context.session_id, context.execution.execution_id
                    )

        except Exception as e:
            node_exec.error = str(e)
            node_exec.outputs = ToolNodeOutputIR(
                output={"error": str(e)},
                is_error=True,
                metadata={"tool_name": normalized_tool},
            ).model_dump()

            # End tool span on error and emit event
            if tool_span:
                tool_span.set_status("error", str(e))
                tool_span.end()
                if obs:
                    await obs.emit(
                        SpanUpdateEvent.from_span(
                            tool_span,
                            session_id=context.session_id,
                            execution_id=context.execution.execution_id,
                        )
                    )
                    # Emit internal sub-spans even on error
                    await self._emit_child_spans(
                        tool_span, obs, context.session_id, context.execution.execution_id
                    )

        finally:
            # Always pop the tool span from TraceContext to prevent context leaks
            if _tc_token is not None:
                TraceContext.pop(_tc_token)


class VerifyNodeExecutor(NodeExecutor):
    """Executor for VERIFY nodes."""

    async def execute(
        self,
        context: ExecutionContext,
        node: NodeIR,
        node_exec: NodeExecutionIR,
    ) -> None:
        config = node.config if isinstance(node.config, dict) else {}
        payload = (node_exec.inputs or {}).get("output")
        rules = build_verification_rules(config)

        results: list[VerificationResult] = []
        for rule in rules:
            verifier_type = (rule.verifier_type or "").lower()
            if verifier_type == "sql":
                verifier = SQLVerifier()
                results.append(await verifier.verify(payload, rule))
                continue

            if verifier_type == "python":
                verifier = PythonVerifier()
                results.append(await verifier.verify(payload, rule))
                continue

            # Default + backward compatible: "constraint"
            # NOTE: ConstraintChecker expects python types in rule_spec; for console JSON
            # plans we also support string-based constraint specs.
            if verifier_type == "constraint":
                rule_spec = rule.rule_spec or {}

                require_keys = rule_spec.get("require_keys")
                if isinstance(require_keys, list) and require_keys:
                    missing = [
                        k for k in require_keys if not isinstance(payload, dict) or k not in payload
                    ]
                    passed = len(missing) == 0
                    results.append(
                        VerificationResult(
                            rule_id=rule.rule_id,
                            passed=passed,
                            error_message=(
                                None
                                if passed
                                else f"Missing keys: {', '.join([str(k) for k in missing])}"
                            ),
                            error_type=None if passed else "missing_key",
                            severity=rule.severity,
                        )
                    )
                    continue

                min_items = rule_spec.get("min_items")
                min_items_path = rule_spec.get("min_items_path")
                if min_items is not None and isinstance(min_items_path, str) and min_items_path:
                    actual = None
                    if isinstance(payload, dict):
                        candidate = payload.get(min_items_path)
                        if isinstance(candidate, list):
                            actual = len(candidate)
                    passed = actual is not None and actual >= int(min_items)
                    results.append(
                        VerificationResult(
                            rule_id=rule.rule_id,
                            passed=passed,
                            error_message=(
                                None
                                if passed
                                else f"min_items violation: expected>={min_items} actual={actual}"
                            ),
                            error_type=None if passed else "min_items",
                            severity=rule.severity,
                            metadata={
                                "expected": int(min_items),
                                "actual": actual,
                                "path": min_items_path,
                            },
                        )
                    )
                    continue

                # Fall back to built-in constraint checker
                verifier = ConstraintChecker()
                results.append(await verifier.verify(payload, rule))
                continue

            # Unknown verifier type
            results.append(
                VerificationResult(
                    rule_id=rule.rule_id,
                    passed=False,
                    error_message=f"Unknown verifier_type: {rule.verifier_type}",
                    error_type="unknown_verifier",
                    severity="error",
                )
            )

        verified = all(r.passed for r in results) if results else True
        failed = [r for r in results if not r.passed]
        errors = [
            {
                "rule_id": r.rule_id,
                "error_type": r.error_type,
                "error_message": r.error_message,
                "fix_suggestion": r.fix_suggestion,
                "auto_fixable": r.auto_fixable,
                "severity": r.severity,
                "metadata": r.metadata,
            }
            for r in failed
        ]

        node_exec.outputs = {
            "verified": verified,
            "errors": errors if errors else None,
            "results": [r.model_dump(mode="json") for r in results],
        }

        if not verified and bool(config.get("raise_on_failure")):
            raise RuntimeError(f"VERIFY failed: {errors}")


class RouteNodeExecutor(NodeExecutor):
    """Executor for ROUTE nodes."""

    async def execute(
        self,
        context: ExecutionContext,
        node: NodeIR,
        node_exec: NodeExecutionIR,
    ) -> None:
        config = node.config if isinstance(node.config, dict) else {}
        inputs = node_exec.inputs or {}
        route_mode, condition, target_ids, matched_rule, rule_trace = evaluate_route(
            config=config,
            inputs=inputs,
        )

        disabled: list[str] = []
        if target_ids:
            for plan_node in context.plan.nodes:
                if plan_node.node_id in target_ids and plan_node.deleted_at is None:
                    plan_node.deleted_at = datetime.now(UTC)
                    disabled.append(plan_node.node_id)

        node_exec.outputs = {
            "condition": condition,
            "disabled_nodes": disabled,
            "route_mode": route_mode,
            "matched_rule": matched_rule,
            "trace": rule_trace if rule_trace else None,
        }


class LogicNodeExecutor(NodeExecutor):
    """Executor for LOGIC nodes."""

    async def execute(
        self,
        context: ExecutionContext,
        node: NodeIR,
        node_exec: NodeExecutionIR,
    ) -> None:
        config = node.config if isinstance(node.config, dict) else {}
        template = config.get("template")
        payload: dict[str, Any] = {}
        if isinstance(context.execution.context, dict):
            payload.update(context.execution.context)
        if isinstance(node_exec.inputs, dict):
            payload.update(node_exec.inputs)

        if isinstance(template, str) and template:
            try:
                rendered = template.format_map(payload)
            except Exception:
                rendered = template
            node_exec.outputs = {"result": rendered}
            return

        node_exec.outputs = {"result": payload}
