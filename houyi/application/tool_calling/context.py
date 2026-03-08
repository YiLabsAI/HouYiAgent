from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.domain.skill.spec import SkillSpec


@dataclass(frozen=True)
class ToolLoopConfig:
    # Maximum number of tool-call loop rounds allowed for one run.
    tool_loop_max_rounds: int
    # Enable parallel tool execution when policy allows it.
    tool_loop_enable_parallel_calls: bool
    # Upper bound for concurrent tool calls in one parallel batch.
    tool_loop_max_parallel_calls: int
    # Enable placeholder fast-path behavior in tool loop.
    tool_loop_enable_fast_path: bool
    # Enable timing/performance logs for loop and tool phases.
    tool_loop_enable_timing: bool
    # Enable summarized tool-result payload in trace/messages.
    tool_loop_result_summary_enabled: bool
    # Maximum characters kept per summarized tool result.
    tool_loop_result_summary_max_chars: int
    # Maximum items kept in summarized tool results.
    tool_loop_result_summary_max_items: int
    # Per-message character budget before truncation for LLM input.
    tool_loop_max_message_chars: int
    # Total input character budget before truncation for LLM input.
    tool_loop_max_total_chars: int
    # Artificial latency injection per tool execution for debugging/simulation.
    tool_loop_injected_tool_latency_seconds: float | None


@dataclass
class ToolLoopState:
    # Mutable conversation messages accumulated during loop execution.
    tool_loop_messages: list[dict[str, Any]]
    # Mutable tool trace entries collected across rounds.
    tool_loop_trace_entries: list[dict[str, Any]]
    # Names of tools invoked so far in the current loop.
    tool_loop_invoked_tool_names: set[str]
    # Resolved outputs keyed by tool name for placeholder substitution.
    tool_loop_resolved_outputs_by_tool: dict[str, Any]
    # Monotonic start timestamp for performance reporting.
    tool_loop_started_at_monotonic: float | None = None


@dataclass(frozen=True)
class ToolLoopRuntimeServices:
    # Adapter used for model inference requests.
    model_adapter: Any
    # Runtime tool executor used to invoke concrete skills/tools.
    tool_executor: Any
    # Tool schemas exposed to the model for tool selection.
    available_tool_schemas: list[dict[str, Any]]
    # Skill specifications indexed by skill name.
    skill_specs_by_name: dict[str, SkillSpec]
    # Set of all available tool names for fast-path termination checks.
    available_tool_names: set[str]
    # Model request options forwarded to adapter calls.
    model_request_options: dict[str, Any]
    # Optional LLM response cache for repeated model calls.
    llm_response_cache: dict[str, Any] | None
    # Optional tool result cache used by tool execution path.
    tool_result_cache: dict[str, dict[str, Any]] | None
    # Hook chain applied before/after tool execution.
    tool_call_hooks: list[Any]
    # Whether tool replacement is allowed by runtime policy.
    allow_tool_replacement: bool


@dataclass
class ToolLoopContext:
    # Runner facade coordinating loop-level collaborators and runtime services.
    runner: Any
    # Static policy and budget configuration for this loop.
    config: ToolLoopConfig
    # Mutable loop state shared across rounds.
    state: ToolLoopState
    # Runtime service collaborators and external adapters.
    services: ToolLoopRuntimeServices


@dataclass(frozen=True)
class ToolRoundPhaseConfig:
    # Zero-based round index for current tool phase.
    round_index: int
    # Enable timing/performance logs for this round.
    tool_loop_enable_timing: bool
    # Enable parallel tool execution when policy allows it.
    tool_loop_enable_parallel_calls: bool
    # Enable placeholder fast-path behavior in tool phase.
    tool_loop_enable_fast_path: bool
    # Upper bound for concurrent tool calls in one parallel batch.
    tool_loop_max_parallel_calls: int
    # Maximum number of tool-call loop rounds allowed for one run.
    tool_loop_max_rounds: int
    # Enable summarized tool-result payload in trace/messages.
    tool_loop_result_summary_enabled: bool
    # Maximum characters kept per summarized tool result.
    tool_loop_result_summary_max_chars: int
    # Maximum items kept in summarized tool results.
    tool_loop_result_summary_max_items: int
    # Artificial latency injection per tool execution for debugging/simulation.
    tool_loop_injected_tool_latency_seconds: float | None


@dataclass
class ToolRoundPhaseState:
    # LLM response that contains tool calls for this round.
    response: Any
    # Resolved outputs keyed by tool name for placeholder substitution.
    tool_loop_resolved_outputs_by_tool: dict[str, Any]
    # Names of tools invoked so far in the current loop.
    tool_loop_invoked_tool_names: set[str]
    # Mutable conversation messages accumulated during loop execution.
    tool_loop_messages: list[dict[str, Any]]
    # Mutable tool trace entries collected across rounds.
    tool_loop_trace_entries: list[dict[str, Any]]
    # Monotonic round start timestamp for performance reporting.
    tool_round_started_at_monotonic: float


@dataclass(frozen=True)
class ToolRoundPhaseServices:
    # Skill specifications indexed by skill name.
    skill_specs_by_name: dict[str, SkillSpec]
    # Runtime tool executor used to invoke concrete skills/tools.
    tool_executor: Any
    # Hook chain applied before/after tool execution.
    tool_call_hooks: list[Any]
    # Whether tool replacement is allowed by runtime policy.
    allow_tool_replacement: bool
    # Optional tool result cache used by tool execution path.
    tool_result_cache: dict[str, dict[str, Any]] | None
    # Set of all available tool names for fast-path termination checks.
    available_tool_names: set[str]


@dataclass
class ToolRoundPhaseContext:
    # Runner facade coordinating per-round execution collaborators.
    runner: Any
    # Static policy and budget configuration for this round phase.
    config: ToolRoundPhaseConfig
    # Mutable round-phase state.
    state: ToolRoundPhaseState
    # Runtime service collaborators and external adapters.
    services: ToolRoundPhaseServices


@dataclass(frozen=True)
class ToolCallBatchExecutionConfig:
    # One-based round index value used for trace/span attributes.
    round_index_value: int
    # Enable timing/performance logs for this batch execution.
    tool_loop_enable_timing: bool
    # Artificial latency injection per tool execution for debugging/simulation.
    tool_loop_injected_tool_latency_seconds: float | None
    # Enable summarized tool-result payload in trace/messages.
    tool_loop_result_summary_enabled: bool
    # Maximum characters kept per summarized tool result.
    tool_loop_result_summary_max_chars: int
    # Maximum items kept in summarized tool results.
    tool_loop_result_summary_max_items: int
    # Upper bound for concurrent tool calls in one parallel batch.
    tool_loop_max_parallel_calls: int


@dataclass
class ToolCallBatchExecutionState:
    # Parsed tool calls paired with pre-parsed arguments.
    parsed_tool_calls: list[tuple[Any, dict[str, Any] | None]]
    # Optional pre-resolved outputs used by serial execution mode.
    resolved_outputs: dict[str, Any] | None
    # Optional parallel group id for trace/span grouping.
    parallel_group_id: str | None


@dataclass(frozen=True)
class ToolCallBatchExecutionServices:
    # Skill specifications indexed by skill name.
    skill_specs_by_name: dict[str, SkillSpec]
    # Runtime tool executor used to invoke concrete skills/tools.
    tool_executor: Any
    # Hook chain applied before/after tool execution.
    tool_call_hooks: list[Any]
    # Whether tool replacement is allowed by runtime policy.
    allow_tool_replacement: bool
    # Optional tool result cache used by tool execution path.
    tool_result_cache: dict[str, dict[str, Any]] | None
    # Names of tools invoked so far in the current loop.
    tool_loop_invoked_tool_names: set[str]


@dataclass
class ToolCallBatchExecutionContext:
    # Runner facade coordinating batch execution collaborators.
    runner: Any
    # Static policy and batch execution configuration.
    config: ToolCallBatchExecutionConfig
    # Mutable batch execution state.
    state: ToolCallBatchExecutionState
    # Runtime service collaborators and external adapters.
    services: ToolCallBatchExecutionServices


@dataclass(frozen=True)
class ToolCallExecutionConfig:
    # Zero-based index of tool call within the current batch.
    index: int
    # One-based round index value used for trace/span attributes.
    round_index_value: int | None
    # Enable timing/performance logs for this tool call.
    tool_loop_enable_timing: bool
    # Artificial latency injection per tool execution for debugging/simulation.
    tool_loop_injected_tool_latency_seconds: float | None
    # Enable summarized tool-result payload in trace/messages.
    tool_loop_result_summary_enabled: bool
    # Maximum characters kept per summarized tool result.
    tool_loop_result_summary_max_chars: int
    # Maximum items kept in summarized tool results.
    tool_loop_result_summary_max_items: int


@dataclass
class ToolCallExecutionState:
    # Raw tool call payload emitted by the model.
    tool_call: Any
    # Parsed tool-call arguments, if already decoded.
    parsed_args: dict[str, Any] | None
    # Optional pre-resolved outputs used for placeholder substitution.
    resolved_outputs: dict[str, Any] | None
    # Optional parallel group id for trace/span grouping.
    parallel_group_id: str | None


@dataclass(frozen=True)
class ToolCallExecutionServices:
    # Skill specifications indexed by skill name.
    skill_specs_by_name: dict[str, SkillSpec]
    # Runtime tool executor used to invoke concrete skills/tools.
    tool_executor: Any
    # Hook chain applied before/after tool execution.
    tool_call_hooks: list[Any]
    # Whether tool replacement is allowed by runtime policy.
    allow_tool_replacement: bool
    # Optional tool result cache used by tool execution path.
    tool_result_cache: dict[str, dict[str, Any]] | None
    # Names of tools invoked so far in the current loop.
    tool_loop_invoked_tool_names: set[str]


@dataclass
class ToolCallExecutionContext:
    # Static configuration for one concrete tool call.
    config: ToolCallExecutionConfig
    # Mutable state for one concrete tool call.
    state: ToolCallExecutionState
    # Runtime service collaborators and external adapters.
    services: ToolCallExecutionServices


def build_tool_round_phase_context(
    *,
    loop_ctx: ToolLoopContext,
    response: Any,
    round_index: int,
    round_start: float,
) -> ToolRoundPhaseContext:
    """Build one round-phase context from loop context and current round state."""
    config = loop_ctx.config
    state = loop_ctx.state
    services = loop_ctx.services
    return ToolRoundPhaseContext(
        runner=loop_ctx.runner,
        config=ToolRoundPhaseConfig(
            round_index=round_index,
            tool_loop_enable_timing=config.tool_loop_enable_timing,
            tool_loop_enable_parallel_calls=config.tool_loop_enable_parallel_calls,
            tool_loop_enable_fast_path=config.tool_loop_enable_fast_path,
            tool_loop_max_parallel_calls=config.tool_loop_max_parallel_calls,
            tool_loop_max_rounds=config.tool_loop_max_rounds,
            tool_loop_result_summary_enabled=config.tool_loop_result_summary_enabled,
            tool_loop_result_summary_max_chars=config.tool_loop_result_summary_max_chars,
            tool_loop_result_summary_max_items=config.tool_loop_result_summary_max_items,
            tool_loop_injected_tool_latency_seconds=config.tool_loop_injected_tool_latency_seconds,
        ),
        state=ToolRoundPhaseState(
            response=response,
            tool_loop_resolved_outputs_by_tool=state.tool_loop_resolved_outputs_by_tool,
            tool_loop_invoked_tool_names=state.tool_loop_invoked_tool_names,
            tool_loop_messages=state.tool_loop_messages,
            tool_loop_trace_entries=state.tool_loop_trace_entries,
            tool_round_started_at_monotonic=round_start,
        ),
        services=ToolRoundPhaseServices(
            skill_specs_by_name=services.skill_specs_by_name,
            tool_executor=services.tool_executor,
            tool_call_hooks=services.tool_call_hooks,
            allow_tool_replacement=services.allow_tool_replacement,
            tool_result_cache=services.tool_result_cache,
            available_tool_names=services.available_tool_names,
        ),
    )


def build_tool_call_batch_execution_context(
    *,
    phase_ctx: ToolRoundPhaseContext,
    parsed_tool_calls: list[tuple[Any, dict[str, Any] | None]],
    resolved_outputs: dict[str, Any] | None,
    parallel_group_id: str | None,
) -> ToolCallBatchExecutionContext:
    """Build one batch execution context from a round-phase context."""
    config = phase_ctx.config
    state = phase_ctx.state
    services = phase_ctx.services
    return ToolCallBatchExecutionContext(
        runner=phase_ctx.runner,
        config=ToolCallBatchExecutionConfig(
            round_index_value=config.round_index + 1,
            tool_loop_enable_timing=config.tool_loop_enable_timing,
            tool_loop_injected_tool_latency_seconds=config.tool_loop_injected_tool_latency_seconds,
            tool_loop_result_summary_enabled=config.tool_loop_result_summary_enabled,
            tool_loop_result_summary_max_chars=config.tool_loop_result_summary_max_chars,
            tool_loop_result_summary_max_items=config.tool_loop_result_summary_max_items,
            tool_loop_max_parallel_calls=config.tool_loop_max_parallel_calls,
        ),
        state=ToolCallBatchExecutionState(
            parsed_tool_calls=parsed_tool_calls,
            resolved_outputs=resolved_outputs,
            parallel_group_id=parallel_group_id,
        ),
        services=ToolCallBatchExecutionServices(
            skill_specs_by_name=services.skill_specs_by_name,
            tool_executor=services.tool_executor,
            tool_call_hooks=services.tool_call_hooks,
            allow_tool_replacement=services.allow_tool_replacement,
            tool_result_cache=services.tool_result_cache,
            tool_loop_invoked_tool_names=state.tool_loop_invoked_tool_names,
        ),
    )


def build_tool_call_execution_context(
    *,
    batch_ctx: ToolCallBatchExecutionContext,
    index: int,
    tool_call: Any,
    parsed_args: dict[str, Any] | None,
    resolved_outputs: dict[str, Any] | None,
) -> ToolCallExecutionContext:
    """Build one tool-call execution context from a batch execution context."""
    config = batch_ctx.config
    state = batch_ctx.state
    services = batch_ctx.services
    return ToolCallExecutionContext(
        config=ToolCallExecutionConfig(
            index=index,
            round_index_value=config.round_index_value,
            tool_loop_enable_timing=config.tool_loop_enable_timing,
            tool_loop_injected_tool_latency_seconds=config.tool_loop_injected_tool_latency_seconds,
            tool_loop_result_summary_enabled=config.tool_loop_result_summary_enabled,
            tool_loop_result_summary_max_chars=config.tool_loop_result_summary_max_chars,
            tool_loop_result_summary_max_items=config.tool_loop_result_summary_max_items,
        ),
        state=ToolCallExecutionState(
            tool_call=tool_call,
            parsed_args=parsed_args,
            resolved_outputs=resolved_outputs,
            parallel_group_id=state.parallel_group_id,
        ),
        services=ToolCallExecutionServices(
            skill_specs_by_name=services.skill_specs_by_name,
            tool_executor=services.tool_executor,
            tool_call_hooks=services.tool_call_hooks,
            allow_tool_replacement=services.allow_tool_replacement,
            tool_result_cache=services.tool_result_cache,
            tool_loop_invoked_tool_names=services.tool_loop_invoked_tool_names,
        ),
    )


__all__ = [
    "ToolCallBatchExecutionConfig",
    "ToolCallBatchExecutionContext",
    "ToolCallBatchExecutionServices",
    "ToolCallBatchExecutionState",
    "ToolCallExecutionConfig",
    "ToolCallExecutionContext",
    "ToolCallExecutionServices",
    "ToolCallExecutionState",
    "ToolLoopConfig",
    "ToolLoopContext",
    "ToolLoopRuntimeServices",
    "ToolLoopState",
    "ToolRoundPhaseConfig",
    "ToolRoundPhaseContext",
    "ToolRoundPhaseServices",
    "ToolRoundPhaseState",
    "build_tool_call_batch_execution_context",
    "build_tool_call_execution_context",
    "build_tool_round_phase_context",
]
