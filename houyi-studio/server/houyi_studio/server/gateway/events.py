"""Server events sent to frontend clients."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from houyi.interface.protocol.ir.checkpoint_ir import LLMCallLog
from houyi.interface.protocol.ir.execution_ir import ExecutionStatus, NodeStatus
from houyi.interface.protocol.ir.plan_ir import PlanIR

if TYPE_CHECKING:
    from houyi.infrastructure.observability import Span


class EventType(str, Enum):
    """Types of server events."""

    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    NODE_STATUS = "node_status"
    STREAMING_OUTPUT = "streaming_output"
    CHECKPOINT_CREATED = "checkpoint_created"
    EXECUTION_STATUS = "execution_status"
    RETRY_STATUS = "retry_status"
    RETRY_SUCCESS = "retry_success"
    RETRY_FAILED = "retry_failed"
    RESTORE_CHECKPOINT_RESULT = "restore_checkpoint_result"
    WORKFLOW_LIST = "workflow_list"
    CONFLICT = "conflict"
    LOG_LEVEL = "log_level"
    SPAN_UPDATE = "span_update"
    # Knowledge Base events
    KNOWLEDGE_LIBRARY_LIST = "knowledge_library_list"
    KNOWLEDGE_LIBRARY_CREATED = "knowledge_library_created"
    KNOWLEDGE_LIBRARY_UPDATED = "knowledge_library_updated"
    KNOWLEDGE_LIBRARY_DELETED = "knowledge_library_deleted"
    KNOWLEDGE_SEARCH_RESULTS = "knowledge_search_results"
    KNOWLEDGE_INGEST_PROGRESS = "knowledge_ingest_progress"
    KNOWLEDGE_INGEST_COMPLETE = "knowledge_ingest_complete"
    KNOWLEDGE_ERROR = "knowledge_error"
    # Document management events
    DOCUMENT_LIST = "document_list"
    DOCUMENT_DETAIL = "document_detail"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_STATUS_CHANGED = "document_status_changed"
    CHUNK_LIST = "chunk_list"
    CHUNK_PREVIEW = "chunk_preview"

    # SimpleSkill Console integration events
    SKILL_LIST = "skill_list"
    SKILL_DETAIL = "skill_detail"
    SKILL_METRICS = "skill_metrics"
    SKILL_LOADED = "skill_loaded"
    SKILL_UNLOADED = "skill_unloaded"
    SKILL_ERROR = "skill_error"
    CONSENT_REQUESTED = "consent_requested"
    CONSENT_RESULT = "consent_result"
    SKILL_CONFIGURED = "skill_configured"
    SKILL_BLOCKED = "skill_blocked"
    DRY_RUN_RESULT = "dry_run_result"


class ServerEvent(BaseModel):
    """Base class for server events."""

    event_type: EventType = Field(..., description="Type of event")
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp",
    )
    session_id: str = Field(..., description="Session identifier")


class PlanCreatedEvent(ServerEvent):
    """Event when a new plan is created."""

    event_type: EventType = Field(default=EventType.PLAN_CREATED)
    plan: PlanIR = Field(..., description="Created plan")


class PlanUpdatedEvent(ServerEvent):
    """Event when a plan is updated (runtime editing)."""

    event_type: EventType = Field(default=EventType.PLAN_UPDATED)
    plan: PlanIR = Field(..., description="Updated plan")
    changes: list[str] = Field(
        default_factory=list,
        description="Description of changes made",
    )


class NodeStatusEvent(ServerEvent):
    """Event when a node's execution status changes."""

    event_type: EventType = Field(default=EventType.NODE_STATUS)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    status: NodeStatus = Field(..., description="New status")

    # Optional fields depending on status
    inputs: dict[str, Any] | None = Field(
        default=None,
        description="Node inputs (when starting)",
    )
    outputs: dict[str, Any] | None = Field(
        default=None,
        description="Node outputs (when completed)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (when failed)",
    )
    execution_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Execution-level metadata snapshot",
    )

    # NOTE(core): Optional observability payload (OTel-aligned). Kept generic for incremental rollout.
    # Expected shape (minimum): trace_id/span_id/parent_span_id/span_type/start_time/end_time/status
    # Extended fields (phase 1): tokens_*, cost_*, cache_hit; reasoning artifacts are sent but collapsed by default in UI.
    observation: dict[str, Any] | None = Field(
        default=None,
        description="Optional observability payload (trace/span + AI cost/reasoning artifacts)",
    )


class StreamingOutputEvent(ServerEvent):
    """Event for streaming LLM output."""

    event_type: EventType = Field(default=EventType.STREAMING_OUTPUT)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    chunk: str = Field(..., description="Output chunk")
    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata payload for final chunk (usage/trace_id/etc.)",
    )


class CheckpointCreatedEvent(ServerEvent):
    """Event when a checkpoint is created."""

    event_type: EventType = Field(default=EventType.CHECKPOINT_CREATED)
    checkpoint_id: str = Field(..., description="Checkpoint identifier")
    execution_id: str = Field(..., description="Execution identifier")
    sequence_number: int = Field(..., description="Checkpoint sequence number")
    trigger: str = Field(..., description="What triggered the checkpoint")
    llm_call_logs: list[LLMCallLog] = Field(
        default_factory=list,
        description="LLM call logs captured at this checkpoint",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Checkpoint metadata (e.g. trigger_node_id)",
    )


class ExecutionStatusEvent(ServerEvent):
    """Event when overall execution status changes."""

    event_type: EventType = Field(default=EventType.EXECUTION_STATUS)
    execution_id: str = Field(..., description="Execution identifier")
    status: ExecutionStatus = Field(..., description="New execution status")
    message: str | None = Field(
        default=None,
        description="Optional status message",
    )


class RestoreCheckpointResultEvent(ServerEvent):
    """Event emitted after a restore_checkpoint command is processed."""

    event_type: EventType = Field(default=EventType.RESTORE_CHECKPOINT_RESULT)
    checkpoint_id: str = Field(..., description="Checkpoint identifier")
    execution_id: str | None = Field(
        default=None,
        description="Execution identifier (if known / restored)",
    )
    replay_mode: str | None = Field(
        default=None,
        description="Replay mode requested (deterministic/fresh)",
    )
    success: bool = Field(..., description="Whether restore succeeded")
    message: str | None = Field(default=None, description="Optional details")


class RetryStatusEvent(ServerEvent):
    """Event when a retry attempt is scheduled or executed."""

    event_type: EventType = Field(default=EventType.RETRY_STATUS)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    attempt: int = Field(..., description="Current retry attempt (1-indexed)")
    max_retries: int = Field(..., description="Maximum retries allowed")
    error: str | None = Field(
        default=None,
        description="Error that triggered the retry",
    )


class RetrySuccessEvent(ServerEvent):
    """Event when a retry eventually succeeds."""

    event_type: EventType = Field(default=EventType.RETRY_SUCCESS)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    attempt: int = Field(..., description="Attempt that succeeded (1-indexed)")


class RetryFailedEvent(ServerEvent):
    """Event when retries are exhausted without success."""

    event_type: EventType = Field(default=EventType.RETRY_FAILED)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    max_retries: int = Field(..., description="Maximum retries attempted")
    error: str | None = Field(
        default=None,
        description="Final error after retries",
    )


class WorkflowListEvent(ServerEvent):
    """Event when workflow list is requested."""

    event_type: EventType = Field(default=EventType.WORKFLOW_LIST)
    workflows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of saved workflows",
    )


class ConflictEvent(ServerEvent):
    """Event when a plan modification conflicts (DECISION-002)."""

    event_type: EventType = Field(default=EventType.CONFLICT)
    your_version: int = Field(..., description="Client's base version")
    server_version: int = Field(..., description="Current server version")
    server_plan: PlanIR = Field(..., description="Current server plan")
    your_changes: list[dict[str, Any]] = Field(
        ...,
        description="Client's attempted changes",
    )


class LogLevelEvent(ServerEvent):
    """Event when server log level is updated or synced."""

    event_type: EventType = Field(default=EventType.LOG_LEVEL)
    level: str = Field(..., description="Effective log level")
    requested_level: str | None = Field(
        default=None,
        description="Requested log level (if provided by client)",
    )


class SpanUpdateEvent(ServerEvent):
    """Event for fine-grained span updates (llm/tool sub-spans).

    Enables real-time timeline visualization with:
    - Hierarchical span tree (execution -> node -> llm/tool)
    - AI-native fields (tokens, cost, cache_hit)
    - Checkpoint lineage tracking
    """

    event_type: EventType = Field(default=EventType.SPAN_UPDATE)
    execution_id: str = Field(..., description="Execution identifier")
    trace_id: str = Field(..., description="Trace identifier (= execution_id)")
    span_id: str = Field(..., description="Unique span identifier")
    parent_span_id: str | None = Field(
        default=None,
        description="Parent span ID (None for root execution span)",
    )
    span_type: str = Field(
        ...,
        description="Span type: execution/node/llm/tool/retriever",
    )
    name: str = Field(..., description="Span name (e.g., 'node.llm', 'llm.completion')")
    status: str = Field(
        default="ok",
        description="Span status: ok/error",
    )
    start_time: float = Field(..., description="Start timestamp (epoch seconds)")
    end_time: float | None = Field(
        default=None,
        description="End timestamp (None if still running)",
    )

    # AI-native fields
    node_id: str | None = Field(default=None, description="IR node_id if applicable")
    model: str | None = Field(default=None, description="LLM model name")
    tokens_input: int | None = Field(default=None, description="Input token count")
    tokens_output: int | None = Field(default=None, description="Output token count")
    cost_usd: float | None = Field(default=None, description="Cost in USD")
    cache_hit: bool | None = Field(default=None, description="Whether cache was hit")
    tool_name: str | None = Field(default=None, description="Tool/skill name")

    # Checkpoint lineage
    parent_trace_id: str | None = Field(
        default=None,
        description="Original trace ID if restored from checkpoint",
    )
    restore_checkpoint_id: str | None = Field(
        default=None,
        description="Checkpoint ID restored from",
    )
    replay_mode: bool = Field(
        default=False,
        description="Whether this is a deterministic replay",
    )

    # Generic attributes
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional span attributes",
    )

    @classmethod
    def from_span(
        cls,
        span: Span,
        session_id: str,
        execution_id: str,
    ) -> SpanUpdateEvent:
        """Create SpanUpdateEvent from a Span object.

        Args:
            span: Span object to convert
            session_id: Session identifier
            execution_id: Execution identifier

        Returns:
            SpanUpdateEvent instance
        """
        tokens_input = None
        tokens_output = None
        if span.tokens is not None:
            tokens_input = span.tokens.input
            tokens_output = span.tokens.output

        cost_usd = None
        if span.cost is not None:
            cost_usd = span.cost.usd

        return cls(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            execution_id=execution_id,
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_id,
            span_type=span.span_type.value
            if hasattr(span.span_type, "value")
            else str(span.span_type),
            name=span.name,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            node_id=span.node_id,
            model=span.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            cache_hit=span.cache_hit,
            tool_name=span.tool_name,
            parent_trace_id=span.parent_trace_id,
            restore_checkpoint_id=span.restore_checkpoint_id,
            replay_mode=span.replay_mode,
            attributes=span.attributes,
        )


# ============================================================================
# Knowledge Base Events
# ============================================================================


class KnowledgeLibraryListEvent(ServerEvent):
    """Event when knowledge library list is requested."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_LIBRARY_LIST)
    libraries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of knowledge libraries",
    )


class KnowledgeLibraryCreatedEvent(ServerEvent):
    """Event when a new knowledge library is created."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_LIBRARY_CREATED)
    library: dict[str, Any] = Field(..., description="Created library metadata")


class KnowledgeLibraryDeletedEvent(ServerEvent):
    """Event when a knowledge library is deleted."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_LIBRARY_DELETED)
    library_id: str = Field(..., description="Deleted library ID")


class KnowledgeSearchResultsEvent(ServerEvent):
    """Event with knowledge search results."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_SEARCH_RESULTS)
    query: str = Field(..., description="Search query")
    library_id: str = Field(default="", description="Library ID searched")
    results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Search results",
    )
    mode_used: str = Field(default="", description="RAG mode used for search")
    total_results: int = Field(default=0, description="Total number of results")
    quality: dict[str, Any] | None = Field(
        default=None, description="Quality assessment summary (v1.1)"
    )


class KnowledgeErrorEvent(ServerEvent):
    """Event when a knowledge operation fails."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_ERROR)
    error: str = Field(..., description="Error message")
    operation: str = Field(..., description="Operation that failed")


class KnowledgeLibraryUpdatedEvent(ServerEvent):
    """Event when a knowledge library is updated."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_LIBRARY_UPDATED)
    library: dict[str, Any] = Field(..., description="Updated library metadata")


class KnowledgeIngestProgressEvent(ServerEvent):
    """Event for ingest progress updates."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_INGEST_PROGRESS)
    library_id: str = Field(..., description="Library ID being ingested")
    progress: float = Field(..., description="Progress percentage (0-100)")
    current_file: str = Field(default="", description="Current file being processed")
    files_processed: int = Field(default=0, description="Number of files processed")
    total_files: int = Field(default=0, description="Total number of files")


class KnowledgeIngestCompleteEvent(ServerEvent):
    """Event when ingest is complete."""

    event_type: EventType = Field(default=EventType.KNOWLEDGE_INGEST_COMPLETE)
    library_id: str = Field(..., description="Library ID")
    success: bool = Field(..., description="Whether ingest succeeded")
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Ingest statistics (docs, chunks, errors)",
    )
    message: str = Field(default="", description="Completion message")
    warning: str | None = Field(
        default=None,
        description="Warning message when operation succeeded but with degraded quality "
        "(e.g. no embedding provider, search unavailable)",
    )


# ========== Document Management Events ==========


class DocumentListEvent(ServerEvent):
    """Event containing list of documents in a library."""

    event_type: EventType = Field(default=EventType.DOCUMENT_LIST)
    library_id: str = Field(..., description="Library ID")
    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of document metadata",
    )


class DocumentDetailEvent(ServerEvent):
    """Event containing document details."""

    event_type: EventType = Field(default=EventType.DOCUMENT_DETAIL)
    library_id: str = Field(..., description="Library ID")
    document: dict[str, Any] = Field(..., description="Document metadata")


class DocumentDeletedEvent(ServerEvent):
    """Event when a document is deleted."""

    event_type: EventType = Field(default=EventType.DOCUMENT_DELETED)
    library_id: str = Field(..., description="Library ID")
    doc_id: str = Field(..., description="Deleted document ID")


class DocumentStatusChangedEvent(ServerEvent):
    """Event when document status changes."""

    event_type: EventType = Field(default=EventType.DOCUMENT_STATUS_CHANGED)
    library_id: str = Field(..., description="Library ID")
    doc_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="New status")


class ChunkListEvent(ServerEvent):
    """Event containing list of chunks for a document."""

    event_type: EventType = Field(default=EventType.CHUNK_LIST)
    library_id: str = Field(..., description="Library ID")
    doc_id: str = Field(..., description="Document ID")
    chunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of chunk metadata",
    )


class ChunkPreviewEvent(ServerEvent):
    """Event containing chunk preview results."""

    event_type: EventType = Field(default=EventType.CHUNK_PREVIEW)
    chunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Preview chunks",
    )
    chunk_size: int = Field(..., description="Chunk size used")
    chunk_overlap: int = Field(..., description="Chunk overlap used")
    strategy: str = Field(..., description="Chunking strategy used")


# =============================================================================
# SimpleSkill Console Integration Events
# =============================================================================


class SkillSummary(BaseModel):
    """Summary of a skill for list view."""

    name: str = Field(..., description="Skill unique identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str | None = Field(default=None, description="Short description")
    tools: list[str] = Field(default_factory=list, description="Tools provided by skill")
    policy_action: str = Field(
        default="allow",
        description="Default policy action (allow/allow_with_consent/deny)",
    )
    side_effect: str = Field(default="none", description="Side effect type")
    certification: str = Field(
        default="unverified",
        description="Certification level (unverified/bronze/silver/gold)",
    )
    is_core: bool = Field(default=False, description="Whether this is a host core built-in skill")
    source: str = Field(
        default="local",
        description="Skill source classification (builtin/community/third_party/local)",
    )
    source_group: str | None = Field(
        default=None,
        description="Optional package/group key for external skills",
    )
    capability_tier: str = Field(
        default="metadata",
        description="How deeply integrated: metadata / schema / executable",
    )
    runtime_status: str = Field(
        default="unavailable",
        description="Operational readiness: ready / degraded / unavailable",
    )
    is_external_alias: bool = Field(
        default=False,
        description="Whether this skill name is an ext__ alias",
    )
    alias_target: str | None = Field(
        default=None,
        description="Core name targeted by ext__ alias, if any",
    )
    instructions_length: int = Field(
        default=0,
        description="Length of parsed markdown instructions body",
    )
    runtime_binding: str = Field(
        default="none",
        description="Runtime binding mode: python_executor | prompt_instructions | none",
    )


class SkillListEvent(ServerEvent):
    """Event containing list of registered skills."""

    event_type: EventType = Field(default=EventType.SKILL_LIST)
    skills: list[SkillSummary] = Field(
        default_factory=list,
        description="List of skill summaries",
    )


class SkillPermission(BaseModel):
    """A permission required by a skill."""

    name: str = Field(..., description="Permission name")
    description: str | None = Field(default=None, description="Why permission is needed")
    is_sensitive: bool = Field(default=False, description="Whether this is a sensitive permission")


class SkillDetail(BaseModel):
    """Full detail of a single skill."""

    name: str = Field(..., description="Skill unique identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str | None = Field(default=None, description="Full description")
    version: str | None = Field(default=None, description="Skill version")
    author: str | None = Field(default=None, description="Skill author")
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools with full schema",
    )
    permissions: list[SkillPermission] = Field(
        default_factory=list,
        description="Required permissions",
    )
    policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Invocation policy details",
    )
    hooks: list[str] = Field(default_factory=list, description="Registered hook names")
    certification: str = Field(default="unverified", description="Certification level")
    side_effect: str = Field(default="none", description="Side effect type")
    is_core: bool = Field(default=False, description="Whether this is a host core built-in skill")
    source: str = Field(
        default="local",
        description="Skill source classification (builtin/community/third_party/local)",
    )
    source_group: str | None = Field(
        default=None,
        description="Optional package/group key for external skills",
    )
    capability_tier: str = Field(
        default="metadata",
        description="How deeply integrated: metadata / schema / executable",
    )
    runtime_status: str = Field(
        default="unavailable",
        description="Operational readiness: ready / degraded / unavailable",
    )
    is_external_alias: bool = Field(
        default=False,
        description="Whether this skill name is an ext__ alias",
    )
    alias_target: str | None = Field(
        default=None,
        description="Core name targeted by ext__ alias, if any",
    )
    instructions_length: int = Field(
        default=0,
        description="Length of parsed markdown instructions body",
    )
    runtime_binding: str = Field(
        default="none",
        description="Runtime binding mode: python_executor | prompt_instructions | none",
    )
    instructions: str | None = Field(
        default=None,
        description="Parsed instructions body from SKILL.md",
    )
    hook_specs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured hook specs including matcher/type/command/handler",
    )
    package_examples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Package-native dry-run examples loaded from skill assets",
    )
    available_workflows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted workflow candidates for script-compatible skills",
    )


class SkillDetailEvent(ServerEvent):
    """Event containing full detail of a skill."""

    event_type: EventType = Field(default=EventType.SKILL_DETAIL)
    skill: SkillDetail = Field(..., description="Skill detail")


class SkillMetricsData(BaseModel):
    """Metrics data for a skill."""

    skill_name: str = Field(..., description="Skill name")
    total_calls: int = Field(default=0, description="Total invocations")
    success_count: int = Field(default=0, description="Successful invocations")
    failure_count: int = Field(default=0, description="Failed invocations")
    avg_latency_ms: float = Field(default=0.0, description="Average latency in milliseconds")
    p50_latency_ms: float = Field(default=0.0, description="P50 latency")
    p99_latency_ms: float = Field(default=0.0, description="P99 latency")
    success_rate: float = Field(default=0.0, description="Success rate (0.0-1.0)")
    last_invoked: datetime | None = Field(default=None, description="Last invocation time")


class SkillMetricsEvent(ServerEvent):
    """Event containing metrics for a skill."""

    event_type: EventType = Field(default=EventType.SKILL_METRICS)
    metrics: SkillMetricsData = Field(..., description="Skill metrics")


class SkillLoadedEvent(ServerEvent):
    """Event when a skill is successfully loaded."""

    event_type: EventType = Field(default=EventType.SKILL_LOADED)
    skill_name: str = Field(..., description="Loaded skill name")
    message: str | None = Field(default=None, description="Optional success message")


class SkillUnloadedEvent(ServerEvent):
    """Event when a skill is unloaded."""

    event_type: EventType = Field(default=EventType.SKILL_UNLOADED)
    skill_name: str = Field(..., description="Unloaded skill name")


class SkillConfiguredEvent(ServerEvent):
    """Event when a skill configuration is updated."""

    event_type: EventType = Field(default=EventType.SKILL_CONFIGURED)
    skill_name: str = Field(..., description="Configured skill name")
    policy_action: str | None = Field(default=None, description="New policy action")
    auto_invoke: bool | None = Field(default=None, description="New auto-invoke setting")
    message: str = Field(default="", description="Success message")


class SkillErrorEvent(ServerEvent):
    """Event when a skill operation fails."""

    event_type: EventType = Field(default=EventType.SKILL_ERROR)
    skill_name: str | None = Field(default=None, description="Skill name (if applicable)")
    error_code: str = Field(..., description="Error code (skill_not_found, load_failed, etc.)")
    message: str = Field(..., description="Human-readable error message")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested actions",
    )


class ConsentRequestedEvent(ServerEvent):
    """Event when user consent is required for a skill operation."""

    event_type: EventType = Field(default=EventType.CONSENT_REQUESTED)
    request_id: str = Field(..., description="Unique consent request identifier")
    skill_name: str = Field(..., description="Skill requesting consent")
    tool_name: str = Field(..., description="Specific tool requiring consent")
    reason: str = Field(..., description="Why consent is needed")
    permissions: list[str] = Field(
        default_factory=list,
        description="Permissions being requested",
    )
    timeout_seconds: int = Field(default=60, description="Consent request timeout")


class ConsentResultEvent(ServerEvent):
    """Event reporting the outcome of a consent request."""

    event_type: EventType = Field(default=EventType.CONSENT_RESULT)
    request_id: str = Field(..., description="Consent request identifier")
    granted: bool = Field(..., description="Whether consent was granted")
    remembered: bool = Field(default=False, description="Whether choice was remembered")


class SkillBlockedEvent(ServerEvent):
    """Event when a skill invocation is blocked by policy."""

    event_type: EventType = Field(default=EventType.SKILL_BLOCKED)
    skill_name: str = Field(..., description="Blocked skill name")
    tool_name: str = Field(..., description="Blocked tool name")
    policy_action: str = Field(..., description="Policy action that blocked (deny)")
    reason: str = Field(..., description="Reason for blocking")


class DisclosurePhase(BaseModel):
    """A single phase in the progressive disclosure timeline."""

    name: str = Field(
        ..., description="Phase identifier (discovery/activation/negotiation/execution)"
    )
    label: str = Field(..., description="Human-readable phase label")
    timestamp_ms: int = Field(..., description="Wall-clock ms since dry-run start")
    status: str = Field(default="pass", description="Phase status: pass/fail")
    data: dict[str, Any] = Field(default_factory=dict, description="Phase-specific payload")


class LlmVerificationResult(BaseModel):
    """Result of live LLM verification in a dry-run.

    Includes a phases timeline for progressive disclosure:
      1. discovery   — skill metadata loaded
      2. activation  — tool definitions built
      3. negotiation — system + user prompt constructed
      4. execution   — real LLM API call completed
    """

    success: bool = Field(..., description="Whether LLM produced correct tool call")
    message: str = Field(default="", description="Verification status message")
    tool_call: dict[str, Any] | None = Field(
        default=None,
        description="The tool call the LLM generated, if any",
    )
    probe_prompt: str | None = Field(
        default=None,
        description="The natural-language user query sent to the LLM",
    )
    system_prompt: str | None = Field(
        default=None,
        description="The system prompt providing skill context to the LLM",
    )
    tool_definitions: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI-format tool definitions sent to the LLM",
    )
    model_name: str | None = Field(
        default=None,
        description="The LLM model used for verification",
    )
    raw_content: str | None = Field(
        default=None,
        description="Raw text content from the LLM response (if any)",
    )
    usage: dict[str, Any] | None = Field(
        default=None,
        description="Token usage stats from the LLM call",
    )
    requested_input: dict[str, Any] | None = Field(
        default=None,
        description="Exact input payload requested in the LLM probe prompt",
    )
    phases: list[DisclosurePhase] = Field(
        default_factory=list,
        description="Progressive disclosure timeline phases with timestamps",
    )


class DryRunResult(BaseModel):
    """Result of a dry-run validation."""

    valid: bool = Field(..., description="Whether input passes validation")
    schema_errors: list[str] = Field(
        default_factory=list,
        description="Schema validation errors",
    )
    policy_result: str = Field(
        default="allow",
        description="Policy evaluation result",
    )
    capability_gaps: list[str] = Field(
        default_factory=list,
        description="Missing host capabilities",
    )
    estimated_side_effects: list[str] = Field(
        default_factory=list,
        description="Predicted side effects",
    )
    available_workflows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted workflow candidates for script-compatible skills",
    )
    llm_verification: LlmVerificationResult | None = Field(
        default=None,
        description="Live LLM verification result (only when live=True)",
    )


class DryRunResultEvent(ServerEvent):
    """Event containing dry-run validation result."""

    event_type: EventType = Field(default=EventType.DRY_RUN_RESULT)
    skill_name: str = Field(..., description="Skill name")
    result: DryRunResult = Field(..., description="Dry-run result")
