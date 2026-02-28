/**
 * WebSocket message types
 * Matches backend event and command definitions
 */

import type { PlanIR, NodeStatus, ExecutionStatus, KnowledgeLibrary, KnowledgeSearchResult, KnowledgeDocument, KnowledgeChunk, ChunkPreview, QualitySummary } from './ir';

// Event types (server -> client)
export type EventType =
  | 'plan_created'
  | 'plan_updated'
  | 'node_status'
  | 'streaming_output'
  | 'checkpoint_created'
  | 'restore_checkpoint_result'
  | 'execution_status'
  | 'retry_status'
  | 'retry_success'
  | 'retry_failed'
  | 'workflow_list'
  | 'conflict'
  | 'log_level'
  | 'span_update'
  // Knowledge Base events
  | 'knowledge_library_list'
  | 'knowledge_library_created'
  | 'knowledge_library_updated'
  | 'knowledge_library_deleted'
  | 'knowledge_search_results'
  | 'knowledge_ingest_progress'
  | 'knowledge_ingest_complete'
  | 'knowledge_error'
  // Document management events
  | 'document_list'
  | 'document_detail'
  | 'document_deleted'
  | 'document_status_changed'
  | 'chunk_list'
  | 'chunk_preview'
  // SimpleSkill Console integration events
  | 'skill_list'
  | 'skill_detail'
  | 'skill_metrics'
  | 'skill_loaded'
  | 'skill_unloaded'
  | 'skill_error'
  | 'consent_requested'
  | 'consent_result'
  | 'skill_blocked'
  | 'dry_run_result';

export interface ServerEvent {
  event_type: EventType;
  event_id: string;
  timestamp: string;
  session_id: string;
}

export interface PlanCreatedEvent extends ServerEvent {
  event_type: 'plan_created';
  plan: PlanIR;
}

export interface PlanUpdatedEvent extends ServerEvent {
  event_type: 'plan_updated';
  plan: PlanIR;
  changes: string[];
}

export interface NodeStatusEvent extends ServerEvent {
  event_type: 'node_status';
  execution_id: string;
  node_id: string;
  status: NodeStatus;
  inputs?: Record<string, any>;
  outputs?: Record<string, any>;
  error?: string;
  execution_metadata?: Record<string, any>;
  // NOTE(core): Optional observability payload (OTel-aligned placeholder).
  // Shape is intentionally loose for incremental rollout; UI may ignore until Observability Mode is implemented.
  observation?: Record<string, any>;
}

export interface StreamingOutputEvent extends ServerEvent {
  event_type: 'streaming_output';
  execution_id: string;
  node_id: string;
  chunk: string;
  is_final: boolean;
}

export interface CheckpointCreatedEvent extends ServerEvent {
  event_type: 'checkpoint_created';
  checkpoint_id: string;
  execution_id: string;
  sequence_number: number;
  trigger: string;
  llm_call_logs?: any[];
}

export interface RestoreCheckpointResultEvent extends ServerEvent {
  event_type: 'restore_checkpoint_result';
  checkpoint_id: string;
  execution_id?: string | null;
  replay_mode?: 'deterministic' | 'fresh' | string | null;
  success: boolean;
  message?: string | null;
}

export interface ExecutionStatusEvent extends ServerEvent {
  event_type: 'execution_status';
  execution_id: string;
  status: ExecutionStatus;
  message?: string;
}

export interface RetryStatusEvent extends ServerEvent {
  event_type: 'retry_status';
  execution_id: string;
  node_id: string;
  attempt: number;
  max_retries: number;
  error?: string;
}

export interface RetrySuccessEvent extends ServerEvent {
  event_type: 'retry_success';
  execution_id: string;
  node_id: string;
  attempt: number;
}

export interface RetryFailedEvent extends ServerEvent {
  event_type: 'retry_failed';
  execution_id: string;
  node_id: string;
  max_retries: number;
  error?: string;
}

export interface ConflictEvent extends ServerEvent {
  event_type: 'conflict';
  your_version: number;
  server_version: number;
  server_plan: PlanIR;
  your_changes: any[];
}

export interface LogLevelEvent extends ServerEvent {
  event_type: 'log_level';
  level: 'debug' | 'info' | 'warning' | 'error';
  requested_level?: string | null;
}

// Span types for timeline visualization
export type SpanType = 'execution' | 'node' | 'llm' | 'tool' | 'retriever' | 'retry' | 'internal';

export interface SpanUpdateEvent extends ServerEvent {
  event_type: 'span_update';
  execution_id: string;
  trace_id: string;
  span_id: string;
  parent_span_id?: string | null;
  span_type: SpanType;
  name: string;
  status: 'ok' | 'error';
  start_time: number;
  end_time?: number | null;

  // AI-native fields
  node_id?: string | null;
  model?: string | null;
  tokens_input?: number | null;
  tokens_output?: number | null;
  cost_usd?: number | null;
  cache_hit?: boolean | null;
  tool_name?: string | null;

  // Checkpoint lineage
  parent_trace_id?: string | null;
  restore_checkpoint_id?: string | null;
  replay_mode?: boolean;

  // Generic attributes
  attributes?: Record<string, any>;
}

export interface WorkflowListEvent extends ServerEvent {
  event_type: 'workflow_list';
  workflows: Array<{
    name: string;
    saved_at: string;
    node_count?: number;
    edge_count?: number;
    nodes_count?: number;
  }>;
}

// ============================================================================
// Knowledge Base Events
// ============================================================================

export interface KnowledgeLibraryListEvent extends ServerEvent {
  event_type: 'knowledge_library_list';
  libraries: KnowledgeLibrary[];
}

export interface KnowledgeLibraryCreatedEvent extends ServerEvent {
  event_type: 'knowledge_library_created';
  library: KnowledgeLibrary;
}

export interface KnowledgeLibraryDeletedEvent extends ServerEvent {
  event_type: 'knowledge_library_deleted';
  library_id: string;
}

export interface KnowledgeSearchResultsEvent extends ServerEvent {
  event_type: 'knowledge_search_results';
  query: string;
  library_id: string;
  results: KnowledgeSearchResult[];
  mode_used: string;
  strategies_used?: string[]; 
  total_results: number;
  quality?: QualitySummary | null; 
}

export interface KnowledgeErrorEvent extends ServerEvent {
  event_type: 'knowledge_error';
  error: string;
  operation: string;
}

export interface KnowledgeLibraryUpdatedEvent extends ServerEvent {
  event_type: 'knowledge_library_updated';
  library: KnowledgeLibrary;
}

export interface KnowledgeIngestProgressEvent extends ServerEvent {
  event_type: 'knowledge_ingest_progress';
  library_id: string;
  progress: number;
  current_file: string;
  files_processed: number;
  total_files: number;
}

export interface KnowledgeIngestCompleteEvent extends ServerEvent {
  event_type: 'knowledge_ingest_complete';
  library_id: string;
  success: boolean;
  stats: Record<string, any>;
  message: string;
  warning?: string | null;
}

// ============================================================================
// Document Management Events
// ============================================================================

export interface DocumentListEvent extends ServerEvent {
  event_type: 'document_list';
  library_id: string;
  documents: KnowledgeDocument[];
}

export interface DocumentDetailEvent extends ServerEvent {
  event_type: 'document_detail';
  library_id: string;
  document: KnowledgeDocument;
}

export interface DocumentDeletedEvent extends ServerEvent {
  event_type: 'document_deleted';
  library_id: string;
  doc_id: string;
}

export interface DocumentStatusChangedEvent extends ServerEvent {
  event_type: 'document_status_changed';
  library_id: string;
  doc_id: string;
  status: string;
}

export interface ChunkListEvent extends ServerEvent {
  event_type: 'chunk_list';
  library_id: string;
  doc_id: string;
  chunks: KnowledgeChunk[];
}

export interface ChunkPreviewEvent extends ServerEvent {
  event_type: 'chunk_preview';
  chunks: ChunkPreview[];
  chunk_size: number;
  chunk_overlap: number;
  strategy: string;
}

// =============================================================================
// SimpleSkill Console Integration Events
// =============================================================================

export interface SkillSummary {
  name: string;
  display_name: string;
  description?: string;
  tools: string[];
  policy_action: 'allow' | 'allow_with_consent' | 'deny';
  side_effect: 'none' | 'network' | 'filesystem' | 'exec';
  certification: 'unverified' | 'bronze' | 'silver' | 'gold';
  is_core: boolean;
  source?: 'builtin' | 'community' | 'third_party' | 'local';
  capability_tier?: 'metadata' | 'schema' | 'executable';
  runtime_status?: 'ready' | 'degraded' | 'unavailable';
  is_external_alias?: boolean;
  alias_target?: string;
  instructions_length?: number;
  runtime_binding?: 'python_executor' | 'prompt_instructions' | 'none' | string;
}

export interface SkillHookSpec {
  event: string;
  type: string;
  matcher: string;
  command?: string | null;
  handler?: string | null;
}

export interface SkillListEvent extends ServerEvent {
  event_type: 'skill_list';
  skills: SkillSummary[];
}

export interface SkillPermission {
  name: string;
  description?: string;
  is_sensitive: boolean;
}

export interface SkillTool {
  name: string;
  description?: string;
  input_schema?: Record<string, any>;
}

export interface SkillDetail {
  name: string;
  display_name: string;
  description?: string;
  version: string;
  author?: string;
  tools: SkillTool[];
  permissions: SkillPermission[];
  policy: Record<string, any>;
  hooks: string[];
  certification: 'unverified' | 'bronze' | 'silver' | 'gold';
  side_effect: 'none' | 'network' | 'filesystem' | 'exec';
  is_core: boolean;
  source?: 'builtin' | 'community' | 'third_party' | 'local';
  capability_tier?: 'metadata' | 'schema' | 'executable';
  runtime_status?: 'ready' | 'degraded' | 'unavailable';
  is_external_alias?: boolean;
  alias_target?: string;
  instructions_length?: number;
  runtime_binding?: 'python_executor' | 'prompt_instructions' | 'none' | string;
  instructions?: string;
  hook_specs?: SkillHookSpec[];
  package_examples?: Array<{
    id: string;
    label: string;
    description: string;
    input: Record<string, unknown>;
    expectedFocus: string[];
    objective: string;
    source?: string;
    confidence?: 'high' | 'medium' | 'low' | string;
    confidence_reason?: string;
    confidence_breakdown?: Record<string, number>;
  }>;
}

export interface SkillDetailEvent extends ServerEvent {
  event_type: 'skill_detail';
  skill: SkillDetail;
}

export interface SkillMetricsData {
  skill_name: string;
  total_calls: number;
  success_count: number;
  failure_count: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p99_latency_ms: number;
  success_rate: number;
  last_invoked?: string;
}

export interface SkillMetricsEvent extends ServerEvent {
  event_type: 'skill_metrics';
  metrics: SkillMetricsData;
}

export interface SkillLoadedEvent extends ServerEvent {
  event_type: 'skill_loaded';
  skill_name: string;
  message?: string;
}

export interface SkillUnloadedEvent extends ServerEvent {
  event_type: 'skill_unloaded';
  skill_name: string;
}

export interface SkillErrorEvent extends ServerEvent {
  event_type: 'skill_error';
  skill_name?: string;
  error_code: string;
  message: string;
  suggestions: string[];
}

export interface ConsentRequestedEvent extends ServerEvent {
  event_type: 'consent_requested';
  request_id: string;
  skill_name: string;
  tool_name: string;
  reason: string;
  permissions: string[];
  timeout_seconds: number;
}

export interface ConsentResultEvent extends ServerEvent {
  event_type: 'consent_result';
  request_id: string;
  granted: boolean;
  remembered: boolean;
}

export interface SkillBlockedEvent extends ServerEvent {
  event_type: 'skill_blocked';
  skill_name: string;
  tool_name: string;
  policy_action: string;
  reason: string;
}

export interface DryRunResult {
  valid: boolean;
  schema_errors: string[];
  policy_result: 'allow' | 'allow_with_consent' | 'deny';
  capability_gaps: string[];
  estimated_side_effects: string[];
}

export interface DryRunResultEvent extends ServerEvent {
  event_type: 'dry_run_result';
  skill_name: string;
  result: DryRunResult;
}

export type AnyServerEvent =
  | PlanCreatedEvent
  | PlanUpdatedEvent
  | NodeStatusEvent
  | StreamingOutputEvent
  | CheckpointCreatedEvent
  | RestoreCheckpointResultEvent
  | ExecutionStatusEvent
  | RetryStatusEvent
  | RetrySuccessEvent
  | RetryFailedEvent
  | WorkflowListEvent
  | ConflictEvent
  | LogLevelEvent
  | SpanUpdateEvent
  | KnowledgeLibraryListEvent
  | KnowledgeLibraryCreatedEvent
  | KnowledgeLibraryUpdatedEvent
  | KnowledgeLibraryDeletedEvent
  | KnowledgeSearchResultsEvent
  | KnowledgeIngestProgressEvent
  | KnowledgeIngestCompleteEvent
  | KnowledgeErrorEvent
  | DocumentListEvent
  | DocumentDetailEvent
  | DocumentDeletedEvent
  | DocumentStatusChangedEvent
  | ChunkListEvent
  | ChunkPreviewEvent
  // SimpleSkill events
  | SkillListEvent
  | SkillDetailEvent
  | SkillMetricsEvent
  | SkillLoadedEvent
  | SkillUnloadedEvent
  | SkillErrorEvent
  | ConsentRequestedEvent
  | ConsentResultEvent
  | SkillBlockedEvent
  | DryRunResultEvent;

// Command types (client -> server)
export type CommandType =
  | 'start_execution'
  | 'pause'
  | 'resume'
  | 'abort'
  | 'retry_node'
  | 'patch_plan'
  | 'restore_checkpoint'
  | 'set_log_level'
  // Knowledge Base commands
  | 'list_knowledge_libraries'
  | 'create_knowledge_library'
  | 'update_knowledge_library'
  | 'delete_knowledge_library'
  | 'search_knowledge'
  | 'ingest_knowledge_files'
  | 'rebuild_knowledge_index'
  | 'cancel_ingest'
  // Document management commands
  | 'list_documents'
  | 'get_document'
  | 'delete_document'
  | 'disable_document'
  | 'enable_document'
  | 'list_chunks'
  | 'preview_chunks'
  // SimpleSkill commands
  | 'list_skills'
  | 'get_skill_detail'
  | 'get_skill_metrics'
  | 'load_skill'
  | 'unload_skill'
  | 'dry_run_skill'
  | 'consent_response';

export interface ClientCommand {
  command_type: CommandType;
  command_id: string;
  session_id: string;
}

export interface StartExecutionCommand extends ClientCommand {
  command_type: 'start_execution';
  plan_id: string;
  inputs: Record<string, any>;
}

export interface PauseCommand extends ClientCommand {
  command_type: 'pause';
  execution_id: string;
}

export interface ResumeCommand extends ClientCommand {
  command_type: 'resume';
  execution_id: string;
}

export interface AbortCommand extends ClientCommand {
  command_type: 'abort';
  execution_id: string;
}

export interface RetryNodeCommand extends ClientCommand {
  command_type: 'retry_node';
  execution_id: string;
  node_id: string;
  new_inputs: Record<string, any>;
}

export interface PlanPatch {
  operation: string;
  path: string;
  value?: any;
}

export interface PatchPlanCommand extends ClientCommand {
  command_type: 'patch_plan';
  execution_id: string;
  base_version: number;
  patches: PlanPatch[];
}

export interface RestoreCheckpointCommand extends ClientCommand {
  command_type: 'restore_checkpoint';
  checkpoint_id: string;
  replay_mode: 'deterministic' | 'fresh';
}

export interface SetLogLevelCommand extends ClientCommand {
  command_type: 'set_log_level';
  level: string;
}

// ============================================================================
// Knowledge Base Commands
// ============================================================================

export interface ListKnowledgeLibrariesCommand extends ClientCommand {
  command_type: 'list_knowledge_libraries';
}

export interface CreateKnowledgeLibraryCommand extends ClientCommand {
  command_type: 'create_knowledge_library';
  name: string;
  description: string;
  mode: 'agentic' | 'indexed' | 'auto';
  knowledge_dir: string;
  metadata?: Record<string, any>;
}

export interface DeleteKnowledgeLibraryCommand extends ClientCommand {
  command_type: 'delete_knowledge_library';
  library_id: string;
}

export interface SearchKnowledgeCommand extends ClientCommand {
  command_type: 'search_knowledge';
  query: string;
  library_id?: string;
  mode?: 'agentic' | 'indexed' | 'auto';
  top_k?: number;
}

export interface IngestKnowledgeFilesCommand extends ClientCommand {
  command_type: 'ingest_knowledge_files';
  library_id: string;
  paths: string[];
}

export interface UpdateKnowledgeLibraryCommand extends ClientCommand {
  command_type: 'update_knowledge_library';
  library_id: string;
  updates: Record<string, any>;
}

export interface RebuildKnowledgeIndexCommand extends ClientCommand {
  command_type: 'rebuild_knowledge_index';
  library_id: string;
  incremental?: boolean;
}

export interface CancelIngestCommand extends ClientCommand {
  command_type: 'cancel_ingest';
  library_id: string;
}

// ============================================================================
// Document Management Commands
// ============================================================================

export interface ListDocumentsCommand extends ClientCommand {
  command_type: 'list_documents';
  library_id: string;
}

export interface GetDocumentCommand extends ClientCommand {
  command_type: 'get_document';
  library_id: string;
  doc_id: string;
}

export interface DeleteDocumentCommand extends ClientCommand {
  command_type: 'delete_document';
  library_id: string;
  doc_id: string;
}

export interface DisableDocumentCommand extends ClientCommand {
  command_type: 'disable_document';
  library_id: string;
  doc_id: string;
}

export interface EnableDocumentCommand extends ClientCommand {
  command_type: 'enable_document';
  library_id: string;
  doc_id: string;
}

export interface ListChunksCommand extends ClientCommand {
  command_type: 'list_chunks';
  library_id: string;
  doc_id: string;
}

export interface PreviewChunksCommand extends ClientCommand {
  command_type: 'preview_chunks';
  content: string;
  chunk_size: number;
  chunk_overlap: number;
  strategy: string;
}

// =============================================================================
// SimpleSkill Commands
// =============================================================================

export interface ListSkillsCommand extends ClientCommand {
  command_type: 'list_skills';
}

export interface GetSkillDetailCommand extends ClientCommand {
  command_type: 'get_skill_detail';
  skill_name: string;
}

export interface GetSkillMetricsCommand extends ClientCommand {
  command_type: 'get_skill_metrics';
  skill_name: string;
}

export interface LoadSkillCommand extends ClientCommand {
  command_type: 'load_skill';
  source: string;
  install_strategy?: 'copy' | 'symlink';
}

export interface UnloadSkillCommand extends ClientCommand {
  command_type: 'unload_skill';
  skill_name: string;
}

export interface DryRunSkillCommand extends ClientCommand {
  command_type: 'dry_run_skill';
  skill_name: string;
  tool_name: string;
  input: Record<string, any>;
  live?: boolean;
  llm_provider?: string;
  llm_model?: string;
}

export interface ConsentResponseCommand extends ClientCommand {
  command_type: 'consent_response';
  request_id: string;
  granted: boolean;
  remember: boolean;
}

export type AnyClientCommand =
  | StartExecutionCommand
  | PauseCommand
  | ResumeCommand
  | AbortCommand
  | RetryNodeCommand
  | PatchPlanCommand
  | RestoreCheckpointCommand
  | SetLogLevelCommand
  | ListKnowledgeLibrariesCommand
  | CreateKnowledgeLibraryCommand
  | UpdateKnowledgeLibraryCommand
  | DeleteKnowledgeLibraryCommand
  | SearchKnowledgeCommand
  | IngestKnowledgeFilesCommand
  | RebuildKnowledgeIndexCommand
  | CancelIngestCommand
  | ListDocumentsCommand
  | GetDocumentCommand
  | DeleteDocumentCommand
  | DisableDocumentCommand
  | EnableDocumentCommand
  | ListChunksCommand
  | PreviewChunksCommand
  // SimpleSkill commands
  | ListSkillsCommand
  | GetSkillDetailCommand
  | GetSkillMetricsCommand
  | LoadSkillCommand
  | UnloadSkillCommand
  | DryRunSkillCommand
  | ConsentResponseCommand;
