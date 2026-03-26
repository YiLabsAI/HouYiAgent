/**
 * Chat type definitions for Chatbox.
 *
 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

export interface Attachment {
  filename: string;
  mime_type: string;
  data: string;  // base64 data URI
  size: number;
}

export interface ChatMessage {
  message_id: string;
  ui_render_id?: string;
  role: MessageRole;
  content: string;
  reasoning_content?: string | null;
  tool_calls?: Array<Record<string, any>>;
  tool_call_id?: string | null;
  name?: string | null;
  attachments?: Attachment[];
  bookmarked?: boolean;
  metadata: Record<string, any>;
  created_at: number;
}

export type PinStatus = 'active' | 'archived' | 'removed' | 'superseded';

export interface PinnedContextRecord {
  pin_id: string;
  conversation_id: string;
  source_message_id: string;
  title: string;
  content: string;
  role: 'context' | 'user' | 'assistant' | 'system' | 'tool';
  scope: 'conversation';
  status: PinStatus;
  priority: number;
  token_count: number;
  created_at: number;
  updated_at: number;
  metadata: Record<string, any>;
}

export type ConversationStatus = 'active' | 'archived';

export type ConversationContextHealth = 'healthy' | 'elevated' | 'near_compaction' | 'compacted_recently';

export interface ConversationContextState {
  conversation_id: string;
  used_units: number;
  max_units: number;
  state: ConversationContextHealth;
  last_compacted_at?: number | null;
  last_compaction_delta?: number | null;
  last_compacted_message_count?: number | null;
  updated_at: number;
}

export interface ActiveStreamingState {
  conversation_id: string;
  message_id: string;
  request_id: string;
  status: 'streaming' | 'finishing';
  started_at: number;
  updated_at: number;
}

export interface Conversation {
  conversation_id: string;
  title: string;
  status: ConversationStatus;
  messages: ChatMessage[];
  model: string;
  system_instructions: string;
  temperature: number | null;
  max_tokens: number | null;
  top_p: number | null;
  stream: boolean | null;
  bookmarked: boolean;
  conversation_context_state?: ConversationContextState | null;
  active_streaming_state?: ActiveStreamingState | null;
  metadata: Record<string, any>;
  created_at: number;
  updated_at: number;
  schema_version: number;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  status: ConversationStatus;
  message_count: number;
  visible_message_count?: number;
  model: string;
  created_at: number;
  updated_at: number;
  last_message_at: number | null;
  bookmarked: boolean;
  active_streaming_state?: ActiveStreamingState | null;
}

export interface ContextUsage {
  model: string;
  max_context_tokens: number;
  used_tokens: number;
  reserved_output_tokens: number;
  available_tokens: number;
  block_breakdown: Record<string, number>;
  dropped_blocks?: string[];
  drop_reasons?: Record<string, string>;
  dropped_block_details?: Array<{
    candidate_id: string;
    block_type: string;
    source: string;
    token_count: number;
    message_count?: number | null;
    pinned?: boolean;
  }>;
  planned_prompt_tokens?: number;
  available_input_tokens?: number;
  timestamp: number;
}

export interface CompactionMetrics {
  messages_compacted: number;
  tokens_before: number;
  tokens_after: number;
  compression_ratio?: number;
  entity_retention_ratio?: number;
  pin_violation_count?: number;
}

export interface TokenUsage {
  input_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  reasoning_tokens?: number;
  reasoning_tokens_reported?: boolean;
  answer_tokens?: number;
  answer_tokens_reported?: boolean;
  cached_prompt_tokens?: number;
  cached_prompt_tokens_reported?: boolean;
  cache_hit?: boolean;
  cache_hit_reported?: boolean;
  usage_source?: string;
  usage_confidence?: string;
  first_token_ms?: number;
  decode_tokens_per_second?: number;
  end_to_end_tokens_per_second?: number;
  [key: string]: any;
}

export interface CompactionRecord {
  compaction_id: string;
  trigger: string;
  pressure_level?: string;
  backup_id?: string | null;
  summary: string;
  source_message_ids: string[];
  pinned_message_ids: string[];
  retained_refs: string[];
  metrics: CompactionMetrics;
  restore_status?: string | null;
  created_at: number;
  metadata: Record<string, any>;
}

export interface CompactionBackupRecord {
  backup_id: string;
  conversation_id: string;
  trigger: string;
  created_at: number;
  path: string;
  record_id?: string | null;
  metadata: Record<string, any>;
}

export interface CompactionMessagePreview {
  message_id: string;
  role: string;
  name?: string | null;
  created_at?: number;
  preview: string;
}

export interface CompactionDiff {
  source_message_ids: string[];
  backup_message_count: number | null;
  current_message_count: number;
  backup_visible_message_count: number | null;
  current_visible_message_count: number;
  removed_message_ids: string[];
  added_message_ids: string[];
  source_message_previews?: CompactionMessagePreview[];
  added_message_previews?: CompactionMessagePreview[];
}

export interface CompactionHistoryItem {
  compaction: CompactionRecord;
  backup: CompactionBackupRecord | null;
  diff: CompactionDiff;
}

// SSE event types
export interface SSEMessageDelta {
  message_id: string;
  seq: number;
  content?: string;
  reasoning_content?: string;
}

export interface SSEMessageFinish {
  message_id: string;
  model: string;
  finish_reason: string;
  total_chunks: number;
  content_length: number;
  reasoning_length?: number;
  timestamp: number;
}

export interface SSEMessageError {
  message_id: string;
  error: string;
  error_type: string;
  chunks_sent: number;
  timestamp: number;
  error_code?: string;
  public_message?: string;
  retryable?: boolean;
  status_code?: number | null;
  provider_code?: string | null;
}

export interface SSEContextUsage {
  message_id: string;
  usage: ContextUsage;
}

export interface SSEContextCompacted {
  message_id: string;
  compaction: CompactionRecord;
}

export interface SSEContextStateUpdated {
  message_id: string;
  conversation_id: string;
  conversation_context_state: ConversationContextState;
  source: string;
  reason: string;
}

export interface SSEAgentIteration {
  message_id: string;
  trace_id?: string;
  round_index: number;
}

export interface SSEToolCallStart {
  message_id: string;
  trace_id?: string;
  tool_call_id?: string;
  tool_name?: string;
  requested_tool_name?: string;
  parallel_group_id?: string;
  round_index?: number;
  duration_ms?: number;
  arguments?: Record<string, any>;
}

export interface SSEToolCallResult {
  message_id: string;
  trace_id?: string;
  tool_call_id?: string;
  tool_name?: string;
  requested_tool_name?: string;
  parallel_group_id?: string;
  round_index?: number;
  duration_ms?: number;
  result?: any;
}

export interface SSEToolCallError {
  message_id: string;
  trace_id?: string;
  tool_call_id?: string;
  tool_name?: string;
  requested_tool_name?: string;
  parallel_group_id?: string;
  round_index?: number;
  duration_ms?: number;
  error?: any;
}

export interface SSEMessageComplete {
  message_id: string;
  metadata?: {
    trace_id?: string;
    usage?: Record<string, any>;
    [key: string]: any;
  };
}

export interface TraceSpanEvent {
  name?: string;
  timestamp?: number;
  attributes?: Record<string, unknown>;
}

export interface TraceSpan {
  name?: string;
  span_type?: string;
  status?: string;
  duration_ms?: number;
  start_time_ms?: number;
  attributes?: Record<string, unknown>;
  events?: TraceSpanEvent[];
  children?: TraceSpan[];
}

export interface TraceTotalTokens {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  llm_spans?: number;
  llm_spans_with_usage?: number;
  is_partial?: boolean;
}

export interface TraceRequestContext {
  request_id?: string | null;
  conversation_id?: string | null;
  model?: string | null;
  max_context_tokens?: number | null;
  llm_messages_count?: number | null;
}

export interface TraceContextPlan {
  used_tokens?: number | null;
  planned_prompt_tokens?: number | null;
  reserved_output_tokens?: number | null;
  available_input_tokens?: number | null;
  block_breakdown?: Record<string, number>;
}

export interface TraceCompactionSummary {
  triggered?: boolean;
  trigger?: string | null;
  messages_compacted?: number | null;
  tokens_before?: number | null;
  tokens_after?: number | null;
  saved_tokens?: number | null;
  pin_violation_count?: number | null;
}

export interface TraceContextGovernance {
  dropped_blocks?: string[];
  drop_reasons?: Record<string, string>;
  dropped_block_details?: Array<{
    candidate_id: string;
    block_type: string;
    source: string;
    token_count: number;
    message_count?: number | null;
    pinned?: boolean;
  }>;
  compaction?: TraceCompactionSummary;
}

export interface TracePayload {
  trace_id?: string;
  total_duration_ms?: number;
  total_tokens?: TraceTotalTokens;
  request_context?: TraceRequestContext;
  context_plan?: TraceContextPlan;
  context_governance?: TraceContextGovernance;
  root_span?: TraceSpan;
}

// API request types
export interface CreateConversationRequest {
  title?: string;
  model?: string;
  system_instructions?: string;
  metadata?: Record<string, any>;
}

export interface SendMessageRequest {
  content: string;
  attachments?: Attachment[];
  parent_message_id?: string;
  enable_reasoning?: boolean;
  enable_tool_calls?: boolean;
  tool_call_strategy?: 'conservative' | 'balanced' | 'aggressive';
  enable_web_search?: boolean;
  enable_deep_research?: boolean;
  enable_skills?: string[];
  max_tool_iterations?: number;
  model?: string;
  system_instructions?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stream?: boolean;
}

export interface UpdateConversationRequest {
  title?: string;
  status?: ConversationStatus;
  system_instructions?: string;
  model?: string;
  temperature?: number | null;
  max_tokens?: number | null;
  top_p?: number | null;
  bookmarked?: boolean;
}
