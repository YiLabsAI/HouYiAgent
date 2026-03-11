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

export type ConversationStatus = 'active' | 'archived';

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
}

export interface ContextUsage {
  model: string;
  max_context_tokens: number;
  used_tokens: number;
  reserved_output_tokens: number;
  available_tokens: number;
  block_breakdown: Record<string, number>;
  timestamp: number;
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
}

export interface SSEContextUsage {
  message_id: string;
  usage: ContextUsage;
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
  model?: string;
  temperature?: number;
  max_tokens?: number;
  enable_reasoning?: boolean;
  enable_tool_calls?: boolean;
  tool_call_strategy?: 'conservative' | 'balanced' | 'aggressive';
  enable_web_search?: boolean;
  enable_skills?: string[];
  max_tool_iterations?: number;
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
