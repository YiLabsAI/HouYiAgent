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
  role: MessageRole;
  content: string;
  reasoning_content?: string | null;
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
  enable_web_search?: boolean;
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
