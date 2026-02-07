/**
 * WebSocket message types
 * Matches backend event and command definitions
 */

import type { PlanIR, NodeStatus, ExecutionStatus } from './ir';

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
  | 'span_update';

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
  metadata?: Record<string, any>;
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
  | SpanUpdateEvent;

// Command types (client -> server)
export type CommandType =
  | 'start_execution'
  | 'pause'
  | 'resume'
  | 'abort'
  | 'retry_node'
  | 'patch_plan'
  | 'restore_checkpoint'
  | 'set_log_level';

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

export type AnyClientCommand =
  | StartExecutionCommand
  | PauseCommand
  | ResumeCommand
  | AbortCommand
  | RetryNodeCommand
  | PatchPlanCommand
  | RestoreCheckpointCommand
  | SetLogLevelCommand;
