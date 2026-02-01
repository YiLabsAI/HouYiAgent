/**
 * IR (Intermediate Representation) types
 * Shared between frontend and backend
 */

export type NodeType = 'llm' | 'tool' | 'verify' | 'logic' | 'route';

export type NodeStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export type ExecutionStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'aborted';

export interface NodeIR {
  node_id: string;
  node_type: NodeType;
  position: { x: number; y: number };
  config: Record<string, any>;
  inputs: Record<string, any>;
  outputs: Record<string, string>;
  deleted_at: string | null;
  metadata: Record<string, any>;
}

export interface PlanPositionIR {
  x: number;
  y: number;
}

export interface PlanLayoutIR {
  positions: Record<string, PlanPositionIR>;
}

export interface ToolResultIR {
  call_id?: string | null;
  raw: Record<string, any>;
  content: string;
  is_error: boolean;
  metadata: Record<string, any>;
}

export interface ToolOverrideIR {
  from?: string | null;
  to?: string | null;
  allowed: boolean;
  applied: boolean;
}

export interface ToolCallTraceIR {
  tool_name?: string | null;
  requested_tool_name?: string | null;
  tool_call_id?: string | null;
  args: Record<string, any>;
  result: ToolResultIR;
  tool_override?: ToolOverrideIR | null;
}

export interface ToolErrorIR {
  tool_name?: string | null;
  requested_tool_name?: string | null;
  tool_call_id?: string | null;
  error?: Record<string, any> | null;
}

export interface LLMToolCallOutputIR {
  type: 'llm_response';
  content: string;
  tool_calls: ToolCallTraceIR[];
  finish_reason?: string | null;
  tool_finish_reason?: string | null;
  tool_call_rounds: number;
  max_rounds_reached: boolean;
  tool_errors: ToolErrorIR[];
  messages: Array<Record<string, any>>;
  error?: string | null;
}

export interface ToolNodeOutputIR {
  type: 'tool_result';
  output: Record<string, any>;
  is_error: boolean;
  metadata: Record<string, any>;
}

export type NodeOutputsIR = LLMToolCallOutputIR | ToolNodeOutputIR | Record<string, any>;

export interface EdgeIR {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  label?: string;
  metadata: Record<string, any>;
}

export interface PlanIR {
  plan_id: string;
  version: number;
  nodes: NodeIR[];
  edges: EdgeIR[];
  entry_node_id: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
  layout?: PlanLayoutIR;
}

export interface NodeExecutionIR {
  node_id: string;
  status: NodeStatus;
  started_at: string | null;
  completed_at: string | null;
  inputs: Record<string, any>;
  outputs: NodeOutputsIR;
  error: string | null;
  streaming_output: string;
  metadata: Record<string, any>;
}

export interface ExecutionIR {
  execution_id: string;
  plan_id: string;
  status: ExecutionStatus;
  node_executions: Record<string, NodeExecutionIR>;
  context: Record<string, any>;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  metadata: Record<string, any>;
}

export interface CheckpointIR {
  checkpoint_id: string;
  execution_id: string;
  plan_id: string;
  sequence_number: number;
  trigger: string;
  created_at: string;
  execution_snapshot: Record<string, any>;
  llm_call_logs: any[];
  parent_checkpoint_id: string | null;
  delta: Record<string, any> | null;
  metadata: Record<string, any>;
}
