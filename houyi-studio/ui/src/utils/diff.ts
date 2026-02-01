import { ExecutionIR, NodeExecutionIR, NodeStatus, ExecutionStatus } from '@/types/ir';

export interface KeyChange<T> {
  before: T | null;
  after: T | null;
}

export interface NodeExecutionChangeSet {
  added?: boolean;
  removed?: boolean;
  status?: KeyChange<NodeStatus>;
  error?: KeyChange<string | null>;
  started_at?: KeyChange<string | null>;
  completed_at?: KeyChange<string | null>;
  inputs?: boolean;
  outputs?: boolean;
  streaming_output?: boolean;
  metadata?: boolean;
}

export interface NodeExecutionDiff {
  node_id: string;
  before: NodeExecutionIR | null;
  after: NodeExecutionIR | null;
  changes: NodeExecutionChangeSet;
}

export interface ExecutionDiff {
  status?: KeyChange<ExecutionStatus>;
  error?: KeyChange<string | null>;
  nodeChanges: NodeExecutionDiff[];
  addedNodes: string[];
  removedNodes: string[];
  contextKeysChanged: string[];
  metadataKeysChanged: string[];
}

const normalizeValue = (value: any): any => {
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeValue(entry));
  }
  if (value && typeof value === 'object') {
    return Object.keys(value)
      .sort()
      .reduce<Record<string, any>>((acc, key) => {
        acc[key] = normalizeValue(value[key]);
        return acc;
      }, {});
  }
  return value === undefined ? null : value;
};

const stableStringify = (value: any): string => JSON.stringify(normalizeValue(value));

const hasChanged = (before: any, after: any): boolean => {
  return stableStringify(before) !== stableStringify(after);
};

const diffKeys = (before: Record<string, any>, after: Record<string, any>): string[] => {
  const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
  return Array.from(keys).filter((key) => hasChanged(before[key], after[key]));
};

export const diffExecutions = (before: ExecutionIR, after: ExecutionIR): ExecutionDiff => {
  const status = before.status !== after.status
    ? { before: before.status, after: after.status }
    : undefined;
  const error = before.error !== after.error
    ? { before: before.error, after: after.error }
    : undefined;

  const beforeNodes = before.node_executions || {};
  const afterNodes = after.node_executions || {};
  const nodeIds = new Set([...Object.keys(beforeNodes), ...Object.keys(afterNodes)]);

  const nodeChanges: NodeExecutionDiff[] = [];
  const addedNodes: string[] = [];
  const removedNodes: string[] = [];

  nodeIds.forEach((nodeId) => {
    const beforeNode = beforeNodes[nodeId] ?? null;
    const afterNode = afterNodes[nodeId] ?? null;
    const changes: NodeExecutionChangeSet = {};

    if (!beforeNode && afterNode) {
      changes.added = true;
      addedNodes.push(nodeId);
      nodeChanges.push({ node_id: nodeId, before: null, after: afterNode, changes });
      return;
    }

    if (beforeNode && !afterNode) {
      changes.removed = true;
      removedNodes.push(nodeId);
      nodeChanges.push({ node_id: nodeId, before: beforeNode, after: null, changes });
      return;
    }

    if (!beforeNode || !afterNode) {
      return;
    }

    if (beforeNode.status !== afterNode.status) {
      changes.status = { before: beforeNode.status, after: afterNode.status };
    }
    if (beforeNode.error !== afterNode.error) {
      changes.error = { before: beforeNode.error, after: afterNode.error };
    }
    if (beforeNode.started_at !== afterNode.started_at) {
      changes.started_at = { before: beforeNode.started_at, after: afterNode.started_at };
    }
    if (beforeNode.completed_at !== afterNode.completed_at) {
      changes.completed_at = { before: beforeNode.completed_at, after: afterNode.completed_at };
    }
    if (hasChanged(beforeNode.inputs, afterNode.inputs)) {
      changes.inputs = true;
    }
    if (hasChanged(beforeNode.outputs, afterNode.outputs)) {
      changes.outputs = true;
    }
    if (hasChanged(beforeNode.streaming_output, afterNode.streaming_output)) {
      changes.streaming_output = true;
    }
    if (hasChanged(beforeNode.metadata, afterNode.metadata)) {
      changes.metadata = true;
    }

    if (Object.keys(changes).length > 0) {
      nodeChanges.push({ node_id: nodeId, before: beforeNode, after: afterNode, changes });
    }
  });

  return {
    status,
    error,
    nodeChanges,
    addedNodes,
    removedNodes,
    contextKeysChanged: diffKeys(before.context || {}, after.context || {}),
    metadataKeysChanged: diffKeys(before.metadata || {}, after.metadata || {}),
  };
};
