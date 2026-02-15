import React from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import type { ExecutionIR } from '@/types/ir';
import { diffExecutions, type NodeExecutionChangeSet } from '@/utils/diff';
import { GitCompareArrows } from 'lucide-react';

interface ComparePanelProps {
  /** Pre-selected checkpoint IDs from the fused Checkpoints tab.
   *  When provided with exactly 2 entries, auto-sets before/after. */
  preSelectedCheckpoints?: Array<{ checkpointId: string; executionId: string }>;
}

export const ComparePanel: React.FC<ComparePanelProps> = ({ preSelectedCheckpoints }) => {
  const { checkpoints, currentExecution, liveExecution, checkpointExecution, viewMode } = useConsoleStore();

  const buildChangeBadges = (changes: NodeExecutionChangeSet) => {
    const badges: Array<{ label: string; className: string }> = [];

    if (changes.added) {
      badges.push({ label: 'Added', className: 'bg-green-700/60 text-green-100' });
    }
    if (changes.removed) {
      badges.push({ label: 'Removed', className: 'bg-red-700/60 text-red-100' });
    }
    if (changes.status) {
      badges.push({
        label: `Status ${changes.status.before} → ${changes.status.after}`,
        className: 'bg-blue-700/60 text-blue-100',
      });
    }
    if (changes.error) {
      badges.push({ label: 'Error changed', className: 'bg-red-700/50 text-red-100' });
    }
    if (changes.started_at) {
      badges.push({ label: 'Started time', className: 'bg-gray-700/60 text-gray-200' });
    }
    if (changes.completed_at) {
      badges.push({ label: 'Completed time', className: 'bg-gray-700/60 text-gray-200' });
    }
    if (changes.inputs) {
      badges.push({ label: 'Inputs', className: 'bg-gray-700/60 text-gray-200' });
    }
    if (changes.outputs) {
      badges.push({ label: 'Outputs', className: 'bg-gray-700/60 text-gray-200' });
    }
    if (changes.streaming_output) {
      badges.push({ label: 'Streaming', className: 'bg-gray-700/60 text-gray-200' });
    }
    if (changes.metadata) {
      badges.push({ label: 'Metadata', className: 'bg-gray-700/60 text-gray-200' });
    }

    return badges;
  };

  const viewExecution = React.useMemo(() => {
    return viewMode === 'checkpoint' ? checkpointExecution : liveExecution || currentExecution;
  }, [viewMode, checkpointExecution, liveExecution, currentExecution]);

  const executionIds = React.useMemo(() => {
    const ids = new Set<string>();
    checkpoints.forEach((cp) => {
      if (cp.execution_id) ids.add(cp.execution_id);
    });
    return Array.from(ids).sort((a, b) => a.localeCompare(b));
  }, [checkpoints]);

  const latestExecutionId = React.useMemo(() => {
    if (checkpoints.length === 0) return '';
    const sorted = [...checkpoints].sort((a, b) => {
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
    return sorted[0]?.execution_id || '';
  }, [checkpoints]);

  const [scopeExecutionId, setScopeExecutionId] = React.useState<string>('');

  const effectiveScopeExecutionId = React.useMemo(() => {
    if (scopeExecutionId) return scopeExecutionId;
    if (executionIds.length > 1) return 'all';
    if (viewExecution?.execution_id) return viewExecution.execution_id;
    return executionIds[0] || latestExecutionId || 'all';
  }, [scopeExecutionId, executionIds, viewExecution?.execution_id, latestExecutionId]);

  const executionScopedCheckpoints = React.useMemo(() => {
    const scoped = effectiveScopeExecutionId === 'all'
      ? checkpoints
      : checkpoints.filter((cp) => cp.execution_id === effectiveScopeExecutionId);
    return [...scoped].sort((a, b) => {
      const aSeq = typeof a.sequence_number === 'number' ? a.sequence_number : 0;
      const bSeq = typeof b.sequence_number === 'number' ? b.sequence_number : 0;
      return aSeq - bSeq;
    });
  }, [checkpoints, effectiveScopeExecutionId]);

  const [beforeIndex, setBeforeIndex] = React.useState(0);
  const [afterIndex, setAfterIndex] = React.useState(
    Math.min(1, executionScopedCheckpoints.length - 1),
  );

  React.useEffect(() => {
    // When pre-selected checkpoints are provided (from fusion), map to indices.
    if (preSelectedCheckpoints && preSelectedCheckpoints.length >= 2) {
      const findIndex = (cpId: string, execId: string) =>
        executionScopedCheckpoints.findIndex(
          (cp) => cp.checkpoint_id === cpId && cp.execution_id === execId,
        );
      const idx0 = findIndex(preSelectedCheckpoints[0].checkpointId, preSelectedCheckpoints[0].executionId);
      const idx1 = findIndex(preSelectedCheckpoints[1].checkpointId, preSelectedCheckpoints[1].executionId);
      if (idx0 >= 0 && idx1 >= 0) {
        // Earlier checkpoint as before, later as after.
        setBeforeIndex(Math.min(idx0, idx1));
        setAfterIndex(Math.max(idx0, idx1));
        return;
      }
    }
    // Default: latest two checkpoints.
    const lastIndex = Math.max(executionScopedCheckpoints.length - 1, 0);
    setAfterIndex(lastIndex);
    setBeforeIndex(Math.max(lastIndex - 1, 0));
  }, [executionScopedCheckpoints.length, effectiveScopeExecutionId, preSelectedCheckpoints, executionScopedCheckpoints]);
  const beforeCheckpoint = executionScopedCheckpoints[beforeIndex];
  const afterCheckpoint = executionScopedCheckpoints[afterIndex];
  const beforeExec = beforeCheckpoint?.execution_snapshot as ExecutionIR | undefined;
  const afterExec = afterCheckpoint?.execution_snapshot as ExecutionIR | undefined;

  const executionDiff = React.useMemo(() => {
    if (!beforeExec || !afterExec) return null;
    return diffExecutions(beforeExec, afterExec);
  }, [beforeExec, afterExec]);

  const llmLogSummary = React.useMemo(() => {
    const beforeLogs = Array.isArray(beforeCheckpoint?.llm_call_logs)
      ? beforeCheckpoint?.llm_call_logs
      : [];
    const afterLogs = Array.isArray(afterCheckpoint?.llm_call_logs)
      ? afterCheckpoint?.llm_call_logs
      : [];
    return {
      beforeCount: beforeLogs.length,
      afterCount: afterLogs.length,
      hasLogs: beforeLogs.length + afterLogs.length > 0,
    };
  }, [beforeCheckpoint, afterCheckpoint]);

  const llmDiffs = React.useMemo(() => {
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
    const beforeLogs = Array.isArray(beforeCheckpoint?.llm_call_logs)
      ? beforeCheckpoint?.llm_call_logs
      : [];
    const afterLogs = Array.isArray(afterCheckpoint?.llm_call_logs)
      ? afterCheckpoint?.llm_call_logs
      : [];

    if (beforeLogs.length === 0 && afterLogs.length === 0) {
      return [] as Array<{ nodeId: string; changes: string[] }>;
    }

    const toMap = (logs: any[]) => {
      const map: Record<string, any> = {};
      logs.forEach((log) => {
        const nodeId = log?.node_id || log?.nodeId;
        if (nodeId) {
          map[nodeId] = log;
        }
      });
      return map;
    };

    const beforeMap = toMap(beforeLogs);
    const afterMap = toMap(afterLogs);
    const nodeIds = new Set([...Object.keys(beforeMap), ...Object.keys(afterMap)]);

    const diffs: Array<{ nodeId: string; changes: string[] }> = [];

    nodeIds.forEach((nodeId) => {
      const beforeLog = beforeMap[nodeId];
      const afterLog = afterMap[nodeId];
      const changes: string[] = [];

      if (!beforeLog && afterLog) {
        changes.push('Log added');
      } else if (beforeLog && !afterLog) {
        changes.push('Log removed');
      } else if (beforeLog && afterLog) {
        if (hasChanged(beforeLog.model, afterLog.model)) {
          changes.push(`Model ${beforeLog.model ?? '∅'} → ${afterLog.model ?? '∅'}`);
        }
        if (hasChanged(beforeLog.prompt, afterLog.prompt)) {
          changes.push('Prompt changed');
        }
        if (hasChanged(beforeLog.response, afterLog.response)) {
          changes.push('Response changed');
        }

        const beforeMeta = beforeLog.metadata || {};
        const afterMeta = afterLog.metadata || {};
        if (hasChanged(beforeMeta.max_tokens, afterMeta.max_tokens)) {
          changes.push(`max_tokens ${beforeMeta.max_tokens ?? '∅'} → ${afterMeta.max_tokens ?? '∅'}`);
        }
        if (hasChanged(beforeMeta.enable_reasoning, afterMeta.enable_reasoning)) {
          changes.push(`reasoning ${beforeMeta.enable_reasoning ?? 'off'} → ${afterMeta.enable_reasoning ?? 'off'}`);
        }
        if (hasChanged(beforeMeta.thinking_budget, afterMeta.thinking_budget)) {
          changes.push(`thinking_budget ${beforeMeta.thinking_budget ?? '∅'} → ${afterMeta.thinking_budget ?? '∅'}`);
        }

        const metaKeysChanged = diffKeys(beforeMeta, afterMeta);
        if (metaKeysChanged.length > 0) {
          changes.push(`metadata ${metaKeysChanged.join(', ')}`);
        }
      }

      if (changes.length > 0) {
        diffs.push({ nodeId, changes });
      }
    });

    return diffs;
  }, [beforeCheckpoint, afterCheckpoint]);

  if (executionScopedCheckpoints.length < 2) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <GitCompareArrows size={32} className="mx-auto mb-2 opacity-50" />
          <div className="text-sm">Need at least 2 checkpoints to compare</div>
          <div className="text-xs mt-1">Execute workflow to create checkpoints</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">
          Compare Checkpoints
        </h3>
      </div>

      {executionIds.length > 1 && (
        <div className="mb-3 flex items-center justify-between text-xs text-gray-400">
          <span>Scope</span>
          <select
            value={effectiveScopeExecutionId}
            onChange={(e) => setScopeExecutionId(e.target.value)}
            className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-200"
          >
            <option value="all">All executions</option>
            {executionIds.map((execId) => (
              <option key={execId} value={execId}>
                {execId}
              </option>
            ))}
          </select>
        </div>
      )}

      {llmLogSummary.hasLogs && (
        <div className="mb-4 rounded border border-gray-700/60 bg-gray-900/40 p-3">
          <div className="text-xs font-semibold text-gray-300 mb-2">LLM Config Changes</div>
          {llmDiffs.length === 0 ? (
            <div className="text-xs text-gray-500">No LLM changes detected</div>
          ) : (
            <div className="space-y-2">
              {llmDiffs.map((diff) => (
                <div key={diff.nodeId} className="flex flex-col gap-1">
                  <div className="text-xs text-gray-200 font-medium">{diff.nodeId}</div>
                  <div className="flex flex-wrap gap-1">
                    {diff.changes.map((label) => (
                      <span
                        key={`${diff.nodeId}-${label}`}
                        className="px-2 py-0.5 rounded text-[10px] bg-gray-700/60 text-gray-200"
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {executionDiff && (
        <div className={`mb-4 grid gap-3 ${llmLogSummary.hasLogs ? 'grid-cols-3' : 'grid-cols-2'}`}>
          <div className="bg-gray-900 p-2 rounded text-xs text-gray-300">
            <div className="text-gray-400 mb-1">Execution Changes</div>
            <div className="space-y-1">
              <div>
                Status: {executionDiff.status ? `${executionDiff.status.before} → ${executionDiff.status.after}` : 'unchanged'}
              </div>
              <div>
                Error: {executionDiff.error ? 'changed' : 'unchanged'}
              </div>
              <div>
                Context keys: {executionDiff.contextKeysChanged.length}
              </div>
              <div>
                Metadata keys: {executionDiff.metadataKeysChanged.length}
              </div>
            </div>
          </div>
          <div className="bg-gray-900 p-2 rounded text-xs text-gray-300">
            <div className="text-gray-400 mb-1">Node Changes</div>
            <div className="space-y-1">
              <div>Changed: {executionDiff.nodeChanges.length}</div>
              <div>Added: {executionDiff.addedNodes.length}</div>
              <div>Removed: {executionDiff.removedNodes.length}</div>
              <div>Context keys: {executionDiff.contextKeysChanged.length}</div>
            </div>
          </div>
          {llmLogSummary.hasLogs && (
            <div className="bg-gray-900 p-2 rounded text-xs text-gray-300">
              <div className="text-gray-400 mb-1">LLM Logs</div>
              <div className="space-y-1">
                <div>Before: {llmLogSummary.beforeCount}</div>
                <div>After: {llmLogSummary.afterCount}</div>
                <div>Changed nodes: {llmDiffs.length}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {executionDiff && (
        <div className="mb-4 rounded border border-gray-700/60 bg-gray-900/40 p-3">
          <div className="text-xs font-semibold text-gray-300 mb-2">Diff Details</div>
          <div className="space-y-3">
            <div className="text-xs text-gray-300">
              <span className="text-gray-400">Context keys changed:</span>{' '}
              {executionDiff.contextKeysChanged.length > 0
                ? executionDiff.contextKeysChanged.join(', ')
                : 'None'}
            </div>
            <div className="text-xs text-gray-300">
              <span className="text-gray-400">Metadata keys changed:</span>{' '}
              {executionDiff.metadataKeysChanged.length > 0
                ? executionDiff.metadataKeysChanged.join(', ')
                : 'None'}
            </div>
            <div className="space-y-2">
              <div className="text-xs text-gray-400">Node-level changes</div>
              {executionDiff.nodeChanges.length === 0 ? (
                <div className="text-xs text-gray-500">No node changes detected</div>
              ) : (
                executionDiff.nodeChanges.map((change) => {
                  const badges = buildChangeBadges(change.changes);
                  return (
                    <div key={change.node_id} className="flex flex-col gap-1">
                      <div className="text-xs text-gray-200 font-medium">{change.node_id}</div>
                      <div className="flex flex-wrap gap-1">
                        {badges.map((badge) => (
                          <span
                            key={`${change.node_id}-${badge.label}`}
                            className={`px-2 py-0.5 rounded text-[10px] ${badge.className}`}
                          >
                            {badge.label}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 h-[calc(100%-2rem)]">
        {/* Before */}
        <div className="border border-gray-700 rounded p-3 overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-300">Before</h4>
            <select
              value={beforeIndex}
              onChange={(e) => setBeforeIndex(Number(e.target.value))}
              className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-200"
            >
              {executionScopedCheckpoints.map((cp, idx) => (
                <option key={`${cp.execution_id}:${cp.checkpoint_id}`} value={idx}>
                  CP #{cp.sequence_number}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-3">
            <div className="bg-gray-900 p-2 rounded">
              <div className="text-xs font-semibold text-gray-400 mb-1">Checkpoint Info</div>
              <div className="text-xs space-y-1">
                <div className="text-gray-300">ID: {beforeCheckpoint.checkpoint_id}</div>
                <div className="text-gray-300">
                  Created: {new Date(beforeCheckpoint.created_at).toLocaleString()}
                </div>
                <div className="text-gray-300">Trigger: {beforeCheckpoint.trigger}</div>
                {beforeExec?.started_at && (
                  <div className="text-gray-300">
                    Started: {new Date(beforeExec.started_at).toLocaleTimeString()}
                  </div>
                )}
                {beforeExec?.completed_at && (
                  <div className="text-gray-300">
                    Completed: {new Date(beforeExec.completed_at).toLocaleTimeString()}
                  </div>
                )}
              </div>
            </div>

            {beforeExec && (
              <>
                <div className="bg-gray-900 p-2 rounded">
                  <div className="text-xs font-semibold text-gray-400 mb-1">Execution Status</div>
                  <div className="text-xs">
                    <div className="text-gray-300">Status: {beforeExec.status}</div>
                    <div className="text-gray-300">
                      Nodes: {Object.keys(beforeExec.node_executions || {}).length}
                    </div>
                  </div>
                </div>

                <div className="bg-gray-900 p-2 rounded">
                  <div className="text-xs font-semibold text-gray-400 mb-1">Node Executions</div>
                  <div className="space-y-1">
                    {(Object.entries(beforeExec.node_executions || {}) as Array<[string, any]>).map(([nodeId, nodeExec]) => (
                      <div key={nodeId} className="text-xs text-gray-300">
                        <span className={
                          nodeExec.status === 'completed' ? 'text-green-500' :
                          nodeExec.status === 'running' ? 'text-blue-400' :
                          nodeExec.status === 'failed' ? 'text-red-500' :
                          'text-gray-500'
                        }>●</span> {nodeId}: {nodeExec.status}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* After */}
        <div className="border border-gray-700 rounded p-3 overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-300">After</h4>
            <select
              value={afterIndex}
              onChange={(e) => setAfterIndex(Number(e.target.value))}
              className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-200"
            >
              {executionScopedCheckpoints.map((cp, idx) => (
                <option key={`${cp.execution_id}:${cp.checkpoint_id}`} value={idx}>
                  CP #{cp.sequence_number}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-3">
            <div className="bg-gray-900 p-2 rounded">
              <div className="text-xs font-semibold text-gray-400 mb-1">Checkpoint Info</div>
              <div className="text-xs space-y-1">
                <div className="text-gray-300">ID: {afterCheckpoint.checkpoint_id}</div>
                <div className="text-gray-300">
                  Created: {new Date(afterCheckpoint.created_at).toLocaleString()}
                </div>
                <div className="text-gray-300">Trigger: {afterCheckpoint.trigger}</div>
                {afterExec?.started_at && (
                  <div className="text-gray-300">
                    Started: {new Date(afterExec.started_at).toLocaleTimeString()}
                  </div>
                )}
                {afterExec?.completed_at && (
                  <div className="text-gray-300">
                    Completed: {new Date(afterExec.completed_at).toLocaleTimeString()}
                  </div>
                )}
              </div>
            </div>

            {afterExec && (
              <>
                <div className="bg-gray-900 p-2 rounded">
                  <div className="text-xs font-semibold text-gray-400 mb-1">Execution Status</div>
                  <div className="text-xs">
                    <div className="text-gray-300">
                      Status: {afterExec.status}
                      {beforeExec && afterExec.status !== beforeExec.status && (
                        <span className="ml-2 text-yellow-400">← Changed</span>
                      )}
                    </div>
                    <div className="text-gray-300">
                      Nodes: {Object.keys(afterExec.node_executions || {}).length}
                      {beforeExec && Object.keys(afterExec.node_executions || {}).length !== Object.keys(beforeExec.node_executions || {}).length && (
                        <span className="ml-2 text-yellow-400">
                          ({Object.keys(afterExec.node_executions || {}).length > Object.keys(beforeExec.node_executions || {}).length ? '+' : ''}
                          {Object.keys(afterExec.node_executions || {}).length - Object.keys(beforeExec.node_executions || {}).length})
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="bg-gray-900 p-2 rounded">
                  <div className="text-xs font-semibold text-gray-400 mb-1">Node Executions</div>
                  <div className="space-y-1">
                    {(Object.entries(afterExec.node_executions || {}) as Array<[string, any]>).map(([nodeId, nodeExec]) => {
                      const beforeNodeExec = beforeExec?.node_executions?.[nodeId];
                      const statusChanged = beforeNodeExec && beforeNodeExec.status !== nodeExec.status;

                      return (
                        <div key={nodeId} className="text-xs text-gray-300">
                          <span className={
                            nodeExec.status === 'completed' ? 'text-green-500' :
                            nodeExec.status === 'running' ? 'text-blue-400' :
                            nodeExec.status === 'failed' ? 'text-red-500' :
                            'text-gray-500'
                          }>●</span> {nodeId}: {nodeExec.status}
                          {statusChanged && <span className="ml-2 text-yellow-400">← Changed</span>}
                          {!beforeNodeExec && <span className="ml-2 text-green-400">← New</span>}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
