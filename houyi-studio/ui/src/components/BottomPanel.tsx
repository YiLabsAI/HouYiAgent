import React from 'react';
import { ComparePanel } from './panels/ComparePanel';
import { LogsPanel } from './panels/LogsPanel';
import { useConsoleStore } from '../stores/useConsoleStore';
import type { ExecutionIR } from '@/types/ir';
import { diffExecutions } from '@/utils/diff';

interface BottomPanelProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  activeTab?: 'timeline' | 'checkpoints' | 'context' | 'logs' | 'verification' | 'compare';
  onTabChange?: (
    tab: 'timeline' | 'checkpoints' | 'context' | 'logs' | 'verification' | 'compare',
  ) => void;
}

type TabType = 'timeline' | 'checkpoints' | 'context' | 'logs' | 'verification' | 'compare';

export const BottomPanel: React.FC<BottomPanelProps> = ({
  isCollapsed,
  onToggleCollapse,
  activeTab: controlledActiveTab,
  onTabChange,
}) => {
  const [uncontrolledActiveTab, setUncontrolledActiveTab] = React.useState<TabType>('timeline');
  const activeTab = controlledActiveTab ?? uncontrolledActiveTab;
  const setActiveTab = (tab: TabType) => {
    if (onTabChange) {
      onTabChange(tab);
      return;
    }
    setUncontrolledActiveTab(tab);
  };
  const [expandedCheckpointExecutions, setExpandedCheckpointExecutions] = React.useState<Record<string, boolean>>({});
  const [showRestoreDialog, setShowRestoreDialog] = React.useState(false);
  const [restoreCheckpointId, setRestoreCheckpointId] = React.useState<string | null>(null);
  const [restoreExecutionId, setRestoreExecutionId] = React.useState<string | null>(null);
  const {
    checkpoints,
    selectedCheckpointKey,
    loadCheckpoint,
    exitCheckpointView,
    currentExecution,
    liveExecution,
    checkpointExecution,
    viewMode,
    sendCommand,
    sessionId,
    currentPlan,
    prepareRestoreFromCheckpoint,
    clearCurrentExecutionOutputsForFreshReplay,
  } = useConsoleStore();

  const viewExecution = React.useMemo(() => {
    return viewMode === 'checkpoint' ? checkpointExecution : liveExecution || currentExecution;
  }, [viewMode, checkpointExecution, liveExecution, currentExecution]);

  const effectiveSelectedKey = selectedCheckpointKey;

  const executionScopedCheckpoints = React.useMemo(() => {
    if (!viewExecution?.execution_id) return checkpoints;
    return checkpoints.filter((cp) => cp.execution_id === viewExecution.execution_id);
  }, [checkpoints, viewExecution?.execution_id]);

  const groupedCheckpoints = React.useMemo(() => {
    const byExec = new Map<string, any[]>();
    checkpoints.forEach((cp) => {
      const execId = cp.execution_id || 'unknown';
      const list = byExec.get(execId) ?? [];
      list.push(cp);
      byExec.set(execId, list);
    });

    const result = Array.from(byExec.entries()).map(([execId, list]) => {
      const sorted = [...list].sort((a, b) => {
        const aSeq = typeof a.sequence_number === 'number' ? a.sequence_number : 0;
        const bSeq = typeof b.sequence_number === 'number' ? b.sequence_number : 0;
        if (aSeq !== bSeq) return bSeq - aSeq;
        const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
        return bTime - aTime;
      });
      return { execId, checkpoints: sorted };
    });

    // Sort execution groups by the most recent checkpoint first, but prefer current execution on top.
    const currentExecId = viewExecution?.execution_id;
    result.sort((a, b) => {
      if (currentExecId) {
        if (a.execId === currentExecId && b.execId !== currentExecId) return -1;
        if (b.execId === currentExecId && a.execId !== currentExecId) return 1;
      }
      const aTop = a.checkpoints?.[0];
      const bTop = b.checkpoints?.[0];
      const aSeq = typeof aTop?.sequence_number === 'number' ? aTop.sequence_number : 0;
      const bSeq = typeof bTop?.sequence_number === 'number' ? bTop.sequence_number : 0;
      if (aSeq !== bSeq) return bSeq - aSeq;
      const aTime = aTop?.created_at ? new Date(aTop.created_at).getTime() : 0;
      const bTime = bTop?.created_at ? new Date(bTop.created_at).getTime() : 0;
      return bTime - aTime;
    });

    return result;
  }, [checkpoints, viewExecution?.execution_id]);

  const isCheckpointGroupExpanded = React.useCallback(
    (execId: string) => {
      if (expandedCheckpointExecutions[execId] !== undefined) {
        return Boolean(expandedCheckpointExecutions[execId]);
      }
      const currentExecId = viewExecution?.execution_id;
      if (currentExecId && execId === currentExecId) return true;
      return groupedCheckpoints.length > 0 && groupedCheckpoints[0].execId === execId;
    },
    [expandedCheckpointExecutions, groupedCheckpoints, viewExecution?.execution_id],
  );

  const timelineCheckpoints = React.useMemo(() => {
    if (checkpoints.length === 0) return [] as typeof checkpoints;
    if (viewExecution?.execution_id) {
      const scoped = checkpoints.filter((cp) => cp.execution_id === viewExecution.execution_id);
      if (scoped.length > 1) return scoped;
    }
    return checkpoints;
  }, [checkpoints, viewExecution?.execution_id]);

  const orderedTimelineCheckpoints = React.useMemo(() => {
    return [...timelineCheckpoints].sort((a, b) => {
      const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
      const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
      if (aTime !== bTime) return aTime - bTime;
      const aSeq = typeof a.sequence_number === 'number' ? a.sequence_number : 0;
      const bSeq = typeof b.sequence_number === 'number' ? b.sequence_number : 0;
      return aSeq - bSeq;
    });
  }, [timelineCheckpoints]);

  const timelineSelectedCheckpoint = React.useMemo(() => {
    if (!effectiveSelectedKey) return null;
    return orderedTimelineCheckpoints.find((cp) =>
      cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
      (!effectiveSelectedKey.execution_id || cp.execution_id === effectiveSelectedKey.execution_id),
    ) || null;
  }, [effectiveSelectedKey, orderedTimelineCheckpoints]);

  const timelineSelectedSnapshot = React.useMemo(() => {
    return timelineSelectedCheckpoint?.execution_snapshot as ExecutionIR | undefined;
  }, [timelineSelectedCheckpoint]);

  const previewNodes = React.useMemo(() => {
    if (!timelineSelectedSnapshot?.node_executions) return [] as Array<[string, any]>;
    return Object.entries(timelineSelectedSnapshot.node_executions)
      .filter(([_nodeId, nodeExec]) => {
        const status = (nodeExec as any)?.status;
        return status && status !== 'pending';
      })
      .sort(([_aId, aExec], [_bId, bExec]) => {
        const aTime = (aExec as any)?.completed_at || (aExec as any)?.started_at;
        const bTime = (bExec as any)?.completed_at || (bExec as any)?.started_at;
        const aMs = aTime ? new Date(aTime).getTime() : 0;
        const bMs = bTime ? new Date(bTime).getTime() : 0;
        if (aMs !== bMs) return bMs - aMs;
        return String(_aId).localeCompare(String(_bId));
      })
      .slice(0, 5);
  }, [timelineSelectedSnapshot]);

  const contextDiff = React.useMemo(() => {
    if (viewMode !== 'checkpoint' || !liveExecution || !checkpointExecution) return null;
    return diffExecutions(liveExecution, checkpointExecution);
  }, [viewMode, liveExecution, checkpointExecution]);

  const getCacheHitCount = React.useCallback((nodeExec: any): number => {
    const outputs = nodeExec?.outputs ?? {};
    const toolCalls = outputs?.tool_calls ?? [];
    return toolCalls.filter((call: any) => Boolean(call?.result?.metadata?.cache_hit)).length;
  }, []);

  const getCheckpointNodeChips = React.useCallback((checkpoint: any): string[] => {
    const snapshot = checkpoint?.execution_snapshot as any;
    const nodeExecs = snapshot?.node_executions ?? {};
    if (!nodeExecs || typeof nodeExecs !== 'object') return [];

    const items = Object.entries(nodeExecs)
      .map(([nodeId, nodeExec]: [string, any]) => ({ nodeId, status: nodeExec?.status }))
      .filter((item) => item.status && item.status !== 'pending' && item.status !== 'skipped');

    // Prefer deterministic ordering for stable rendering: completed first, then running, then failed, then others.
    const order: Record<string, number> = { completed: 0, running: 1, failed: 2 };
    items.sort((a, b) => (order[a.status] ?? 99) - (order[b.status] ?? 99) || a.nodeId.localeCompare(b.nodeId));
    return items.map((item) => item.nodeId);
  }, []);

  const handleRestoreCheckpoint = React.useCallback((checkpointId: string, executionId?: string) => {
    setRestoreCheckpointId(checkpointId);
    setRestoreExecutionId(executionId || null);
    setShowRestoreDialog(true);
  }, []);

  const handleCancelRestoreDialog = React.useCallback(() => {
    setShowRestoreDialog(false);
    setRestoreCheckpointId(null);
    setRestoreExecutionId(null);
  }, []);

  const handleConfirmRestoreDialog = React.useCallback(
    (replayMode: 'fresh' | 'deterministic') => {
      if (!restoreCheckpointId) return;

      const checkpoint = (checkpoints || []).find((cp: any) => {
        if (restoreExecutionId) {
          return cp.checkpoint_id === restoreCheckpointId && cp.execution_id === restoreExecutionId;
        }
        return cp.checkpoint_id === restoreCheckpointId;
      });

      if (replayMode === 'fresh') {
        clearCurrentExecutionOutputsForFreshReplay();
      }

      prepareRestoreFromCheckpoint({
        checkpointId: restoreCheckpointId,
        executionId: restoreExecutionId,
        planId: currentPlan?.plan_id,
      });

      sendCommand({
        command_type: 'restore_checkpoint',
        command_id: `cmd_${Date.now()}`,
        session_id: sessionId,
        execution_id: restoreExecutionId || checkpoint?.execution_id || undefined,
        checkpoint_id: restoreCheckpointId,
        replay_mode: replayMode,
      });

      setShowRestoreDialog(false);
      setRestoreCheckpointId(null);
      setRestoreExecutionId(null);
    },
    [
      checkpoints,
      clearCurrentExecutionOutputsForFreshReplay,
      currentPlan?.plan_id,
      prepareRestoreFromCheckpoint,
      restoreCheckpointId,
      restoreExecutionId,
      sendCommand,
      sessionId,
    ],
  );

  const tabs: { id: TabType; label: string }[] = [
    { id: 'timeline', label: 'Timeline' },
    { id: 'checkpoints', label: 'Checkpoints' },
    { id: 'context', label: 'Context' },
    { id: 'logs', label: 'Logs' },
    { id: 'verification', label: 'Verification' },
    { id: 'compare', label: 'Compare' },
  ];

  return (
    <div className={`bg-gray-800 border-t border-gray-700 flex flex-col transition-all duration-300 ease-in-out ${
      isCollapsed ? 'h-8' : 'h-64'
    }`}>
      {isCollapsed ? (
        <div className="h-full flex items-center justify-between px-4">
          <div className="flex gap-2 text-xs text-gray-400">
            <span>Timeline</span>
            <span>•</span>
            <span>Checkpoints</span>
            <span>•</span>
            <span>Context</span>
          </div>
          <button
            onClick={onToggleCollapse}
            className="p-1 hover:bg-gray-700 rounded text-gray-400"
            title="Expand panel"
          >
            ▲
          </button>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between border-b border-gray-700">
            <div className="flex">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 text-xs font-medium ${
                    activeTab === tab.id
                      ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 pr-2">
              {viewMode === 'checkpoint' && (
                <button
                  onClick={exitCheckpointView}
                  className="px-2 py-1 text-[11px] rounded border border-gray-600 text-gray-200 hover:bg-gray-700"
                  title="Back to live execution"
                  type="button"
                >
                  Live
                </button>
              )}
              <button
                onClick={onToggleCollapse}
                className="p-2 hover:bg-gray-700 rounded text-gray-400"
                title="Collapse panel"
                type="button"
              >
                ▼
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === 'timeline' && (
              <div className="space-y-4">
                {checkpoints.length === 0 ? (
                  <div className="text-center text-gray-400 py-8">
                    No timeline available - start execution to create checkpoints
                  </div>
                ) : (
                  <>
                    <div className="relative">
                      <input
                        type="range"
                        min="0"
                        max={Math.max(executionScopedCheckpoints.length - 1, 0)}
                        value={Math.max(
                          executionScopedCheckpoints.findIndex(
                            (cp) =>
                              effectiveSelectedKey
                                ? cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
                                  (!effectiveSelectedKey.execution_id ||
                                    cp.execution_id === effectiveSelectedKey.execution_id)
                                : false,
                          ),
                          0,
                        )}
                        onChange={(e) => {
                          const index = parseInt(e.target.value);
                          if (executionScopedCheckpoints[index]) {
                            loadCheckpoint(
                              executionScopedCheckpoints[index].checkpoint_id,
                              executionScopedCheckpoints[index].execution_id,
                            );
                          }
                        }}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                      />
                      <div className="flex justify-between text-xs text-gray-400 mt-2">
                        <span>Start</span>
                        <span className="text-blue-400 font-medium">
                          {effectiveSelectedKey
                            ? `Checkpoint #${executionScopedCheckpoints.find((cp) => (
                                cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
                                (!effectiveSelectedKey.execution_id || cp.execution_id === effectiveSelectedKey.execution_id)
                              ))?.sequence_number || '?'}`
                            : 'No checkpoint selected'
                          }
                        </span>
                        <span>Latest</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      <span>Total: {executionScopedCheckpoints.length}</span>
                      <span>•</span>
                      <span>
                        {effectiveSelectedKey && executionScopedCheckpoints.find((cp) => (
                          cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
                          (!effectiveSelectedKey.execution_id || cp.execution_id === effectiveSelectedKey.execution_id)
                        ))
                          ? new Date(executionScopedCheckpoints.find((cp) => (
                              cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
                              (!effectiveSelectedKey.execution_id || cp.execution_id === effectiveSelectedKey.execution_id)
                            ))!.created_at).toLocaleString()
                          : 'N/A'
                        }
                      </span>
                    </div>

                    <div className="bg-gray-900/70 border border-gray-700 rounded p-3 text-xs text-gray-300">
                      <div className="text-gray-400 font-semibold mb-2">Snapshot Preview</div>
                      {timelineSelectedCheckpoint && timelineSelectedSnapshot ? (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Status</span>
                            <span className="text-gray-200">{timelineSelectedSnapshot.status}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Nodes</span>
                            <span className="text-gray-200">
                              {Object.keys(timelineSelectedSnapshot.node_executions || {}).length}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Context keys</span>
                            <span className="text-gray-200">
                              {Object.keys(timelineSelectedSnapshot.context || {}).length}
                            </span>
                          </div>
                          <div className="space-y-1">
                            <div className="text-gray-400">Recent nodes</div>
                            {previewNodes.length === 0 ? (
                              <div className="text-gray-500">No node executions</div>
                            ) : (
                              previewNodes.map(([nodeId, nodeExec]) => {
                                const cacheHits = getCacheHitCount(nodeExec);
                                return (
                                  <div key={nodeId} className="flex items-center justify-between">
                                    <span className="text-gray-300 truncate">
                                      {nodeId}
                                      {cacheHits > 0 ? (
                                        <span className="ml-2 text-amber-300">⚡</span>
                                      ) : null}
                                    </span>
                                    <span className="text-gray-400">{nodeExec.status}</span>
                                  </div>
                                );
                              })
                            )}
                          </div>
                        </div>
                      ) : (
                        <div className="text-gray-500">Select a checkpoint to preview</div>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'checkpoints' && (
              <div className="space-y-2">
                {groupedCheckpoints.length === 0 ? (
                  <div className="text-center text-gray-400 py-8">
                    No checkpoints created yet
                  </div>
                ) : (
                  groupedCheckpoints.map(({ execId, checkpoints: execCps }) => {
                    const expanded = isCheckpointGroupExpanded(execId);
                    return (
                      <div key={execId} className="space-y-2">
                        <button
                          type="button"
                          onClick={() => {
                            setExpandedCheckpointExecutions((prev) => ({
                              ...prev,
                              [execId]: !expanded,
                            }));
                          }}
                          className="w-full flex items-center justify-between px-2 py-1 bg-gray-800 border border-gray-700/50 rounded text-xs text-gray-300 transition-colors hover:bg-gray-700"
                          title={execId}
                        >
                          <span className="min-w-0 flex-1 text-[11px] font-mono truncate text-left">
                            {expanded ? '▼' : '▶'} execution: {execId}
                          </span>
                          <span className="ml-2 shrink-0 text-gray-400">{execCps.length}</span>
                        </button>

                        {expanded ? (
                          execCps.map((cp) => {
                            const isActive = effectiveSelectedKey
                              ? cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
                                (!effectiveSelectedKey.execution_id || cp.execution_id === effectiveSelectedKey.execution_id)
                              : false;
                            return (
                              <div
                                key={`${cp.execution_id}:${cp.checkpoint_id}`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  loadCheckpoint(cp.checkpoint_id, cp.execution_id);
                                }}
                                className={`p-3 rounded cursor-pointer transition-colors ${
                                  isActive
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                                }`}
                              >
                                <div className="flex justify-between items-center gap-3">
                                  <span className="text-sm font-medium">
                                    Checkpoint #{cp.sequence_number || '?'}
                                  </span>
                                  <div className="flex items-center gap-2 shrink-0">
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleRestoreCheckpoint(cp.checkpoint_id, cp.execution_id);
                                      }}
                                      className={`px-2 py-1 text-xs rounded transition-colors ${
                                        isActive
                                          ? 'bg-blue-500 hover:bg-blue-400 text-white'
                                          : 'bg-green-600 hover:bg-green-500 text-white'
                                      }`}
                                      title="Restore execution from this checkpoint"
                                    >
                                      Restore
                                    </button>
                                    <span className={isActive ? 'text-blue-200' : 'text-gray-400'}>
                                      {new Date(cp.created_at).toLocaleString()}
                                    </span>
                                  </div>
                                </div>
                                <div className={`text-xs mt-1 ${isActive ? 'text-blue-200' : 'text-gray-400'}`}>
                                  Trigger: {cp.trigger}
                                </div>

                                {(() => {
                                  const nodeIds = getCheckpointNodeChips(cp);
                                  if (nodeIds.length === 0) return null;
                                  const visible = nodeIds.slice(0, 10);
                                  const overflow = nodeIds.length - visible.length;
                                  return (
                                    <div className="mt-2 flex flex-wrap gap-1">
                                      {visible.map((nodeId) => (
                                        <span
                                          key={nodeId}
                                          className={`px-2 py-0.5 rounded text-[10px] border ${
                                            isActive
                                              ? 'border-blue-300/40 bg-blue-500/20 text-blue-100'
                                              : 'border-gray-500/40 bg-gray-800/40 text-gray-200'
                                          }`}
                                          title={nodeId}
                                        >
                                          {nodeId}
                                        </span>
                                      ))}
                                      {overflow > 0 ? (
                                        <span
                                          className={`px-2 py-0.5 rounded text-[10px] border ${
                                            isActive
                                              ? 'border-blue-300/40 bg-blue-500/20 text-blue-100'
                                              : 'border-gray-500/40 bg-gray-800/40 text-gray-200'
                                          }`}
                                          title={`${overflow} more nodes`}
                                        >
                                          +{overflow}
                                        </span>
                                      ) : null}
                                    </div>
                                  );
                                })()}
                                <div className={`text-xs mt-1 font-mono ${isActive ? 'text-blue-300' : 'text-gray-500'}`}>
                                  ID: {cp.checkpoint_id}
                                </div>
                              </div>
                            );
                          })
                        ) : null}
                      </div>
                    );
                  })
                )}
              </div>
            )}

            {activeTab === 'context' && (
              <div className="space-y-3">
                {viewExecution ? (
                  <>
                    {(() => {
                      const contextBundle = viewExecution.metadata?.context_bundle;

                      const runSettingsDisplay = (() => {
                        const raw = contextBundle?.run_settings;
                        if (!raw || typeof raw !== 'object') return raw;
                        const normalized: Record<string, any> = { ...(raw as any) };
                        if (normalized.parallel_tool_calls === null) {
                          normalized.parallel_tool_calls = 'auto';
                        }
                        return normalized;
                      })();

                      const contextEntries = [
                        { label: 'System Context', value: contextBundle?.system_context },
                        { label: 'Session Context', value: contextBundle?.session_context },
                        { label: 'Run Settings', value: runSettingsDisplay },
                        { label: 'Memory Items', value: contextBundle?.memory_items },
                        { label: 'RAG Chunks', value: contextBundle?.rag_chunks },
                        { label: 'Tool State', value: contextBundle?.tool_state },
                        { label: 'Conflicts', value: contextBundle?.conflicts },
                      ];
                      const hasExecutionContext = Object.keys(viewExecution.context || {}).length > 0;
                      const hasBundle = contextEntries.some((entry) => {
                        if (Array.isArray(entry.value)) {
                          return entry.value.length > 0;
                        }
                        return entry.value && Object.keys(entry.value || {}).length > 0;
                      });

                      if (!hasBundle || hasExecutionContext) {
                        return null;
                      }

                      return (
                        <div className="bg-gray-900/70 border border-gray-700 rounded p-3 space-y-3">
                          <div className="text-xs font-semibold text-gray-300">Context Bundle</div>
                          {contextEntries
                            .filter((entry) => {
                              if (Array.isArray(entry.value)) {
                                return entry.value.length > 0;
                              }
                              return entry.value && Object.keys(entry.value || {}).length > 0;
                            })
                            .map((entry) => (
                              <div key={entry.label}>
                                <div className="text-[11px] text-gray-400 mb-1">{entry.label}</div>
                                <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap">
                                  {JSON.stringify(entry.value, null, 2)}
                                </pre>
                              </div>
                            ))}
                        </div>
                      );
                    })()}
                    {Object.keys(viewExecution.context || {}).length > 0 && (
                      <div className="bg-gray-900/70 border border-gray-700 rounded p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-semibold text-gray-300">Context</span>
                        {contextDiff && (
                          <span className="text-[10px] text-blue-300">
                            Δ {contextDiff.contextKeysChanged.length} keys
                          </span>
                        )}
                      </div>
                      {contextDiff?.contextKeysChanged.length ? (
                        <div className="text-[10px] text-gray-400 mb-2">
                          Changed: {contextDiff.contextKeysChanged.join(', ')}
                        </div>
                      ) : null}
                      <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap">
                        {JSON.stringify(viewExecution.context || {}, null, 2)}
                      </pre>
                      </div>
                    )}
                    <div className="bg-gray-900/70 border border-gray-700 rounded p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-semibold text-gray-300">Metadata</span>
                        {contextDiff && (
                          <span className="text-[10px] text-blue-300">
                            Δ {contextDiff.metadataKeysChanged.length} keys
                          </span>
                        )}
                      </div>
                      {contextDiff?.metadataKeysChanged.length ? (
                        <div className="text-[10px] text-gray-400 mb-2">
                          Changed: {contextDiff.metadataKeysChanged.join(', ')}
                        </div>
                      ) : null}
                      <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap">
                        {JSON.stringify(
                          Object.fromEntries(
                            Object.entries(viewExecution.metadata || {}).filter(
                              ([key]) => key !== 'context_bundle' && key !== 'position' && key !== 'layout',
                            ),
                          ),
                          null,
                          2,
                        )}
                      </pre>
                    </div>
                  </>
                ) : (
                  <div className="text-center text-gray-400 py-8">
                    No execution context available
                  </div>
                )}
              </div>
            )}

            {activeTab === 'logs' && <LogsPanel viewExecution={viewExecution} />}

            {activeTab === 'verification' && (
              <div className="space-y-3">
                {viewExecution && Object.keys(viewExecution.node_executions || {}).length > 0 ? (
                  Object.entries(viewExecution.node_executions || {}).map(([nodeId, nodeExec]) => {
                    const hasPassed = nodeExec.status === 'completed';
                    const hasFailed = nodeExec.status === 'failed';

                    return (
                      <div key={nodeId} className="p-3 bg-gray-900 rounded">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold text-gray-200">{nodeId}</span>
                          {hasPassed && <span className="text-green-500">✓ Passed</span>}
                          {hasFailed && <span className="text-red-500">✗ Failed</span>}
                          {!hasPassed && !hasFailed && <span className="text-gray-400">⏸ Pending</span>}
                        </div>
                        <div className="space-y-1 text-xs">
                          <div className="flex items-center gap-2">
                            <span className={hasPassed ? "text-green-500" : "text-gray-500"}>
                              {hasPassed ? "✓" : "○"}
                            </span>
                            <span className="text-gray-300">Status: {nodeExec.status}</span>
                          </div>
                          {nodeExec.started_at && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-500">○</span>
                              <span className="text-gray-300">
                                Started: {new Date(nodeExec.started_at).toLocaleTimeString()}
                              </span>
                            </div>
                          )}
                          {nodeExec.completed_at && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-500">○</span>
                              <span className="text-gray-300">
                                Completed: {new Date(nodeExec.completed_at).toLocaleTimeString()}
                              </span>
                            </div>
                          )}
                          {nodeExec.error && (
                            <div className="flex items-center gap-2">
                              <span className="text-red-500">✗</span>
                              <span className="text-red-300">Error: {nodeExec.error}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center text-gray-400 py-8">
                    No verification data available - start execution to see results
                  </div>
                )}
              </div>
            )}

            {activeTab === 'compare' && <ComparePanel />}
          </div>

          {showRestoreDialog && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
              <div className="bg-gray-800 border border-gray-700 rounded-lg shadow-xl w-[420px] max-w-[90vw]">
                <div className="p-4 border-b border-gray-700">
                  <h3 className="text-sm font-semibold text-gray-200">Restore Checkpoint</h3>
                  <p className="text-xs text-gray-400 mt-1">
                    Choose how to replay after restoring.
                  </p>
                </div>

                <div className="p-4 space-y-3">
                  <button
                    onClick={() => handleConfirmRestoreDialog('deterministic')}
                    className="w-full p-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-white transition-colors"
                    type="button"
                  >
                    Deterministic replay (use recorded output)
                  </button>

                  <button
                    onClick={() => handleConfirmRestoreDialog('fresh')}
                    className="w-full p-2 bg-blue-600 hover:bg-blue-700 rounded text-sm text-white transition-colors"
                    type="button"
                  >
                    Fresh replay (re-run nodes with current settings)
                  </button>
                </div>

                <div className="p-3 border-t border-gray-700 flex justify-end">
                  <button
                    onClick={handleCancelRestoreDialog}
                    className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-white transition-colors"
                    type="button"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
