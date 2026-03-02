/**
 * Panel — bottom tabbed panel for execution replay, observability, logs, etc.
 */
import React from 'react';
import { ComparePanel } from './panels/ComparePanel';
import { LogsPanel } from './panels/LogsPanel';
import { MetricsPanel } from './panels/MetricsPanel';
import { ObsFullView } from './panels/ObsFullView';
import { TimelineWaterfall } from './panels/TimelineWaterfall';
import { useConsoleStore } from '../stores/useConsoleStore';
import { diffExecutions } from '@/utils/diff';
import { ChevronUp, ChevronDown, Activity, Flag, FileText, Maximize2, ArrowLeft } from 'lucide-react';

interface ExecTreeNode {
  execId: string;
  checkpoints: any[];
  parentExecId: string | undefined;
  replayMode: string | undefined;
  parentCheckpointId: string | undefined;
  children: ExecTreeNode[];
}

interface ExecTreeNodeViewProps {
  node: ExecTreeNode;
  depth: number;
  isCheckpointGroupExpanded: (execId: string) => boolean;
  setExpandedCheckpointExecutions: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  effectiveSelectedKey: { checkpoint_id: string; execution_id?: string } | null;
  loadCheckpoint: (checkpointId: string, executionId?: string) => void;
  handleRestoreCheckpoint: (checkpointId: string, executionId?: string) => void;
  getCheckpointNodeChips: (cp: any) => string[];
  /** Set of node IDs that are the last node in the plan — used to detect terminal checkpoints */
  lastNodeIds: Set<string>;
  /** Callback to toggle a checkpoint's checked state for compare mode. */
  onToggleCheck?: (checkpointId: string, executionId: string) => void;
  /** Currently checked checkpoints for compare selection. */
  checkedCheckpoints?: Array<{ checkpointId: string; executionId: string }>;
}

const ExecTreeNodeView: React.FC<ExecTreeNodeViewProps> = ({
  node,
  depth,
  isCheckpointGroupExpanded,
  setExpandedCheckpointExecutions,
  effectiveSelectedKey,
  loadCheckpoint,
  handleRestoreCheckpoint,
  getCheckpointNodeChips,
  lastNodeIds,
  onToggleCheck,
  checkedCheckpoints = [],
}) => {
  const expanded = isCheckpointGroupExpanded(node.execId);
  const { parentExecId, replayMode, parentCheckpointId } = node;
  const execCps = node.checkpoints;

  return (
    <div className={`space-y-1 ${depth > 0 ? 'ml-4 border-l-2 border-blue-500/30 pl-2' : ''}`}>
      <button
        type="button"
        onClick={() => {
          setExpandedCheckpointExecutions((prev) => ({
            ...prev,
            [node.execId]: !expanded,
          }));
        }}
        className="w-full flex items-center justify-between px-2 py-1 bg-gray-800 border border-gray-700/50 rounded text-xs text-gray-300 transition-colors hover:bg-gray-700"
        title={node.execId}
      >
        <span className="min-w-0 flex-1 text-[11px] font-mono truncate text-left">
          {expanded ? '▼' : '▶'}{' '}
          {parentExecId ? '↳ fork' : 'execution'}: {node.execId.slice(0, 16)}...
          {replayMode && (
            <span className="ml-1 text-[10px] text-purple-400">
              ({replayMode})
            </span>
          )}
        </span>
        <span className="ml-2 shrink-0 text-gray-400">{execCps.length}</span>
      </button>
      {parentExecId && parentCheckpointId && (
        <div className="px-2 text-[10px] text-gray-500">
          from <span className="font-mono">{parentExecId.slice(0, 12)}...</span>
          {' '}cp <span className="font-mono">{parentCheckpointId.slice(0, 12)}...</span>
        </div>
      )}

      {expanded && (
        <>
          {execCps.map((cp: any) => {
            const isActive = effectiveSelectedKey
              ? cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
                (!effectiveSelectedKey.execution_id || cp.execution_id === effectiveSelectedKey.execution_id)
              : false;
            const triggerNodeId = cp.metadata?.trigger_node_id as string | undefined;
            const isTerminal = Boolean(triggerNodeId && lastNodeIds.has(triggerNodeId));
            const isChecked = checkedCheckpoints.some(
              (c) => c.checkpointId === cp.checkpoint_id && c.executionId === cp.execution_id,
            );
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
                  <div className="flex items-center gap-2">
                    {/* Checkbox for compare selection */}
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={(e) => {
                        e.stopPropagation();
                        onToggleCheck?.(cp.checkpoint_id, cp.execution_id);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="w-3.5 h-3.5 rounded border-gray-500 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 cursor-pointer shrink-0"
                      title="Select for compare"
                      aria-label={`Select checkpoint #${cp.sequence_number || '?'} for compare`}
                    />
                    <span className="text-sm font-medium">
                      Checkpoint #{cp.sequence_number || '?'}
                      {isTerminal && <span className="ml-1 text-[10px] text-amber-400">(terminal)</span>}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRestoreCheckpoint(cp.checkpoint_id, cp.execution_id);
                      }}
                      className={`px-2 py-1 text-xs rounded transition-colors ${
                        isTerminal
                          ? (isActive ? 'bg-amber-500 hover:bg-amber-400 text-white' : 'bg-amber-600 hover:bg-amber-500 text-white')
                          : (isActive ? 'bg-blue-500 hover:bg-blue-400 text-white' : 'bg-green-600 hover:bg-green-500 text-white')
                      }`}
                      title={isTerminal ? "Replay all nodes from the beginning" : "Restore execution from this checkpoint"}
                    >
                      {isTerminal ? 'Replay All' : 'Restore'}
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
                      {visible.map((nodeId: string) => (
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
          })}
        </>
      )}

      {/* Recursively render children (forked executions) */}
      {node.children.length > 0 && (
        <div className="space-y-1">
          {node.children.map((child) => (
            <ExecTreeNodeView
              key={child.execId}
              node={child}
              depth={depth + 1}
              isCheckpointGroupExpanded={isCheckpointGroupExpanded}
              setExpandedCheckpointExecutions={setExpandedCheckpointExecutions}
              effectiveSelectedKey={effectiveSelectedKey}
              loadCheckpoint={loadCheckpoint}
              handleRestoreCheckpoint={handleRestoreCheckpoint}
              getCheckpointNodeChips={getCheckpointNodeChips}
              lastNodeIds={lastNodeIds}
              onToggleCheck={onToggleCheck}
              checkedCheckpoints={checkedCheckpoints}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface BottomPanelProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  activeTab?: 'observability' | 'checkpoints' | 'context' | 'logs';
  onTabChange?: (
    tab: 'observability' | 'checkpoints' | 'context' | 'logs',
  ) => void;
  height?: number;
  /** When true, ignores `height` and stretches to fill the parent container. */
  fillHeight?: boolean;
  /** Called when user clicks expand button to open current tab in Center Stage L. */
  onExpandTab?: (tab: TabType) => void;
  /** Controlled checked checkpoints (shared between bottom panel and CenterStage). */
  checkedCheckpoints?: Array<{ checkpointId: string; executionId: string }>;
  onCheckedCheckpointsChange?: (checks: Array<{ checkpointId: string; executionId: string }>) => void;
}

type TabType = 'observability' | 'checkpoints' | 'context' | 'logs';

export const BottomPanel: React.FC<BottomPanelProps> = ({
  isCollapsed,
  onToggleCollapse,
  activeTab: controlledActiveTab,
  onTabChange,
  height = 280,
  fillHeight = false,
  onExpandTab,
  checkedCheckpoints: controlledChecks,
  onCheckedCheckpointsChange,
}) => {
  const [uncontrolledActiveTab, setUncontrolledActiveTab] = React.useState<TabType>('observability');
  const [showObsFullView, setShowObsFullView] = React.useState(false);
  // Checkpoints: internal dual-view (list / compare)
  const [replayView, setReplayView] = React.useState<'list' | 'compare'>('list');
  const [internalChecks, setInternalChecks] = React.useState<Array<{ checkpointId: string; executionId: string }>>([]);
  const checkedCheckpoints = controlledChecks ?? internalChecks;
  const setCheckedCheckpoints = (val: Array<{ checkpointId: string; executionId: string }> | ((prev: Array<{ checkpointId: string; executionId: string }>) => Array<{ checkpointId: string; executionId: string }>)) => {
    if (onCheckedCheckpointsChange) {
      const next = typeof val === 'function' ? val(checkedCheckpoints) : val;
      onCheckedCheckpointsChange(next);
    } else {
      setInternalChecks(val as any);
    }
  };
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
    executionLineageMap,
  } = useConsoleStore();

  const viewExecution = React.useMemo(() => {
    return viewMode === 'checkpoint' ? checkpointExecution : liveExecution || currentExecution;
  }, [viewMode, checkpointExecution, liveExecution, currentExecution]);

  const effectiveSelectedKey = selectedCheckpointKey;

  const checkpointTree = React.useMemo((): ExecTreeNode[] => {
    // 1. Group checkpoints by execution_id
    const byExec = new Map<string, any[]>();
    checkpoints.forEach((cp) => {
      const execId = cp.execution_id || 'unknown';
      const list = byExec.get(execId) ?? [];
      list.push(cp);
      byExec.set(execId, list);
    });

    // 2. Build flat nodes with sorted checkpoints and resolved lineage
    const nodeMap = new Map<string, ExecTreeNode>();
    for (const [execId, list] of byExec) {
      const sorted = [...list].sort((a, b) => {
        const aSeq = typeof a.sequence_number === 'number' ? a.sequence_number : 0;
        const bSeq = typeof b.sequence_number === 'number' ? b.sequence_number : 0;
        if (aSeq !== bSeq) return bSeq - aSeq;
        const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
        return bTime - aTime;
      });

      // Resolve parent from executionLineageMap first, fallback to snapshot metadata
      const lineage = executionLineageMap[execId];
      const snapshotMeta = sorted[0]?.execution_snapshot?.metadata;
      const parentExecId = (lineage?.parentExecutionId ?? snapshotMeta?.parent_execution_id) as string | undefined;
      const replayMode = (lineage?.replayMode ?? snapshotMeta?.replay_mode) as string | undefined;
      const parentCheckpointId = (lineage?.parentCheckpointId ?? snapshotMeta?.parent_checkpoint_id) as string | undefined;

      nodeMap.set(execId, {
        execId,
        checkpoints: sorted,
        parentExecId,
        replayMode,
        parentCheckpointId,
        children: [],
      });
    }

    // 3. Build tree: attach children to parents
    const roots: ExecTreeNode[] = [];
    for (const node of nodeMap.values()) {
      if (node.parentExecId && nodeMap.has(node.parentExecId)) {
        nodeMap.get(node.parentExecId)!.children.push(node);
      } else {
        roots.push(node);
      }
    }

    return roots;
  }, [checkpoints, executionLineageMap]);

  // Keep groupedCheckpoints as a flat list for backward compat (isCheckpointGroupExpanded)
  const groupedCheckpoints = React.useMemo(() => {
    const flat: Array<{ execId: string; checkpoints: any[] }> = [];
    const walk = (nodes: ExecTreeNode[]) => {
      for (const n of nodes) {
        flat.push({ execId: n.execId, checkpoints: n.checkpoints });
        walk(n.children);
      }
    };
    walk(checkpointTree);
    return flat;
  }, [checkpointTree]);

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

  const contextDiff = React.useMemo(() => {
    if (viewMode !== 'checkpoint' || !liveExecution || !checkpointExecution) return null;
    return diffExecutions(liveExecution, checkpointExecution);
  }, [viewMode, liveExecution, checkpointExecution]);

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

  // Compute the set of "last node IDs" (DAG sink nodes) from the plan to detect
  // terminal checkpoints. A sink node has no outgoing edges in the plan DAG.
  // A checkpoint is terminal if its trigger_node_id is a sink node.
  const lastNodeIds = React.useMemo(() => {
    const nodes = currentPlan?.nodes;
    const edges = currentPlan?.edges;
    if (!Array.isArray(nodes) || nodes.length === 0) return new Set<string>();

    const allNodeIds = new Set<string>(
      nodes.map((n: any) => n.node_id).filter(Boolean)
    );
    // Nodes that are a source of at least one edge are NOT sinks
    const sourcesWithOutEdges = new Set<string>();
    if (Array.isArray(edges)) {
      for (const e of edges) {
        const src = e.source_node_id;
        if (src) sourcesWithOutEdges.add(src);
      }
    }
    // Sink nodes = all nodes minus those with outgoing edges
    const sinks = new Set<string>();
    for (const id of allNodeIds) {
      if (!sourcesWithOutEdges.has(id)) sinks.add(id);
    }
    // Fallback: if no sinks found (e.g. cyclic graph), use last node in array
    if (sinks.size === 0 && nodes.length > 0) {
      const last = nodes[nodes.length - 1];
      if (last?.node_id) sinks.add(last.node_id);
    }
    return sinks;
  }, [currentPlan?.nodes, currentPlan?.edges]);

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
    { id: 'observability', label: 'Observability' },
    { id: 'checkpoints', label: 'Checkpoints' },
    { id: 'context', label: 'Context' },
    { id: 'logs', label: 'Logs' },
  ];

  // Auto-switch to compare view when 2 checkpoints are checked
  React.useEffect(() => {
    if (checkedCheckpoints.length >= 2) {
      setReplayView('compare');
    }
  }, [checkedCheckpoints]);

  const handleToggleCheckpointCheck = React.useCallback(
    (checkpointId: string, executionId: string) => {
      setCheckedCheckpoints((prev) => {
        const exists = prev.some(
          (c) => c.checkpointId === checkpointId && c.executionId === executionId,
        );
        if (exists) {
          return prev.filter(
            (c) => !(c.checkpointId === checkpointId && c.executionId === executionId),
          );
        }
        // Keep only last 2
        const next = [...prev, { checkpointId, executionId }];
        return next.length > 2 ? next.slice(-2) : next;
      });
    },
    [],
  );

  const handleBackToList = React.useCallback(() => {
    setReplayView('list');
    setCheckedCheckpoints([]);
  }, []);

  return (
    <div
      className={`bg-gray-800 border-t border-gray-700 flex flex-col ${fillHeight ? 'h-full' : ''}`}
      style={fillHeight ? undefined : { height: isCollapsed ? 32 : height }}
    >
      {isCollapsed ? (
        <div className="h-full flex items-center justify-between px-4">
          <div className="flex gap-2 text-xs text-gray-400">
            <span>Checkpoints</span>
            <span>•</span>
            <span>Observability</span>
            <span>•</span>
            <span>Context</span>
          </div>
          <button
            onClick={onToggleCollapse}
            className="p-1 hover:bg-gray-700 rounded text-gray-400"
            title="Expand panel"
          >
            <ChevronUp size={16} />
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
                      ? 'bg-gray-700 text-gray-50 border-b-2 border-blue-500'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 pr-2">
              {onExpandTab && (
                <button
                  onClick={() => onExpandTab(activeTab)}
                  className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
                  title="Expand to Center Stage"
                  type="button"
                >
                  <Maximize2 size={14} />
                </button>
              )}
              {activeTab === 'observability' && (
                <button
                  onClick={() => setShowObsFullView(true)}
                  className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
                  title="Open full observability view (waterfall + metrics)"
                  type="button"
                >
                  <Activity size={16} />
                </button>
              )}
              {activeTab === 'checkpoints' && checkedCheckpoints.length >= 2 && replayView === 'list' && (
                <button
                  onClick={() => setReplayView('compare')}
                  className="px-2 py-1 text-[11px] rounded border border-blue-500 text-blue-300 hover:bg-blue-600/20 transition-colors"
                  title={`Compare ${checkedCheckpoints.length} selected checkpoints`}
                  type="button"
                >
                  Compare ({checkedCheckpoints.length})
                </button>
              )}
              {activeTab === 'checkpoints' && checkedCheckpoints.length > 0 && (
                <button
                  onClick={() => {
                    setCheckedCheckpoints([]);
                    setReplayView('list');
                  }}
                  className="px-2 py-1 text-[11px] rounded border border-gray-600 text-gray-400 hover:text-gray-200 hover:bg-gray-700 transition-colors"
                  title="Clear checkpoint selection"
                  type="button"
                >
                  Clear
                </button>
              )}
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
                <ChevronDown size={16} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === 'observability' && (
              <div className="flex gap-3 h-full min-h-0">
                <div className="flex-1 min-w-0 min-h-0">
                  <TimelineWaterfall executionId={viewExecution?.execution_id} />
                </div>
                <div className="w-72 flex-shrink-0 min-h-0 overflow-y-auto">
                  <MetricsPanel executionId={viewExecution?.execution_id} />
                </div>
              </div>
            )}

            {activeTab === 'checkpoints' && (
              <div className="h-full">
                {replayView === 'compare' ? (
                  <div className="h-full flex flex-col">
                    <div className="flex items-center gap-2 mb-3">
                      <button
                        type="button"
                        onClick={handleBackToList}
                        className="flex items-center gap-1 px-2 py-1 text-xs text-gray-300 hover:text-gray-100 hover:bg-gray-700 rounded transition-colors"
                      >
                        <ArrowLeft size={12} />
                        Back to List
                      </button>
                      <span className="text-[10px] text-gray-500">
                        Comparing {checkedCheckpoints.length} checkpoints
                      </span>
                    </div>
                    <div className="flex-1 min-h-0">
                      <ComparePanel preSelectedCheckpoints={checkedCheckpoints} />
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {checkpointTree.length === 0 ? (
                      <div className="flex items-center justify-center h-full">
                        <div className="text-center text-gray-500">
                          <Flag size={32} className="mx-auto mb-2 opacity-50" />
                          <div className="text-sm">No checkpoints created yet</div>
                          <div className="text-xs mt-1">Checkpoints will appear here during execution</div>
                          <div className="text-[10px] mt-2 text-gray-600">
                            Check any 2 checkpoints to enter Compare mode
                          </div>
                        </div>
                      </div>
                    ) : (
                      <>
                        {checkedCheckpoints.length > 0 && checkedCheckpoints.length < 2 && (
                          <div className="text-[10px] text-blue-400 mb-1">
                            Select {2 - checkedCheckpoints.length} more checkpoint{checkedCheckpoints.length === 0 ? 's' : ''} to compare
                          </div>
                        )}
                        {checkpointTree.map((rootNode) => (
                          <ExecTreeNodeView
                            key={rootNode.execId}
                            node={rootNode}
                            depth={0}
                            isCheckpointGroupExpanded={isCheckpointGroupExpanded}
                            setExpandedCheckpointExecutions={setExpandedCheckpointExecutions}
                            effectiveSelectedKey={effectiveSelectedKey}
                            loadCheckpoint={loadCheckpoint}
                            handleRestoreCheckpoint={handleRestoreCheckpoint}
                            getCheckpointNodeChips={getCheckpointNodeChips}
                            lastNodeIds={lastNodeIds}
                            onToggleCheck={handleToggleCheckpointCheck}
                            checkedCheckpoints={checkedCheckpoints}
                          />
                        ))}
                      </>
                    )}
                  </div>
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
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center text-gray-500">
                      <FileText size={32} className="mx-auto mb-2 opacity-50" />
                      <div className="text-sm">No execution context available</div>
                      <div className="text-xs mt-1">Context will appear here during execution</div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'logs' && <LogsPanel viewExecution={viewExecution} />}
          </div>

          <ObsFullView isOpen={showObsFullView} onClose={() => setShowObsFullView(false)} />

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
