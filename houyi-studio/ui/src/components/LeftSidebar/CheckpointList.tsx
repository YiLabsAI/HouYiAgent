import React, { useMemo, useState } from 'react';

// NOTE(core): Legacy sidebar checkpoints list.
// This component is currently NOT rendered in LeftSidebar (checkpoints UI moved to BottomPanel).
// Keep it for now to avoid losing logic while we stabilize the new IA; remove only when migration is fully validated.

interface CheckpointListProps {
  checkpoints: any[];
  viewMode?: 'live' | 'checkpoint';
  selectedCheckpointKey?: { execution_id: string; checkpoint_id: string } | null;
  onExitCheckpointView: () => void;
  lastRestoredCheckpointId: string | null;
  lastRestoredCheckpointKey?: { execution_id: string | null; checkpoint_id: string } | null;
  onLoadCheckpoint: (checkpointId: string, executionId?: string) => void;
  onRestoreCheckpoint: (checkpointId: string, executionId?: string) => void;
}

export const CheckpointList: React.FC<CheckpointListProps> = ({
  checkpoints,
  viewMode,
  selectedCheckpointKey,
  onExitCheckpointView,
  lastRestoredCheckpointId,
  lastRestoredCheckpointKey,
  onLoadCheckpoint,
  onRestoreCheckpoint,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [filterText, setFilterText] = useState('');
  const [expandedExecutions, setExpandedExecutions] = useState<Record<string, boolean>>({});

  const inCheckpointView = viewMode === 'checkpoint';
  const effectiveSelectedKey = selectedCheckpointKey;

  const filteredCheckpoints = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    if (!q) return checkpoints;
    const normalized = q.replace(/^#/, '').trim();
    return checkpoints.filter((cp: any) => {
      const id = String(cp.checkpoint_id || '').toLowerCase();
      const execId = String(cp.execution_id || '').toLowerCase();
      const seq = String(cp.sequence_number ?? '').toLowerCase();
      const idSuffix = id.replace(/^cp_/, '');
      return (
        id.includes(q) ||
        execId.includes(q) ||
        seq.includes(normalized) ||
        idSuffix.includes(normalized)
      );
    });
  }, [checkpoints, filterText]);

  const buildExecutedNodeSummary = (checkpoint: any) => {
    const snapshot = checkpoint?.execution_snapshot;
    const nodeExecs = snapshot?.node_executions;
    if (!nodeExecs || typeof nodeExecs !== 'object') {
      return { executed: [] as string[], total: 0 };
    }
    const executed = Object.entries(nodeExecs)
      .filter(([_nodeId, nodeExec]: any) => {
        const status = nodeExec?.status;
        return status && status !== 'pending' && status !== 'skipped';
      })
      .map(([nodeId]) => String(nodeId))
      .sort((a, b) => a.localeCompare(b));
    return { executed, total: executed.length };
  };

  const grouped = useMemo(() => {
    const byExec: Record<string, any[]> = {};
    const execOrder: string[] = [];
    filteredCheckpoints.forEach((cp: any) => {
      const execId = cp.execution_id || 'unknown';
      if (!byExec[execId]) {
        byExec[execId] = [];
        execOrder.push(execId);
      }
      byExec[execId].push(cp);
    });

    Object.keys(byExec).forEach((execId) => {
      byExec[execId].sort((a: any, b: any) => {
        const aSeq = typeof a.sequence_number === 'number' ? a.sequence_number : 0;
        const bSeq = typeof b.sequence_number === 'number' ? b.sequence_number : 0;
        if (aSeq !== bSeq) return bSeq - aSeq;
        return String(b.created_at || '').localeCompare(String(a.created_at || ''));
      });
    });

    const groups = execOrder.map((execId) => ({ execId, checkpoints: byExec[execId] }));

    // Sort execution groups by the most recent checkpoint first.
    groups.sort((a, b) => {
      const aTop = a.checkpoints?.[0];
      const bTop = b.checkpoints?.[0];
      const aSeq = typeof aTop?.sequence_number === 'number' ? aTop.sequence_number : 0;
      const bSeq = typeof bTop?.sequence_number === 'number' ? bTop.sequence_number : 0;
      if (aSeq !== bSeq) return bSeq - aSeq;
      const aTime = aTop?.created_at ? new Date(aTop.created_at).getTime() : 0;
      const bTime = bTop?.created_at ? new Date(bTop.created_at).getTime() : 0;
      return bTime - aTime;
    });

    return groups;
  }, [filteredCheckpoints]);

  const getDefaultExpanded = React.useCallback((execId: string, groups: Array<{ execId: string; checkpoints: any[] }>) => {
    if (expandedExecutions[execId] !== undefined) {
      return Boolean(expandedExecutions[execId]);
    }
    const selectedExecId = effectiveSelectedKey?.execution_id;
    const restoredExecId = lastRestoredCheckpointKey?.execution_id ?? null;
    if (selectedExecId && execId === selectedExecId) return true;
    if (restoredExecId && execId === restoredExecId) return true;
    return groups.length > 0 && groups[0].execId === execId;
  }, [expandedExecutions, effectiveSelectedKey?.execution_id, lastRestoredCheckpointKey?.execution_id]);

  const handleToggle = () => {
    console.log('[CheckpointList] Toggle clicked, current:', isExpanded, 'will set to:', !isExpanded);
    setIsExpanded(!isExpanded);
  };

  console.log('[CheckpointList] Rendering, isExpanded:', isExpanded, 'checkpoints:', checkpoints.length);

  return (
    <div className="h-full min-h-0 p-3 border-t border-gray-700 flex flex-col">
      <div
        className="flex items-center justify-between cursor-pointer hover:bg-gray-700 transition-colors"
        onClick={handleToggle}
      >
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold text-gray-400">Checkpoints</h3>
          {checkpoints.length > 0 && (
            <span className="text-xs bg-blue-600 text-white px-1.5 py-0.5 rounded">
              {checkpoints.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {inCheckpointView && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onExitCheckpointView();
              }}
              className="px-2 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-[10px] text-white transition-colors"
              title="Return to live execution"
            >
              ← Live
            </button>
          )}
          <span className="text-xs text-gray-400 font-bold">
            {isExpanded ? '▼' : '▶'}
          </span>
        </div>
      </div>

      {isExpanded && (
        <div className="mt-2 flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
          <div className="sticky top-0 z-20 bg-gray-800 pb-2">
            <input
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter: #<seq>, checkpoint_id, or execution_id"
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-200 placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>

            {filteredCheckpoints.length === 0 ? (
              <div className="text-xs text-gray-500 italic p-2 text-center">
                No checkpoints yet
              </div>
            ) : (
              grouped.map(({ execId, checkpoints: execCps }) => {
                const isGroupExpanded = getDefaultExpanded(execId, grouped);
                return (
                  <div key={execId} className="mb-2">
                    <button
                      onClick={() => {
                        setExpandedExecutions((prev) => ({
                          ...prev,
                          [execId]: !(getDefaultExpanded(execId, grouped)),
                        }));
                      }}
                      className="w-full flex items-center justify-between px-2 py-1 bg-gray-800 border border-gray-700/50 rounded text-xs text-gray-300 transition-colors hover:bg-gray-700"
                      title={execId}
                      type="button"
                    >
                      <span className="min-w-0 flex-1 text-[10px] text-left">
                        {isGroupExpanded ? '▼' : '▶'}{' '}
                        <span className="font-mono inline-block max-w-[140px] truncate align-bottom" title={execId}>
                          {String(execId)}
                        </span>
                      </span>
                      <span className="ml-2 shrink-0 text-gray-400">{execCps.length}</span>
                    </button>

                    {isGroupExpanded && (
                      <div className="mt-2 space-y-1">
                        {execCps.map((cp: any) => {
                          const isActive = effectiveSelectedKey
                            ? cp.checkpoint_id === effectiveSelectedKey.checkpoint_id &&
                              (!effectiveSelectedKey.execution_id || cp.execution_id === effectiveSelectedKey.execution_id)
                            : false;
                          const isRestored = lastRestoredCheckpointKey
                            ? cp.checkpoint_id === lastRestoredCheckpointKey.checkpoint_id &&
                              cp.execution_id === lastRestoredCheckpointKey.execution_id
                            : cp.checkpoint_id === lastRestoredCheckpointId;

                          return (
                            <div
                              key={`${cp.execution_id}:${cp.checkpoint_id}`}
                              className={`p-2 rounded transition-all duration-200 ${
                                isActive
                                  ? 'bg-blue-600 text-white'
                                  : isRestored
                                    ? 'bg-gray-700 text-gray-200 border border-green-500/70'
                                    : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                              }`}
                              onClick={() => onLoadCheckpoint(cp.checkpoint_id, cp.execution_id)}
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div
                                  className="flex-1 min-w-0 cursor-pointer"
                                >
                                  <span className="text-xs font-medium">
                                    {isActive ? '📍' : '📌'} Checkpoint #{cp.sequence_number || execCps.indexOf(cp) + 1}
                                  </span>
                                </div>
                                <div className="flex flex-col items-end gap-1 shrink-0">
                                  <span className="text-[10px] leading-4 opacity-75 whitespace-nowrap">
                                    {new Date(cp.created_at).toLocaleTimeString()}
                                  </span>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      onRestoreCheckpoint(cp.checkpoint_id, cp.execution_id);
                                    }}
                                    className={`px-2 py-1 text-xs rounded transition-colors ${
                                      isActive
                                        ? 'bg-blue-500 hover:bg-blue-400 text-white'
                                        : 'bg-green-600 hover:bg-green-500 text-white'
                                    }`}
                                    title="Restore execution from this checkpoint"
                                  >
                                    ↻ Restore
                                  </button>
                                </div>
                              </div>

                              {(() => {
                                const summary = buildExecutedNodeSummary(cp);
                                if (summary.total === 0) return null;
                                const shown = summary.executed.slice(0, 5);
                                const remaining = summary.total - shown.length;
                                const remainingTitle = remaining > 0
                                  ? summary.executed.slice(shown.length).join('\n')
                                  : '';
                                return (
                                  <div className="mt-2 flex items-center gap-1 flex-wrap">
                                    {shown.map((nodeId) => (
                                      <span
                                        key={nodeId}
                                        className="inline-flex items-center gap-1 rounded bg-green-900/40 border border-green-600/40 px-1.5 py-0.5 text-[10px] text-green-200"
                                        title={nodeId}
                                      >
                                        <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-400" />
                                        <span className="max-w-[92px] truncate">{nodeId}</span>
                                      </span>
                                    ))}
                                    {remaining > 0 && (
                                      <span
                                        className="text-[10px] text-gray-400"
                                        title={remainingTitle}
                                      >
                                        +{remaining}
                                      </span>
                                    )}
                                  </div>
                                );
                              })()}
                              <div
                                className={`text-xs mt-1 truncate ${isActive ? 'text-blue-200' : 'text-gray-500'}`}
                              >
                                {cp.checkpoint_id.slice(0, 16)}...
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })
            )}
        </div>
      )}
    </div>
  );
};
