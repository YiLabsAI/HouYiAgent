import React from 'react';

export interface ObservabilityLogsViewProps {
  viewExecution: any;
  nodeObservations: Record<string, Record<string, Record<string, any>>>;
  executionLineageMap?: Record<string, { parentExecutionId: string; parentCheckpointId?: string; replayMode?: string }>;
}

const toMs = (v: unknown): number | null => {
  if (typeof v === 'number') return v < 1e12 ? v * 1000 : v; // epoch seconds → ms
  if (typeof v === 'string') { const p = Date.parse(v); return Number.isFinite(p) ? p : null; }
  return null;
};

const formatDurationMs = (start: unknown, end: unknown) => {
  const s = toMs(start);
  const e = toMs(end);
  if (s == null || e == null) return null;
  const ms = Math.max(0, e - s);
  if (ms < 1) return `${Math.round(ms * 1000)}µs`;
  if (ms < 1000) return `${Math.round(ms * 10) / 10}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const formatTime = (v: unknown): string => {
  const ms = toMs(v);
  if (ms == null) return '--';
  return new Date(ms).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 } as Intl.DateTimeFormatOptions);
};

export const ObservabilityLogsView: React.FC<ObservabilityLogsViewProps> = ({ viewExecution, nodeObservations, executionLineageMap }) => {
  const [selectedExecId, setSelectedExecId] = React.useState<string | null>(null);

  // Available execution IDs from nodeObservations (most recent last)
  const availableExecIds = React.useMemo(() => {
    return Object.keys(nodeObservations || {}).sort();
  }, [nodeObservations]);

  const effectiveExecId = selectedExecId || viewExecution?.execution_id;

  const observationSpans = React.useMemo(() => {
    if (!effectiveExecId) return [] as Array<{ nodeId: string; obs: Record<string, any> }>;
    const byNode = (nodeObservations || {})[effectiveExecId] || {};
    return Object.entries(byNode)
      .map(([nodeId, obs]) => ({ nodeId, obs: (obs || {}) as Record<string, any> }))
      .sort((a, b) => a.nodeId.localeCompare(b.nodeId));
  }, [nodeObservations, effectiveExecId]);

  if (!effectiveExecId && availableExecIds.length === 0) {
    return <div className="text-gray-400 text-center py-8 text-xs">No execution selected</div>;
  }

  return (
    <div className="space-y-2 text-xs">
      {/* Execution selector */}
      {availableExecIds.length > 1 && (
        <div className="flex items-center gap-2 pb-1 border-b border-gray-700/50">
          <span className="text-[10px] text-gray-500 shrink-0">Execution:</span>
          <select
            className="flex-1 min-w-0 px-2 py-0.5 text-[11px] bg-gray-800 border border-gray-700 rounded text-gray-300 font-mono truncate"
            value={effectiveExecId || ''}
            onChange={(e) => setSelectedExecId(e.target.value || null)}
          >
            {availableExecIds.map((eid, idx) => {
              const lineage = executionLineageMap?.[eid];
              const isCurrent = eid === viewExecution?.execution_id;
              const modeLabel = lineage?.replayMode === 'deterministic' ? ' ⟳det' : lineage?.replayMode === 'fresh' ? ' ⟳fresh' : '';
              const shortEid = eid.length > 16 ? eid.slice(0, 8) + '…' + eid.slice(-4) : eid;
              return (
                <option key={eid} value={eid}>
                  #{idx + 1} {shortEid}{modeLabel}{isCurrent ? ' (current)' : ''}
                </option>
              );
            })}
          </select>
        </div>
      )}

      {observationSpans.length === 0 ? (
        <div className="text-gray-400 text-center py-4">No span observations for this execution</div>
      ) : observationSpans.map(({ nodeId, obs }) => {
        const status = (obs.status as string | undefined) ?? 'unknown';
        const start = (obs.start_time as string | undefined) ?? null;
        const end = (obs.end_time as string | undefined) ?? null;
        const duration = formatDurationMs(start, end);
        const cacheHit = Boolean(obs.cache_hit || obs.attributes?.['llm.cache_hit']);
        const replayMode = (obs.attributes?.['llm.replay_mode'] as string | undefined) ?? null;

        return (
          <details key={nodeId} className="bg-gray-900/60 border border-gray-700 rounded p-2">
            <summary className="flex items-center gap-3 cursor-pointer select-none">
              <div className="w-28 shrink-0 font-mono text-[11px] text-gray-200 truncate" title={nodeId}>
                {nodeId}
              </div>
              <div className="w-20 shrink-0 text-[11px] text-gray-300">{status}</div>
              <div className="w-20 shrink-0 text-[11px] text-gray-400">{duration ?? '--'}</div>
              {cacheHit && (
                <div className="shrink-0 text-[10px] text-green-400 font-medium">cached</div>
              )}
              {replayMode && (
                <div className="shrink-0 text-[10px] text-purple-400">({replayMode})</div>
              )}
              <div className="flex-1 min-w-0 text-[11px] text-gray-500 truncate" title={String(start ?? '')}>
                start: {formatTime(start)}
              </div>
              <div className="flex-1 min-w-0 text-[11px] text-gray-500 truncate" title={String(end ?? '')}>
                end: {formatTime(end)}
              </div>
            </summary>
            <pre className="mt-2 text-[11px] text-gray-300 font-mono whitespace-pre-wrap">{JSON.stringify(obs, null, 2)}</pre>
          </details>
        );
      })}
      <div className="text-[10px] text-gray-500 pt-1">NOTE: reasoning/artifacts are hidden by default in B1.</div>
    </div>
  );
};
