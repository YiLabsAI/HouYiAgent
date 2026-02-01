import React from 'react';

export interface ObservabilityLogsViewProps {
  viewExecution: any;
  nodeObservations: Record<string, Record<string, Record<string, any>>>;
}

const formatDurationMs = (start?: string | null, end?: string | null) => {
  if (!start || !end) return null;
  const startMs = Date.parse(start);
  const endMs = Date.parse(end);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return null;
  return `${Math.max(0, endMs - startMs)}ms`;
};

export const ObservabilityLogsView: React.FC<ObservabilityLogsViewProps> = ({ viewExecution, nodeObservations }) => {
  const observationSpans = React.useMemo(() => {
    const execId = viewExecution?.execution_id;
    if (!execId) return [] as Array<{ nodeId: string; obs: Record<string, any> }>;
    const byNode = (nodeObservations || {})[execId] || {};
    return Object.entries(byNode)
      .map(([nodeId, obs]) => ({ nodeId, obs: (obs || {}) as Record<string, any> }))
      .sort((a, b) => a.nodeId.localeCompare(b.nodeId));
  }, [nodeObservations, viewExecution?.execution_id]);

  if (!viewExecution?.execution_id) {
    return <div className="text-gray-400 text-center py-8 text-xs">No execution selected</div>;
  }

  if (observationSpans.length === 0) {
    return <div className="text-gray-400 text-center py-8 text-xs">No span observations yet</div>;
  }

  return (
    <div className="space-y-2 text-xs">
      {observationSpans.map(({ nodeId, obs }) => {
        const status = (obs.status as string | undefined) ?? 'unknown';
        const start = (obs.start_time as string | undefined) ?? null;
        const end = (obs.end_time as string | undefined) ?? null;
        const duration = formatDurationMs(start, end);

        return (
          <details key={nodeId} className="bg-gray-900/60 border border-gray-700 rounded p-2">
            <summary className="flex items-center gap-3 cursor-pointer select-none">
              <div className="w-28 shrink-0 font-mono text-[11px] text-gray-200 truncate" title={nodeId}>
                {nodeId}
              </div>
              <div className="w-20 shrink-0 text-[11px] text-gray-300">{status}</div>
              <div className="w-20 shrink-0 text-[11px] text-gray-400">{duration ?? '--'}</div>
              <div className="flex-1 min-w-0 text-[11px] text-gray-500 truncate" title={start ?? ''}>
                start: {start ? new Date(start).toLocaleTimeString() : '--'}
              </div>
              <div className="flex-1 min-w-0 text-[11px] text-gray-500 truncate" title={end ?? ''}>
                end: {end ? new Date(end).toLocaleTimeString() : '--'}
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
