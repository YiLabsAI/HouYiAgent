import React from 'react';
import { X } from 'lucide-react';
import { TracePayload } from '@/types/chat';
import { TraceAggregateCards } from './TraceAggregateCards';
import { TraceContextGovernance } from './TraceContextGovernance';
import { buildTraceDetailRuntime, type TraceMetricKey } from './TraceDetailRuntime';
import { TracePipelineStages } from './TracePipelineStages';
import { TraceRequestContext } from './TraceRequestContext';
import { TraceSpanTree } from './TraceSpanTree';
import { TraceSummaryStats } from './TraceSummaryStats';

interface TraceDetailPanelProps {
  traceId: string;
  onClose: () => void;
}

export const TraceDetailPanel: React.FC<TraceDetailPanelProps> = ({ traceId, onClose }) => {
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [payload, setPayload] = React.useState<TracePayload | null>(null);
  const [viewMode, setViewMode] = React.useState<'tree' | 'raw'>('tree');
  const [selectedMetric, setSelectedMetric] = React.useState<TraceMetricKey | null>(null);

  React.useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);

    fetch(`/api/chat/trace/${traceId}`)
      .then(async (resp) => {
        if (!resp.ok) {
          let detail = '';
          try {
            const errBody = await resp.json();
            detail = errBody?.detail ? ` - ${String(errBody.detail)}` : '';
          } catch {
            detail = '';
          }
          throw new Error(`Trace API ${resp.status}${detail}`);
        }
        return resp.json();
      })
      .then((data: TracePayload) => {
        if (!mounted) return;
        setPayload(data);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (!mounted) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [traceId]);

  const hasTraceData = Boolean(payload?.root_span);
  const traceRuntime = React.useMemo(
    () => buildTraceDetailRuntime(payload, selectedMetric),
    [payload, selectedMetric],
  );

  return (
    <div className="fixed inset-y-0 right-0 z-40 w-[440px] max-w-[92vw] border-l border-gray-700 bg-gray-900 shadow-2xl">
      <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
        <div className="min-w-0">
          <div className="text-[12px] text-gray-300">Trace Detail</div>
          <code className="block truncate text-[10px] text-gray-500">{traceId}</code>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            className={`rounded px-2 py-0.5 text-[10px] ${viewMode === 'tree' ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:text-gray-200'}`}
            onClick={() => setViewMode('tree')}
          >
            Tree
          </button>
          <button
            type="button"
            className={`rounded px-2 py-0.5 text-[10px] ${viewMode === 'raw' ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:text-gray-200'}`}
            onClick={() => setViewMode('raw')}
          >
            JSON
          </button>
          <button type="button" onClick={onClose} className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300">
            <X size={14} />
          </button>
        </div>
      </div>

      <div className="h-[calc(100%-56px)] overflow-auto p-3">
        {loading ? <div className="text-[12px] text-gray-500">Loading trace...</div> : null}
        {error ? <div className="text-[12px] text-red-400">{error}</div> : null}
        {!loading && !error && !hasTraceData ? (
          <div className="rounded border border-gray-700 bg-gray-850 p-3 text-[12px] text-gray-400">
            Trace data is empty for this id.
            <div className="mt-1 text-[11px] text-gray-500">
              This usually means the trace already expired or observability storage is not retaining spans.
            </div>
          </div>
        ) : null}
        {!loading && !error && hasTraceData && payload ? (
          <div className="space-y-3">
            <TraceSummaryStats totalDurationMs={payload.total_duration_ms} totalTokens={payload.total_tokens} />
            <TraceAggregateCards
              aggregates={traceRuntime.aggregates}
              selectedMetric={selectedMetric}
              onSelect={(key) => setSelectedMetric((current: TraceMetricKey | null) => (current === key ? null : key))}
              metricBreakdown={traceRuntime.metricBreakdown}
              toolLoopBreakdown={traceRuntime.toolLoopBreakdown}
              toolLoopMode={traceRuntime.toolLoopMode}
              toolLoopDecision={traceRuntime.toolLoopDecision}
            />
            <TracePipelineStages stages={traceRuntime.pipelineStages} />
            <TraceRequestContext requestContext={payload.request_context} contextPlan={payload.context_plan} />
            <TraceContextGovernance governance={payload.context_governance} />
            {viewMode === 'tree' && payload.root_span ? <TraceSpanTree rootSpan={payload.root_span} /> : null}
            {viewMode === 'raw' ? (
              <pre className="whitespace-pre-wrap break-words rounded border border-gray-800 bg-gray-950 p-3 text-[11px] leading-relaxed text-gray-300">
                {JSON.stringify(payload, null, 2)}
              </pre>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
};
