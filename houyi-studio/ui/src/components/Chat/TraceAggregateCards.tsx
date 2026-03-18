import React from 'react';
import { TraceSection } from './TraceSection';
import { formatDuration } from './TraceDetailUtils';
import type {
  TraceMetricBreakdownEntry,
  TraceMetricKey,
  TraceMetricSummary,
  TraceToolLoopBreakdown,
  TraceToolLoopDecision,
} from './TraceDetailRuntime';

interface TraceAggregateCardsProps {
  aggregates: Record<TraceMetricKey, TraceMetricSummary>;
  selectedMetric: TraceMetricKey | null;
  onSelect: (key: TraceMetricKey) => void;
  metricBreakdown: TraceMetricBreakdownEntry[];
  toolLoopBreakdown: TraceToolLoopBreakdown | null;
  toolLoopMode: string | null;
  toolLoopDecision: TraceToolLoopDecision;
}

const styles: Record<TraceMetricKey, string> = {
  llm: 'border-purple-500/30 bg-purple-500/10 text-purple-300',
  tool: 'border-orange-500/30 bg-orange-500/10 text-orange-300',
  execution: 'border-sky-500/30 bg-sky-500/10 text-sky-300',
};

const labels: Record<TraceMetricKey, string> = {
  llm: 'LLM',
  tool: 'Tool',
  execution: 'Orchestration',
};

export const TraceAggregateCards: React.FC<TraceAggregateCardsProps> = ({
  aggregates,
  selectedMetric,
  onSelect,
  metricBreakdown,
  toolLoopBreakdown,
  toolLoopMode,
  toolLoopDecision,
}) => (
  <TraceSection title="Runtime breakdown" testId="trace-runtime-breakdown">
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
      {(['llm', 'tool', 'execution'] as TraceMetricKey[]).map((key) => {
        const metric = aggregates[key];
        return (
          <button
            key={key}
            type="button"
            title="Aggregate cards summarize span types and may overlap total wall time."
            className={`rounded border px-2 py-2 text-left text-[10px] transition-colors ${styles[key]} ${selectedMetric === key ? 'ring-1 ring-gray-300/50' : 'hover:border-gray-500/50'}`}
            onClick={() => onSelect(key)}
          >
            <div className="font-medium">
              {labels[key]} {metric.count}x · {metric.count > 0 ? formatDuration(metric.totalMs) : '—'}
            </div>
            <div className="mt-1 text-[10px] text-gray-400">Click to inspect composition</div>
          </button>
        );
      })}
    </div>
    {selectedMetric ? (
      <div className="mt-3 rounded border border-gray-800 bg-gray-950/70 px-2 py-2 text-[10px] text-gray-400">
        <div className="mb-1 text-gray-300">{labels[selectedMetric]} composition ({metricBreakdown.length} groups)</div>
        {metricBreakdown.length === 0 ? (
          <div>No spans found.</div>
        ) : (
          <div className="space-y-1">
            {metricBreakdown.slice(0, 12).map((entry) => (
              <div key={entry.name} className="flex items-center justify-between gap-2">
                <span className="truncate">{entry.name} ({entry.count}x)</span>
                <span>{formatDuration(entry.totalMs)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    ) : null}
    {toolLoopBreakdown ? (
      <div className="mt-3 rounded border border-gray-800 bg-gray-950/70 px-2 py-2 text-[10px] text-gray-400">
        <div className="mb-1 text-gray-300">Tool loop runtime</div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded border border-purple-500/30 bg-purple-500/10 px-1.5 py-0.5 text-purple-300">
            LLM {toolLoopBreakdown.llm.percent} · {formatDuration(toolLoopBreakdown.llm.totalMs)}
          </span>
          <span className="rounded border border-orange-500/30 bg-orange-500/10 px-1.5 py-0.5 text-orange-300">
            Tool {toolLoopBreakdown.tool.percent} · {formatDuration(toolLoopBreakdown.tool.totalMs)}
          </span>
          <span className="rounded border border-sky-500/30 bg-sky-500/10 px-1.5 py-0.5 text-sky-300">
            Execution overhead {toolLoopBreakdown.execution.percent} · {formatDuration(toolLoopBreakdown.execution.totalMs)}
          </span>
        </div>
        <div className="mt-2 text-gray-500">
          Execution overhead = tool loop total - LLM - tool
        </div>
      </div>
    ) : null}
    {toolLoopMode === 'disabled_by_request' ? (
      <div className="mt-3 rounded border border-gray-800 bg-gray-950/70 px-2 py-2 text-[10px] text-gray-400">
        <span className="rounded border border-sky-500/30 bg-sky-500/10 px-1.5 py-0.5 text-sky-300">
          Tool loop disabled by request
        </span>
      </div>
    ) : null}
    {toolLoopDecision.strategy || toolLoopDecision.reason ? (
      <div className="mt-2 rounded border border-gray-800 bg-gray-950/70 px-2 py-2 text-[10px] text-gray-400">
        <div className="mb-1 text-gray-300">Tool loop decision</div>
        <div className="flex flex-wrap items-center gap-2">
          {toolLoopDecision.strategy ? (
            <span className="rounded border border-indigo-500/30 bg-indigo-500/10 px-1.5 py-0.5 text-indigo-300">
              Strategy {toolLoopDecision.strategy}
            </span>
          ) : null}
          {toolLoopDecision.reason ? (
            <span className="rounded border border-slate-500/30 bg-slate-500/10 px-1.5 py-0.5 text-slate-300">
              Gate {toolLoopDecision.reason}
            </span>
          ) : null}
        </div>
      </div>
    ) : null}
  </TraceSection>
);
