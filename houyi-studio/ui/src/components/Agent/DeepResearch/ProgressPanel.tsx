import React from 'react';
import type { ResearchProgress, SSEEvent } from '@/stores/useResearchStore';
import { Loader2, XCircle, CheckCircle2, Search as SearchIcon, AlertCircle } from 'lucide-react';

interface Props {
  progress: ResearchProgress | null;
  events: SSEEvent[];
  onCancel: () => void;
  error?: string | null;
}

export const ProgressPanel: React.FC<Props> = ({ progress, events, onCancel, error }) => {
  const totalSteps = progress?.total_steps || 0;
  const completedSteps = progress?.completed_steps || 0;
  const pct = totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0;
  const allSearchDone = totalSteps > 0 && completedSteps >= totalSteps;
  const terminated = !!error;
  const idle = terminated;

  const recentEvents = events.slice(-20);

  const eventIcon = (type: string) => {
    if (type.includes('source_found')) return <SearchIcon size={12} className="text-blue-400" />;
    if (type.includes('completed') || type.includes('step_completed')) return <CheckCircle2 size={12} className="text-green-400" />;
    if (type.includes('failed') || type.includes('cancelled')) return <AlertCircle size={12} className="text-red-400" />;
    if (type.includes('report_section')) return <CheckCircle2 size={12} className="text-purple-400" />;
    if (terminated) return <AlertCircle size={12} className="text-red-400" />;
    return <Loader2 size={12} className="text-gray-400 animate-spin" />;
  };

  const eventLabel = (evt: SSEEvent): string => {
    const p = evt.payload;
    if (evt.event_type === 'research.step_started') return `Researching: ${(p.step as string) || 'sub-question'}`;
    if (evt.event_type === 'research.step_completed') return `Completed: ${(p.step as string) || 'step'}`;
    if (evt.event_type === 'research.source_found') return `Found: ${(p.title as string) || (p.url as string) || 'source'}`;
    if (evt.event_type === 'research.agent_spawned') return `Agent started: ${(p.agent_id as string) || ''}`;
    if (evt.event_type === 'research.agent_completed') return `Agent finished: ${(p.agent_id as string) || ''}`;
    if (evt.event_type === 'research.report_section') return `Writing: ${(p.title as string) || 'section'}`;
    if (evt.event_type === 'research.quality_evaluated') return 'Quality evaluation complete';
    if (evt.event_type === 'memory.candidate_extracted') return `Extracted ${(p.count as number) || 0} memory candidates`;
    if (evt.event_type === 'research.failed') return `Failed: ${(p.error as string) || 'unknown error'}`;
    if (evt.event_type === 'research.cancelled') return `Cancelled: ${(p.reason as string) || ''}`;
    return evt.event_type.replace('research.', '');
  };

  const barColor = terminated ? 'bg-red-500' : allSearchDone ? 'bg-green-500' : 'bg-purple-500';
  const stepLabel = terminated
    ? 'Stopped'
    : allSearchDone
      ? 'Writing report...'
      : (progress?.current_step || (completedSteps > 0 ? 'Searching...' : 'Initializing...'));

  return (
    <div className="space-y-6">
      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <div className="flex items-center gap-2">
            {allSearchDone && !terminated && (
              <Loader2 size={12} className="text-purple-400 animate-spin" />
            )}
            <span>{stepLabel}</span>
          </div>
          <span>{allSearchDone && !terminated ? 'Search complete' : `${pct}%`}</span>
        </div>
        <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
          {allSearchDone && !terminated ? (
            <div className="h-full rounded-full bg-purple-500 animate-pulse w-full" />
          ) : (
            <div
              className={`h-full rounded-full ${barColor} transition-all duration-500 ease-out`}
              style={{ width: `${Math.max(pct, terminated ? 100 : 0)}%` }}
            />
          )}
        </div>
        <div className="flex justify-between text-xs text-gray-600">
          <span>{completedSteps} / {totalSteps} search steps</span>
          {progress?.elapsed_seconds != null && (
            <span>{Math.round(progress.elapsed_seconds)}s elapsed</span>
          )}
        </div>
      </div>

      {/* Event log */}
      <div className="rounded-xl border border-gray-700/50 bg-gray-800/30 overflow-hidden">
        <div className="px-4 py-2 border-b border-gray-700/50 flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">Activity Log</span>
          {terminated
            ? <AlertCircle size={14} className="text-red-400" />
            : <Loader2 size={14} className="text-purple-400 animate-spin" />
          }
        </div>
        <div className="max-h-72 overflow-y-auto p-3 space-y-1.5">
          {recentEvents.length === 0 && (
            <p className="text-xs text-gray-600 text-center py-4">Waiting for events...</p>
          )}
          {recentEvents.map((evt) => (
            <div key={evt.event_id} className="flex items-start gap-2 text-xs">
              <div className="pt-0.5 shrink-0">{eventIcon(evt.event_type)}</div>
              <span className="text-gray-300">{eventLabel(evt)}</span>
              <span className="text-gray-700 ml-auto shrink-0">#{evt.sequence}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Cancel — only show when still running */}
      {!idle && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="flex items-center gap-1.5 px-4 py-2 text-sm text-gray-400 hover:text-red-400 border border-gray-700 hover:border-red-700 rounded-lg transition-colors"
          >
            <XCircle size={14} />
            Cancel Research
          </button>
        </div>
      )}
    </div>
  );
};
