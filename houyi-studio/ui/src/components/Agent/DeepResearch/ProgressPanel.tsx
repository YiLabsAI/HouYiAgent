import React, { useEffect, useState } from 'react';
import type { ResearchProgress, SSEEvent, SubQuestion } from '@/stores/useResearchStore';
import { Loader2, XCircle } from 'lucide-react';
import { ThinkingTrajectory } from './ThinkingTrajectory';

interface Props {
  progress: ResearchProgress | null;
  events: SSEEvent[];
  subQuestions?: SubQuestion[];
  onCancel: () => void;
  error?: string | null;
}

function useElapsedTimer(events: SSEEvent[], terminated: boolean): number {
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (events.length > 0 && startTime === null) {
      setStartTime(Date.now());
    }
  }, [events.length, startTime]);

  useEffect(() => {
    if (startTime === null || terminated) return;
    setElapsed(Math.round((Date.now() - startTime) / 1000));
    const interval = setInterval(() => {
      setElapsed(Math.round((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime, terminated]);

  useEffect(() => {
    if (terminated && startTime !== null) {
      setElapsed(Math.round((Date.now() - startTime) / 1000));
    }
  }, [terminated, startTime]);

  return elapsed;
}

export const ProgressPanel: React.FC<Props> = ({
  progress,
  events,
  subQuestions,
  onCancel,
  error,
}) => {
  const totalSteps = progress?.total_steps || 0;
  const rawCompleted = progress?.completed_steps || 0;
  const completedSteps = Math.min(rawCompleted, totalSteps);
  const pct = totalSteps > 0 ? Math.min(Math.round((completedSteps / totalSteps) * 100), 100) : 0;
  const allSearchDone = totalSteps > 0 && completedSteps >= totalSteps;
  const terminated = !!error;
  const elapsed = useElapsedTimer(events, terminated);

  const barColor = terminated
    ? 'bg-red-500'
    : allSearchDone
      ? 'bg-green-500'
      : 'bg-purple-500';

  const PHASE_LABELS: Record<string, string> = {
    conflict_detection: 'Detecting conflicts...',
    report_generation: 'Writing report sections...',
    url_validation: 'Validating URLs...',
    validation: 'Validating sections...',
    quality_evaluation: 'Evaluating quality...',
  };
  const latestPhase = [...events].reverse().find(
    (e) => e.event_type === 'research.pipeline_phase',
  );
  const reportPhaseLabel = latestPhase
    ? PHASE_LABELS[(latestPhase.payload.phase as string)] || 'Writing report...'
    : 'Writing report...';

  const stepLabel = terminated
    ? 'Stopped'
    : allSearchDone
      ? reportPhaseLabel
      : progress?.current_step ||
        (completedSteps > 0 ? 'Searching...' : 'Initializing...');

  return (
    <div className="space-y-6">
      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <div className="flex items-center gap-2">
            {!terminated && (
              <Loader2 size={12} className="text-purple-400 animate-spin" />
            )}
            <span>{stepLabel}</span>
          </div>
          <span>
            {allSearchDone && !terminated ? 'Search complete' : `${pct}%`}
          </span>
        </div>
        <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
          {allSearchDone && !terminated ? (
            <div className="h-full rounded-full bg-purple-500 animate-pulse w-full" />
          ) : (
            <div
              className={`h-full rounded-full ${barColor} transition-all duration-500 ease-out ${pct === 0 && !terminated ? 'animate-pulse' : ''}`}
              style={{
                width: `${Math.max(pct, terminated ? 100 : pct === 0 ? 8 : 0)}%`,
              }}
            />
          )}
        </div>
        <div className="flex justify-between text-xs text-gray-600">
          <span
            title={subQuestions && subQuestions.length > 0
              ? `Sub-questions:\n${subQuestions.map((sq, i) => `${i + 1}. ${sq.question}`).join('\n')}`
              : `Research decomposes the query into ${totalSteps} sub-questions, each searched independently.`}
            className="cursor-help border-b border-dotted border-gray-600"
          >
            {completedSteps} / {totalSteps} sub-questions searched
          </span>
          {events.length > 0 && (
            <span>{elapsed}s elapsed</span>
          )}
        </div>
      </div>

      {/* Thinking Trajectory (replaces flat Activity Log) */}
      <ThinkingTrajectory events={events} subQuestions={subQuestions} />

      {/* Cancel — only when still running */}
      {!terminated && (
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
