import React from 'react';
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

export const ProgressPanel: React.FC<Props> = ({
  progress,
  events,
  subQuestions,
  onCancel,
  error,
}) => {
  const totalSteps = progress?.total_steps || 0;
  const completedSteps = progress?.completed_steps || 0;
  const pct = totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0;
  const allSearchDone = totalSteps > 0 && completedSteps >= totalSteps;
  const terminated = !!error;

  const barColor = terminated
    ? 'bg-red-500'
    : allSearchDone
      ? 'bg-green-500'
      : 'bg-purple-500';
  const stepLabel = terminated
    ? 'Stopped'
    : allSearchDone
      ? 'Writing report...'
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
          <span>
            {completedSteps} / {totalSteps} search steps
          </span>
          {progress?.elapsed_seconds != null && (
            <span>{Math.round(progress.elapsed_seconds)}s elapsed</span>
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
