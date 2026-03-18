import React from 'react';
import { TraceSection } from './TraceSection';
import { formatDuration } from './TraceDetailUtils';

interface PipelineStage {
  name: string;
  count: number;
  totalMs: number;
}

interface TracePipelineStagesProps {
  stages: PipelineStage[];
}

export const TracePipelineStages: React.FC<TracePipelineStagesProps> = ({ stages }) => {
  if (stages.length === 0) return null;

  return (
    <TraceSection title="Pipeline stages">
      <div className="space-y-1 text-[11px] text-gray-200">
        {stages.map((stage) => (
          <div key={stage.name} className="flex items-center justify-between gap-2 rounded border border-emerald-500/20 bg-emerald-500/5 px-2 py-1.5">
            <span className="truncate">{stage.name} ({stage.count}x)</span>
            <span className="shrink-0 text-emerald-200">{formatDuration(stage.totalMs)}</span>
          </div>
        ))}
      </div>
    </TraceSection>
  );
};
