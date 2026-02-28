import React from 'react';
import {
  CheckCircle,
  XCircle,
  Circle,
  Loader2,
  AlertTriangle,
} from 'lucide-react';

import type { DryRunResultData } from '../../../LeftSidebar/useSkillsLogic';
import type { PipelineStage, StageStatus } from './computeStages';

const statusColor: Record<StageStatus, string> = {
  pass: 'border-green-500/50 bg-green-500/5',
  fail: 'border-red-500/50 bg-red-500/5',
  warn: 'border-yellow-500/50 bg-yellow-500/5',
  running: 'border-blue-500/50 bg-blue-500/5',
  skip: 'border-gray-700/50 bg-gray-800/30',
  pending: 'border-gray-700/50 bg-gray-800/30',
};

const statusText: Record<StageStatus, string> = {
  pass: 'text-green-400',
  fail: 'text-red-400',
  warn: 'text-yellow-400',
  running: 'text-blue-400',
  skip: 'text-gray-600',
  pending: 'text-gray-600',
};

const StageIcon: React.FC<{ status: StageStatus }> = ({ status }) => {
  switch (status) {
    case 'pass':
      return <CheckCircle size={16} className="text-green-400" />;
    case 'fail':
      return <XCircle size={16} className="text-red-400" />;
    case 'warn':
      return <AlertTriangle size={16} className="text-yellow-400" />;
    case 'running':
      return <Loader2 size={16} className="text-blue-400 animate-spin" />;
    case 'skip':
      return <Circle size={16} className="text-gray-600" />;
    case 'pending':
    default:
      return <Circle size={16} className="text-gray-600" />;
  }
};

const PipelineStageRow: React.FC<{
  stage: PipelineStage;
  isLast: boolean;
  revealed: boolean;
}> = ({ stage, isLast, revealed }) => {
  const active = revealed && stage.status !== 'pending';

  return (
    <div
      className={`relative flex gap-3 ${revealed ? 'animate-stageReveal' : 'opacity-40'}`}
      data-testid={`dry-run-stage-${stage.id}`}
    >
      <div className="flex flex-col items-center shrink-0 w-5">
        <div className={`mt-0.5 transition-all duration-300 ${active ? '' : 'grayscale opacity-50'}`}>
          <StageIcon status={revealed ? stage.status : 'pending'} />
        </div>
        {!isLast && (
          <div className={`w-px flex-1 min-h-[16px] mt-1 transition-colors duration-300 ${
            active ? 'bg-gray-600' : 'bg-gray-700/40'
          }`}
          />
        )}
      </div>

      <div className={`flex-1 pb-4 ${isLast ? 'pb-0' : ''}`}>
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
              active ? statusColor[stage.status] : 'bg-gray-800/30 border border-gray-700/50'
            } border`}
          >
            {stage.number}
          </span>
          <span className={`text-xs font-medium ${active ? 'text-gray-200' : 'text-gray-500'}`}>
            {stage.label}
          </span>
          {active && (
            <span className={`text-[10px] font-medium uppercase tracking-wider ${statusText[stage.status]}`}>
              {stage.status === 'pass' ? 'PASS'
                : stage.status === 'fail' ? 'FAIL'
                  : stage.status === 'warn' ? 'WARN'
                    : stage.status === 'running' ? 'RUNNING'
                      : stage.status === 'skip' ? 'SKIPPED' : ''}
            </span>
          )}
        </div>
        <div className={`text-[11px] ${active ? 'text-gray-400' : 'text-gray-600'}`}>
          {revealed ? stage.summary : 'Waiting...'}
        </div>
        {revealed && stage.details && (
          <div className="mt-1.5">
            {stage.details}
          </div>
        )}
      </div>
    </div>
  );
};

interface PipelineStagePanelProps {
  showPipeline: boolean;
  dryRunResult: DryRunResultData | null;
  isExecuting: boolean;
  pipelineStages: PipelineStage[];
  revealedCount: number;
}

export const PipelineStagePanel: React.FC<PipelineStagePanelProps> = ({
  showPipeline,
  dryRunResult,
  isExecuting,
  pipelineStages,
  revealedCount,
}) => {
  if (!showPipeline) {
    return null;
  }

  return (
    <div data-testid="dry-run-result-panel">
      <div className="flex items-center gap-2 mb-3 pt-1 border-t border-gray-700/50">
        <div
          className={`w-2 h-2 rounded-full ${
            !dryRunResult ? 'bg-blue-400 animate-pulse'
              : dryRunResult.valid ? 'bg-green-400' : 'bg-red-400'
          }`}
        />
        <span className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
          Verification Pipeline
        </span>
        {dryRunResult && (
          <span className={`ml-auto text-[11px] font-semibold ${
            dryRunResult.valid ? 'text-green-400' : 'text-red-400'
          }`}
          >
            {dryRunResult.valid ? 'ALL PASSED' : 'FAILED'}
          </span>
        )}
        {!dryRunResult && isExecuting && (
          <span className="ml-auto text-[11px] text-blue-400">
            Running...
          </span>
        )}
      </div>

      <div className="pl-1">
        {pipelineStages.map((stage, i) => (
          <PipelineStageRow
            key={stage.id}
            stage={stage}
            isLast={i === pipelineStages.length - 1}
            revealed={i < revealedCount}
          />
        ))}
      </div>
    </div>
  );
};
