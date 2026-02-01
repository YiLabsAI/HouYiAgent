import React from 'react';

interface ExecutionControlsProps {
  currentExecution: any;
  viewMode?: 'live' | 'checkpoint';
  isStarting: boolean;
  hasNodes: boolean;
  onStartExecution: () => void;
  onPauseExecution: () => void;
  onResumeExecution: () => void;
  onAbortExecution: () => void;
  runSettingsSummary: string;
  onOpenRunSettings: () => void;
}

export const ExecutionControls: React.FC<ExecutionControlsProps> = ({
  currentExecution,
  viewMode,
  isStarting,
  hasNodes,
  onStartExecution,
  onPauseExecution,
  onResumeExecution,
  onAbortExecution,
  runSettingsSummary,
  onOpenRunSettings,
}) => {
  const inCheckpointView = viewMode === 'checkpoint';
  const isRunning = currentExecution?.status === 'running';
  const isPaused = currentExecution?.status === 'paused';
  const canStart =
    !currentExecution ||
    currentExecution.status === 'completed' ||
    currentExecution.status === 'failed' ||
    currentExecution.status === 'aborted';

  return (
    <div className="p-3 border-b border-gray-700">
      <h3 className="text-xs font-semibold text-gray-400 mb-2">Execution Control</h3>

      <div className="space-y-2">
        {canStart ? (
          <button
            onClick={onStartExecution}
            disabled={isStarting || inCheckpointView || !hasNodes}
            className="w-full h-10 px-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
            title={
              inCheckpointView
                ? "Exit checkpoint view to start new execution"
                : !hasNodes
                  ? "Add at least one node to start"
                  : "Start execution"
            }
          >
            <span>{isStarting ? '⏳' : '▶️'}</span>
            <span className="min-w-0 overflow-hidden text-ellipsis">
              {isStarting ? 'Starting...' : 'Start Execution'}
            </span>
          </button>
        ) : isRunning ? (
          <div className="space-y-2">
            <button
              onClick={onPauseExecution}
              disabled={inCheckpointView}
              className="w-full h-10 px-3 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
              title={inCheckpointView ? "Exit checkpoint view to pause" : "Pause execution"}
            >
              <span>⏸️</span>
              <span className="min-w-0 overflow-hidden text-ellipsis">Pause</span>
            </button>
            <button
              onClick={onAbortExecution}
              disabled={inCheckpointView}
              className="w-full h-10 px-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
              title={inCheckpointView ? "Exit checkpoint view to stop" : "Stop execution"}
            >
              <span>⏹️</span>
              <span className="min-w-0 overflow-hidden text-ellipsis">Stop</span>
            </button>
          </div>
        ) : isPaused ? (
          <div className="space-y-2">
            <button
              onClick={onResumeExecution}
              disabled={inCheckpointView}
              className="w-full h-10 px-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
              title={inCheckpointView ? "Exit checkpoint view to resume" : "Resume execution"}
            >
              <span>▶️</span>
              <span className="min-w-0 overflow-hidden text-ellipsis">Resume</span>
            </button>
            <button
              onClick={onAbortExecution}
              disabled={inCheckpointView}
              className="w-full h-10 px-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
              title={inCheckpointView ? "Exit checkpoint view to stop" : "Stop execution"}
            >
              <span>⏹️</span>
              <span className="min-w-0 overflow-hidden text-ellipsis">Stop</span>
            </button>
          </div>
        ) : null}
      </div>

      <div className="mt-3 rounded border border-gray-700 bg-gray-900/60 px-2 py-2">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">
              Run Settings
            </div>
            <div className="text-[11px] text-gray-300">{runSettingsSummary}</div>
          </div>
          <button
            type="button"
            onClick={onOpenRunSettings}
            className="rounded bg-gray-700 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-600"
          >
            Edit
          </button>
        </div>
      </div>
    </div>
  );
};
