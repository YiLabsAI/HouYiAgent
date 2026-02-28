import React from 'react';
import { Play, Zap } from 'lucide-react';

import { LIVE_LLM_PROVIDER_OPTIONS } from './liveLlmOptions';

interface DryRunExecuteBarProps {
  liveMode: boolean;
  isExecuting: boolean;
  selectedTool: string;
  liveLlmProvider: string;
  liveLlmModel: string;
  liveModelOptions: string[];
  onToggleLiveMode: (enabled: boolean) => void;
  onChangeLiveProvider: (provider: string) => void;
  onChangeLiveModel: (model: string) => void;
  onExecute: () => void;
}

export const DryRunExecuteBar: React.FC<DryRunExecuteBarProps> = ({
  liveMode,
  isExecuting,
  selectedTool,
  liveLlmProvider,
  liveLlmModel,
  liveModelOptions,
  onToggleLiveMode,
  onChangeLiveProvider,
  onChangeLiveModel,
  onExecute,
}) => (
  <div className="flex flex-wrap items-center gap-3">
    <label
      className="flex items-center gap-1.5 cursor-pointer select-none shrink-0"
      title="When enabled, dry-run will send the input to an LLM to verify the skill produces a valid tool call."
    >
      <input
        type="checkbox"
        checked={liveMode}
        onChange={(e) => onToggleLiveMode(e.target.checked)}
        className="w-3.5 h-3.5 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 cursor-pointer"
        data-testid="dry-run-live-toggle"
      />
      <Zap size={11} className={liveMode ? 'text-amber-400' : 'text-gray-600'} />
      <span className={`text-[11px] ${liveMode ? 'text-amber-300' : 'text-gray-500'}`}>
        Live
      </span>
    </label>

    {liveMode && (
      <div className="flex items-center gap-2 shrink-0" data-testid="dry-run-live-llm-config">
        <select
          value={liveLlmProvider}
          onChange={(e) => onChangeLiveProvider(e.target.value)}
          className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-200"
          data-testid="dry-run-live-provider"
        >
          {LIVE_LLM_PROVIDER_OPTIONS.map((option) => (
            <option key={`live-provider-${option.value || 'default'}`} value={option.value}>{option.label}</option>
          ))}
        </select>
        <select
          value={liveLlmModel}
          onChange={(e) => onChangeLiveModel(e.target.value)}
          className="w-44 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-200"
          data-testid="dry-run-live-model"
          disabled={!liveLlmProvider || liveModelOptions.length === 0}
        >
          <option value="">model: default</option>
          {liveModelOptions.map((modelName) => (
            <option key={`live-model-${liveLlmProvider}-${modelName}`} value={modelName}>{modelName}</option>
          ))}
        </select>
      </div>
    )}

    <button
      type="button"
      onClick={onExecute}
      disabled={isExecuting || !selectedTool}
      className={`flex-1 min-w-[220px] whitespace-nowrap flex items-center justify-center gap-2 px-4 py-2 text-white text-sm font-medium rounded-lg transition-colors ${
        liveMode
          ? 'bg-amber-600 hover:bg-amber-500 disabled:bg-gray-700 disabled:text-gray-500'
          : 'bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500'
      }`}
      data-testid="dry-run-execute"
    >
      {liveMode ? <Zap size={14} /> : <Play size={14} />}
      {isExecuting ? 'Verifying...' : liveMode ? 'Execute Live Dry-run' : 'Execute Dry-run'}
    </button>
  </div>
);
