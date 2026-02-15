/**
 * Conversation settings panel for chat mode.
 *
 * Shows quick controls:
 *   - model
 *   - temperature
 *   - deep research toggle
 *
 * Also exposes a "Full Settings..." action for the detailed settings view.
 */
import React from 'react';
import { useConsoleStore } from '../../stores/useConsoleStore';
import { useAvailableModels } from '../../hooks/useAvailableModels';
import { DEFAULT_MODEL } from '../../constants/models';
import { Settings, Thermometer, Search } from 'lucide-react';

export interface ConversationSettingsPanelProps {
  /** Called when the user clicks [Full Settings...]. */
  onOpenFullSettings?: () => void;
}

export const ConversationSettingsPanel: React.FC<ConversationSettingsPanelProps> = ({
  onOpenFullSettings,
}) => {
  const runSettings = useConsoleStore((s) => s.runSettings);
  const updateRunSettings = useConsoleStore((s) => s.updateRunSettings);
  const { modelIds } = useAvailableModels();

  // Deep Research is currently a UI toggle only.
  const [deepResearchEnabled, setDeepResearchEnabled] = React.useState(false);

  const currentModel = (runSettings as unknown as Record<string, unknown>).model as string | undefined ?? DEFAULT_MODEL;
  const temperature = runSettings.temperature ?? 0.7;

  return (
    <div className="flex flex-col gap-3 p-3 text-xs">
      {/* Model */}
      <div>
        <label className="block text-gray-400 mb-1 font-medium">Model</label>
        <select
          value={currentModel}
          onChange={(e) => updateRunSettings({ ...runSettings, model: e.target.value } as any)}
          className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-gray-200 text-xs focus:outline-none focus:border-blue-500"
        >
          {modelIds.length > 0 ? (
            modelIds.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))
          ) : (
            <option value={currentModel}>{currentModel}</option>
          )}
        </select>
      </div>

      {/* Temperature */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-gray-400 font-medium flex items-center gap-1">
            <Thermometer size={12} />
            Temperature
          </label>
          <span className="text-gray-200 font-mono">{temperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={temperature}
          onChange={(e) => updateRunSettings({ temperature: parseFloat(e.target.value) })}
          className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
          <span>Precise</span>
          <span>Creative</span>
        </div>
      </div>

      {/* Deep Research Toggle */}
      <div className="bg-gray-700/50 rounded p-2">
        <div className="flex items-center justify-between">
          <label className="text-gray-300 font-medium flex items-center gap-1.5">
            <Search size={12} className="text-blue-400" />
            Deep Research
          </label>
          <button
            onClick={() => setDeepResearchEnabled(!deepResearchEnabled)}
            className={`relative w-8 h-4 rounded-full transition-colors ${
              deepResearchEnabled ? 'bg-blue-500' : 'bg-gray-600'
            }`}
            role="switch"
            aria-checked={deepResearchEnabled}
            data-testid="deep-research-toggle"
          >
            <span
              className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${
                deepResearchEnabled ? 'translate-x-4' : 'translate-x-0.5'
              }`}
            />
          </button>
        </div>
        <p className="text-[10px] text-gray-500 mt-1">
          {deepResearchEnabled
            ? 'Deep Research mode active — multi-step search & synthesis'
            : 'Enable to perform thorough research before responding'}
        </p>
      </div>

      {/* Quick Stats */}
      <div className="space-y-1 text-gray-400">
        <div className="flex justify-between">
          <span>Tool Calls</span>
          <span className="text-gray-300">{runSettings.enable_tool_calls ? 'Enabled' : 'Disabled'}</span>
        </div>
        <div className="flex justify-between">
          <span>Max Tools</span>
          <span className="text-gray-300">{runSettings.max_tool_calls}</span>
        </div>
        {runSettings.web_search_provider && (
          <div className="flex justify-between">
            <span>Web Search</span>
            <span className="text-gray-300">{runSettings.web_search_provider}</span>
          </div>
        )}
      </div>

      {/* Action bar */}
      <div className="pt-1 border-t border-gray-700">
        <button
          onClick={onOpenFullSettings}
          className="w-full flex items-center justify-center gap-1 px-2 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-200 transition-colors"
          title="Open full conversation settings in Center Stage"
        >
          <Settings size={12} />
          Full Settings...
        </button>
      </div>
    </div>
  );
};
