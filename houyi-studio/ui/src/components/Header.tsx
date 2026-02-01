import React from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';
import { ToolStatistics } from './ToolStatistics';
import { History, Flag } from 'lucide-react';

type ViewMode = 'graph' | 'chat';

// NOTE(core): Header "mode" is a lightweight UI toggle (top-level workspace view switch).
// Agent / Observability will be wired up in later phases; we keep the entry points visible but disabled for now.

interface HeaderProps {
  onOpenBottomPanel?: (tab: 'timeline' | 'checkpoints') => void;
  onSelectLeftTab?: (tab: 'workflow' | 'chat' | 'knowledge' | 'skills') => void;
  activeLeftTab?: 'workflow' | 'chat' | 'knowledge' | 'skills';
}

export const Header: React.FC<HeaderProps> = ({
  onOpenBottomPanel,
  onSelectLeftTab,
  activeLeftTab,
}) => {
  const { connectionStatus, setRunSettingsOpen } = useConsoleStore();
  const [theme, setTheme] = React.useState<'light' | 'dark'>('dark');

  // NOTE(core): Header mode is derived from the primary ActivityBar view to avoid state drift.
  const viewMode: ViewMode = activeLeftTab === 'chat' ? 'chat' : 'graph';

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  return (
    <header className="h-14 bg-gray-900 border-b border-gray-700 flex items-center justify-between px-6">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-lg">H</span>
          </div>
          <h1 className="text-lg font-semibold text-white tracking-tight">HouYi</h1>
        </div>

        <div className="flex items-center bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => {
              // NOTE(core): Header mode toggle is a shortcut into primary ActivityBar views.
              // Graph == Workflow authoring surface.
              onSelectLeftTab?.('workflow');
            }}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
              viewMode === 'graph'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <span className="flex items-center gap-2">
              <span>🔀</span>
              <span>Graph</span>
            </span>
          </button>
          <button
            onClick={() => {
              // NOTE(core): Chat is a high-frequency surface; keep it reachable from both Header and ActivityBar.
              onSelectLeftTab?.('chat');
            }}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
              viewMode === 'chat'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <span className="flex items-center gap-2">
              <span>💬</span>
              <span>Chat</span>
            </span>
          </button>

          <button
            type="button"
            disabled
            className="px-4 py-1.5 rounded-md text-sm font-medium transition-all text-gray-500 cursor-not-allowed"
            title="Agent (Coming soon): for now, use Chat + Workflow; multi-agent runtime will be available here later"
            aria-label="Agent (Coming soon)"
          >
            <span className="flex items-center gap-2">
              <span>🤖</span>
              <span>Agent</span>
            </span>
          </button>

          <button
            type="button"
            disabled
            className="px-4 py-1.5 rounded-md text-sm font-medium transition-all text-gray-500 cursor-not-allowed"
            title="Observability (Coming soon): for now, use BottomPanel Logs / Verification / Context as the temporary entry"
            aria-label="Observability (Coming soon)"
          >
            <span className="flex items-center gap-2">
              <span>📊</span>
              <span>Obs</span>
            </span>
          </button>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <ToolStatistics />

        <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg">
          <div className={`w-2 h-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }`} />
          <span className="text-xs font-medium text-gray-400">
            {connectionStatus === 'connected' ? 'Live' : 'Offline'}
          </span>
        </div>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
          title="Toggle theme"
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>

        <div className="flex items-center gap-1 rounded-lg bg-gray-800 p-1 border border-gray-700/50">
          <button
            type="button"
            onClick={() => onOpenBottomPanel?.('timeline')}
            disabled={!onOpenBottomPanel}
            className="h-9 w-9 rounded-md hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors disabled:cursor-not-allowed disabled:opacity-50 flex items-center justify-center"
            title="Timeline"
            aria-label="Timeline"
          >
            <History size={18} />
          </button>

          <button
            type="button"
            onClick={() => onOpenBottomPanel?.('checkpoints')}
            disabled={!onOpenBottomPanel}
            className="h-9 w-9 rounded-md hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors disabled:cursor-not-allowed disabled:opacity-50 flex items-center justify-center"
            title="Checkpoints"
            aria-label="Checkpoints"
          >
            <Flag size={18} />
          </button>

          <div className="mx-1 h-5 w-px bg-gray-700" />

          <button
            onClick={() => setRunSettingsOpen(true)}
            className="h-9 w-9 rounded-md hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors flex items-center justify-center"
            title="Run settings"
            aria-label="Run settings"
            type="button"
          >
            ⚙️
          </button>
        </div>
      </div>
    </header>
  );
};
