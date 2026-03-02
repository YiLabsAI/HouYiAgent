import React from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';
import type { PrimaryMode } from '../stores/useConsoleStore';
import { useThemeStore, THEMES } from '../stores/useThemeStore';
import { ToolStatistics } from './ToolStatistics';
import { History, Flag, Search, Bookmark, Palette } from 'lucide-react';

// NOTE(core): Title Bar "mode" is a lightweight UI toggle (top-level workspace view switch).
interface HeaderProps {
  onOpenBottomPanel?: (tab: 'observability' | 'checkpoints' | 'context' | 'logs') => void;
  primaryMode: PrimaryMode;
  onSetPrimaryMode: (mode: PrimaryMode) => void;
  onOpenGlobalSettings?: () => void;
  onOpenSearch?: () => void;
  onOpenBookmarks?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenBottomPanel,
  primaryMode,
  onSetPrimaryMode,
  onOpenSearch,
  onOpenBookmarks,
}) => {
  const { connectionStatus } = useConsoleStore();
  const { theme, setTheme } = useThemeStore();
  const [themeMenuOpen, setThemeMenuOpen] = React.useState(false);
  const themeMenuRef = React.useRef<HTMLDivElement>(null);

  // Close theme menu on outside click
  React.useEffect(() => {
    if (!themeMenuOpen) return;
    const handler = (e: MouseEvent) => {
      if (themeMenuRef.current && !themeMenuRef.current.contains(e.target as Node)) {
        setThemeMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [themeMenuOpen]);

  return (
    <header className="h-14 bg-gray-900 border-b border-gray-700 flex items-center justify-between px-6">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-lg">H</span>
          </div>
          <h1 className="text-lg font-semibold text-gray-50 tracking-tight">HouYi</h1>
        </div>

        <div className="flex items-center bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => onSetPrimaryMode('graph')}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
              primaryMode === 'graph'
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
            onClick={() => onSetPrimaryMode('chat')}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
              primaryMode === 'chat'
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
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Chat mode: search + bookmarks (settings unified to ActivityBar gear) */}
        {primaryMode === 'chat' && (
          <>
            <button
              onClick={onOpenSearch}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
              title="Search conversations (Cmd+K)"
              type="button"
            >
              <Search size={18} />
            </button>
            <button
              onClick={onOpenBookmarks}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
              title="Bookmarks"
              type="button"
            >
              <Bookmark size={18} />
            </button>
          </>
        )}

        <ToolStatistics />

        {/* Graph mode: Timeline / Checkpoints */}
        {primaryMode === 'graph' && (
          <>
            <button
              type="button"
              onClick={() => onOpenBottomPanel?.('observability')}
              disabled={!onOpenBottomPanel}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors disabled:cursor-not-allowed disabled:opacity-50"
              title="Timeline"
              aria-label="Timeline"
            >
              <History size={18} />
            </button>
            <button
              type="button"
              onClick={() => onOpenBottomPanel?.('checkpoints')}
              disabled={!onOpenBottomPanel}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors disabled:cursor-not-allowed disabled:opacity-50"
              title="Checkpoints"
              aria-label="Checkpoints"
            >
              <Flag size={18} />
            </button>
          </>
        )}

        {/* Live/Offline indicator — always penultimate, before theme */}
        <div className="flex items-center gap-1.5 px-2 py-1.5">
          <div className={`w-2 h-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }`} />
          <span className="text-xs font-medium text-gray-500">
            {connectionStatus === 'connected' ? 'Live' : 'Offline'}
          </span>
        </div>

        {/* Theme switcher — always last for consistency between Graph and Chat modes */}
        <div className="relative" ref={themeMenuRef}>
          <button
            onClick={() => setThemeMenuOpen(!themeMenuOpen)}
            className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
            title="Switch theme"
            type="button"
          >
            <Palette size={18} />
          </button>
          {themeMenuOpen && (
            <div className="absolute right-0 top-10 z-50 w-40 bg-gray-900 border border-gray-700 rounded-lg shadow-xl overflow-hidden">
              {THEMES.map((t) => (
                <button
                  key={t.id}
                  onClick={() => { setTheme(t.id); setThemeMenuOpen(false); }}
                  className={`w-full text-left px-3 py-2 text-[12px] flex items-center justify-between transition-colors ${
                    theme === t.id
                      ? 'bg-gray-800 text-gray-50'
                      : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                  }`}
                  type="button"
                >
                  <span>{t.label}</span>
                  {theme === t.id && <span className="text-blue-400">✓</span>}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
