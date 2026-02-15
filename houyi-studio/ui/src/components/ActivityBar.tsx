/**
 * Activity Bar — vertical tab strip for Primary Sidebar content.
 */
import React from 'react';
import { BookOpen, FolderTree, MessageSquare, Puzzle, Settings } from 'lucide-react';
import type { PrimaryMode, SidebarTab } from '../stores/useConsoleStore';

interface ActivityBarProps {
  primaryMode: PrimaryMode;
  sidebarTab: SidebarTab;
  onSelectTab: (tab: SidebarTab) => void;
  onOpenSettings: () => void;
}

export const ActivityBar: React.FC<ActivityBarProps> = ({
  primaryMode,
  sidebarTab,
  onSelectTab,
  onOpenSettings,
}) => {
  // The first tab depends on primaryMode (workflow vs conversations).
  const modeTab: SidebarTab = primaryMode === 'graph' ? 'workflow' : 'conversations';
  const modeIcon = primaryMode === 'graph' ? <FolderTree size={18} /> : <MessageSquare size={18} />;
  const modeLabel = primaryMode === 'graph' ? 'Workflow' : 'Conversations';

  const tabButton = (tab: SidebarTab, icon: React.ReactNode, label: string) => (
    <button
      onClick={() => onSelectTab(tab)}
      className={`mx-1 h-10 rounded text-xs font-semibold transition-colors ${
        sidebarTab === tab
          ? 'bg-gray-700 text-gray-50'
          : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
      }`}
      title={label}
      aria-label={label}
      type="button"
    >
      <div className="flex items-center justify-center">
        {icon}
      </div>
    </button>
  );

  return (
    <div className="bg-gray-900 border-r border-gray-800 flex flex-col w-12 shrink-0">
      <div className="flex-1 flex flex-col py-2 gap-1">
        {tabButton(modeTab, modeIcon, modeLabel)}
        {tabButton('knowledge', <BookOpen size={18} />, 'Knowledge')}
        {tabButton('skills', <Puzzle size={18} />, 'Skills')}
      </div>

      <div className="pb-2">
        <button
          onClick={onOpenSettings}
          className="mx-1 h-10 w-10 rounded text-xs font-semibold transition-colors text-gray-400 hover:bg-gray-800 hover:text-gray-200"
          title="Settings"
          aria-label="Settings"
          type="button"
        >
          <div className="flex items-center justify-center">
            <Settings size={18} />
          </div>
        </button>
      </div>
    </div>
  );
};
