import React from 'react';
import { BookOpen, FolderTree, MessageSquare, Puzzle, Settings } from 'lucide-react';

interface ActivityBarProps {
  activeTab: 'workflow' | 'chat' | 'knowledge' | 'skills';
  onSelectTab: (tab: 'workflow' | 'chat' | 'knowledge' | 'skills') => void;
  onOpenSettings: () => void;
}

export const ActivityBar: React.FC<ActivityBarProps> = ({
  activeTab,
  onSelectTab,
  onOpenSettings,
}) => {
  return (
    <div className="bg-gray-900 border-r border-gray-800 flex flex-col w-12 shrink-0">
      <div className="flex-1 flex flex-col py-2 gap-1">
        <button
          onClick={() => onSelectTab('workflow')}
          className={`mx-1 h-10 rounded text-xs font-semibold transition-colors ${
            activeTab === 'workflow'
              ? 'bg-gray-700 text-white'
              : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
          }`}
          title="Workflow"
          aria-label="Workflow"
          type="button"
        >
          <div className="flex items-center justify-center">
            <FolderTree size={18} />
          </div>
        </button>
        <button
          onClick={() => onSelectTab('chat')}
          className={`mx-1 h-10 rounded text-xs font-semibold transition-colors ${
            activeTab === 'chat'
              ? 'bg-gray-700 text-white'
              : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
          }`}
          title="Chat"
          aria-label="Chat"
          type="button"
        >
          <div className="flex items-center justify-center">
            <MessageSquare size={18} />
          </div>
        </button>
        <button
          onClick={() => onSelectTab('knowledge')}
          className={`mx-1 h-10 rounded text-xs font-semibold transition-colors ${
            activeTab === 'knowledge'
              ? 'bg-gray-700 text-white'
              : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
          }`}
          title="Knowledge"
          aria-label="Knowledge"
          type="button"
        >
          <div className="flex items-center justify-center">
            <BookOpen size={18} />
          </div>
        </button>
        <button
          onClick={() => onSelectTab('skills')}
          className={`mx-1 h-10 rounded text-xs font-semibold transition-colors ${
            activeTab === 'skills'
              ? 'bg-gray-700 text-white'
              : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
          }`}
          title="Skills"
          aria-label="Skills"
          type="button"
        >
          <div className="flex items-center justify-center">
            <Puzzle size={18} />
          </div>
        </button>
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
