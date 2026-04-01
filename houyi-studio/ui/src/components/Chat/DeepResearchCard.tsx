/**
 * DeepResearchCard — inline chat card for deep_research skill results.
 *
 * Shown in the Chat timeline when the assistant invokes the deep_research
 * skill. Displays a summary + link to open full Workspace.
 */
import React from 'react';
import { Search, ExternalLink, FileText, Loader2 } from 'lucide-react';

interface Props {
  query: string;
  sessionId?: string;
  summary?: string;
  status?: 'pending' | 'running' | 'completed' | 'failed';
  onOpenWorkspace?: (sessionId: string) => void;
}

export const DeepResearchCard: React.FC<Props> = ({
  query,
  sessionId,
  summary,
  status = 'pending',
  onOpenWorkspace,
}) => {
  const statusLabel: Record<string, { text: string; color: string }> = {
    pending: { text: 'Queued', color: 'text-yellow-400' },
    running: { text: 'Researching...', color: 'text-blue-400' },
    completed: { text: 'Complete', color: 'text-green-400' },
    failed: { text: 'Failed', color: 'text-red-400' },
  };

  const s = statusLabel[status] || statusLabel.pending;

  return (
    <div className="my-2 rounded-xl border border-purple-700/40 bg-purple-900/10 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-purple-700/20">
        {status === 'running' ? (
          <Loader2 size={14} className="text-purple-400 animate-spin" />
        ) : (
          <Search size={14} className="text-purple-400" />
        )}
        <span className="text-xs font-medium text-purple-300">Deep Research</span>
        <span className={`text-[10px] ml-auto ${s.color}`}>{s.text}</span>
      </div>

      <div className="px-4 py-3 space-y-2">
        <p className="text-xs text-gray-400">
          <span className="text-gray-500">Query:</span> {query}
        </p>

        {summary && (
          <div className="text-xs text-gray-300 leading-relaxed border-t border-purple-700/20 pt-2">
            <FileText size={12} className="inline mr-1 text-gray-500" />
            {summary}
          </div>
        )}
      </div>

      {sessionId && onOpenWorkspace && status === 'completed' && (
        <button
          type="button"
          onClick={() => onOpenWorkspace(sessionId)}
          className="w-full flex items-center justify-center gap-1.5 px-4 py-2 text-xs text-purple-300 hover:text-purple-200 bg-purple-900/20 hover:bg-purple-900/30 border-t border-purple-700/20 transition-colors"
        >
          <ExternalLink size={12} />
          Open in Workspace
        </button>
      )}
    </div>
  );
};
