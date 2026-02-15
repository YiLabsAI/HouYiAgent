/**
 * BookmarkModal: overview of all bookmarked conversations and messages.
 *
 * Opens via Header bookmark icon.
 * Fetches all bookmarks from backend API and displays them grouped by date.
 * Clicking a result navigates to the conversation (and message).
 *
 * Style matches SearchModal for consistency.
 */
import React from 'react';
import { Bookmark, Star } from 'lucide-react';
import { useChatStore } from '@/stores/useChatStore';
import { CenterStage } from '@/components/CenterStage';

const API_BASE = '/api/chat';

interface BookmarkEntry {
  type: 'conversation' | 'message';
  conversation_id: string;
  title: string;
  message_id?: string | null;
  role?: string | null;
  snippet?: string;
  message_count?: number;
  model?: string;
  created_at: number;
  updated_at: number;
}

type FilterMode = 'all' | 'conversations' | 'messages';

interface BookmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
}

/** Group entries by date (MM/DD) */
function groupByDate(entries: BookmarkEntry[]): Map<string, BookmarkEntry[]> {
  const groups = new Map<string, BookmarkEntry[]>();
  for (const e of entries) {
    const d = new Date(e.created_at * 1000);
    const key = `${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(e);
  }
  return groups;
}

export const BookmarkModal: React.FC<BookmarkModalProps> = ({ isOpen, onClose }) => {
  const [bookmarks, setBookmarks] = React.useState<BookmarkEntry[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [filter, setFilter] = React.useState<FilterMode>('all');
  const loadConversation = useChatStore((s) => s.loadConversation);

  // Fetch bookmarks when modal opens
  React.useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    fetch(`${API_BASE}/bookmarks`)
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => setBookmarks(data.bookmarks || []))
      .catch(() => setBookmarks([]))
      .finally(() => setLoading(false));
  }, [isOpen]);

  const handleSelect = (entry: BookmarkEntry) => {
    loadConversation(entry.conversation_id, entry.message_id ?? undefined);
    onClose();
  };

  // Filter
  const filtered = filter === 'all'
    ? bookmarks
    : bookmarks.filter((b) => (filter === 'conversations' ? b.type === 'conversation' : b.type === 'message'));

  const convCount = bookmarks.filter((b) => b.type === 'conversation').length;
  const msgCount = bookmarks.filter((b) => b.type === 'message').length;
  const grouped = groupByDate(filtered);

  return (
    <CenterStage isOpen={isOpen} onClose={onClose} size="S" title="Bookmarks">
      {/* Summary + filter tabs */}
      <div className="flex items-center justify-between pb-2 border-b border-gray-700 -mt-1">
        <span className="text-[10px] text-gray-500">
          {convCount} conversations · {msgCount} messages
        </span>
        <div className="flex items-center gap-1 bg-gray-800 rounded-md p-0.5">
          {(['all', 'conversations', 'messages'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setFilter(mode)}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                filter === mode ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
              }`}
              type="button"
            >
              {mode === 'all' ? `All (${bookmarks.length})` : mode === 'conversations' ? `Conv (${convCount})` : `Msg (${msgCount})`}
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      <div className="mt-2">
        {loading && (
          <div className="py-3 text-[12px] text-gray-500 animate-pulse">Loading bookmarks...</div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-gray-500">
            <Bookmark size={32} className="mb-2 opacity-50" />
            <div className="text-sm">No bookmarks yet</div>
            <div className="text-xs mt-1">Star conversations or bookmark messages to see them here</div>
          </div>
        )}

        {!loading && filtered.length > 0 && (
          <div className="py-1">
            {Array.from(grouped.entries()).map(([dateKey, items]) => (
              <div key={dateKey}>
                {/* Date group header */}
                <div className="pt-3 pb-1">
                  <span className="text-[13px] font-semibold text-gray-400">{dateKey}</span>
                </div>
                {items.map((entry, i) => (
                  <button
                    key={`${entry.conversation_id}-${entry.message_id || 'conv'}-${i}`}
                    onClick={() => handleSelect(entry)}
                    className="w-full text-left px-2 py-2 -mx-2 rounded-md hover:bg-gray-700/50 transition-colors flex items-start gap-3"
                    type="button"
                  >
                    <div className="shrink-0 mt-0.5">
                      {entry.type === 'conversation' ? (
                        <Star size={14} className="text-yellow-500 fill-yellow-500" />
                      ) : (
                        <Bookmark size={14} className="text-yellow-500 fill-yellow-500" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between">
                        <div className="text-[12px] text-gray-200 font-medium truncate">
                          {entry.title}
                        </div>
                        <span className="text-[10px] text-gray-600 shrink-0 ml-2">
                          {new Date(entry.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                      {entry.type === 'conversation' ? (
                        <div className="text-[11px] text-gray-500 mt-0.5">
                          {entry.message_count} messages · {entry.model}
                        </div>
                      ) : (
                        <div className="text-[11px] text-gray-500 mt-0.5 line-clamp-2">
                          {entry.role && (
                            <span className="text-gray-600 mr-1">[{entry.role}]</span>
                          )}
                          {entry.snippet}
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </CenterStage>
  );
};
