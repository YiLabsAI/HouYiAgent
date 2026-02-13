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
import { Bookmark, Star, ArrowLeft } from 'lucide-react';
import { useChatStore } from '@/stores/useChatStore';

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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  };

  if (!isOpen) return null;

  // Filter
  const filtered = filter === 'all'
    ? bookmarks
    : bookmarks.filter((b) => (filter === 'conversations' ? b.type === 'conversation' : b.type === 'message'));

  const convCount = bookmarks.filter((b) => b.type === 'conversation').length;
  const msgCount = bookmarks.filter((b) => b.type === 'message').length;
  const grouped = groupByDate(filtered);

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 z-50" onClick={onClose} />

      {/* Modal */}
      <div
        className="fixed top-[10%] left-1/2 -translate-x-1/2 w-[640px] max-h-[75vh] bg-gray-900 border border-gray-700 rounded-xl shadow-2xl z-50 flex flex-col"
        onKeyDown={handleKeyDown}
        tabIndex={-1}
        ref={(el) => el?.focus()}
      >
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-700">
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300 transition-colors"
            type="button"
          >
            <ArrowLeft size={16} />
          </button>
          <Bookmark size={16} className="text-yellow-500 shrink-0" />
          <span className="flex-1 text-[13px] text-gray-200 font-medium">Bookmarks</span>
          <span className="text-[10px] text-gray-500">
            {convCount} conversations · {msgCount} messages
          </span>
          <kbd className="text-[10px] text-gray-600 bg-gray-800 px-1.5 py-0.5 rounded border border-gray-700">
            ESC
          </kbd>
        </div>

        {/* Filter tabs */}
        <div className="flex items-center gap-1 px-4 py-1.5 border-b border-gray-800">
          {(['all', 'conversations', 'messages'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setFilter(mode)}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                filter === mode ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
              }`}
              type="button"
            >
              {mode === 'all' ? `All (${bookmarks.length})` : mode === 'conversations' ? `Conversations (${convCount})` : `Messages (${msgCount})`}
            </button>
          ))}
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className="px-4 py-3 text-[12px] text-gray-500 animate-pulse">Loading bookmarks...</div>
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
                  <div className="px-4 pt-3 pb-1">
                    <span className="text-[13px] font-semibold text-gray-400">{dateKey}</span>
                  </div>
                  {items.map((entry, i) => (
                    <button
                      key={`${entry.conversation_id}-${entry.message_id || 'conv'}-${i}`}
                      onClick={() => handleSelect(entry)}
                      className="w-full text-left px-4 py-2 hover:bg-gray-800 transition-colors flex items-start gap-3"
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
      </div>
    </>
  );
};
