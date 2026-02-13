/**
 * ConversationRail: conversation list for the left sidebar Chat tab.
 *
 * Shows conversation summaries with create/delete actions.
 * Highlights the active conversation.
 */
import React from 'react';
import { Plus, Trash2, MessageSquare, MoreHorizontal, Settings, Star } from 'lucide-react';
import type { ConversationSummary } from '@/types/chat';
import { ConfirmModal } from '../ConfirmModal';

interface ConversationRailProps {
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  isLoading: boolean;
  onSelect: (conversationId: string) => void;
  onCreate: () => void;
  onDelete: (conversationId: string) => void;
  onOpenSettings: (conversationId: string) => void;
  onToggleBookmark?: (conversationId: string, bookmarked: boolean) => void;
}

export const ConversationRail: React.FC<ConversationRailProps> = ({
  conversations,
  activeConversationId,
  isLoading,
  onSelect,
  onCreate,
  onDelete,
  onOpenSettings,
  onToggleBookmark,
}) => {
  const [menuOpenId, setMenuOpenId] = React.useState<string | null>(null);
  const [menuPosition, setMenuPosition] = React.useState<{ top: number; left: number } | null>(null);
  const [deleteTargetId, setDeleteTargetId] = React.useState<string | null>(null);
  const [isCreating, setIsCreating] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement | null>(null);

  const handleCreate = React.useCallback(async () => {
    if (isCreating) return;
    setIsCreating(true);
    try {
      await onCreate();
    } finally {
      // Debounce: prevent rapid clicks for 300ms after creation completes
      setTimeout(() => setIsCreating(false), 300);
    }
  }, [isCreating, onCreate]);

  // Close menu on outside click
  React.useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuOpenId && menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenId(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [menuOpenId]);
  return (
    <div className="flex flex-col h-full">
      {/* Header with New Chat button */}
      <div className="px-3 py-2 flex items-center justify-between">
        <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">
          Conversations
        </span>
        <button
          onClick={handleCreate}
          disabled={isCreating}
          className={`p-1 hover:bg-gray-700 rounded transition-colors ${
            isCreating ? 'text-gray-600 cursor-not-allowed' : 'text-gray-400 hover:text-gray-200'
          }`}
          title="New conversation"
          type="button"
          data-testid="new-conversation-btn"
        >
          <Plus size={14} />
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-1">
        {isLoading && conversations.length === 0 && (
          <div className="px-3 py-4 text-[11px] text-gray-500 text-center">Loading...</div>
        )}

        {!isLoading && conversations.length === 0 && (
          <div className="px-3 py-4 text-center">
            <MessageSquare size={20} className="mx-auto mb-2 text-gray-600" />
            <p className="text-[11px] text-gray-500">No conversations yet</p>
            <button
              onClick={onCreate}
              className="mt-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-[11px] text-gray-200 transition-colors"
              type="button"
            >
              Start a chat
            </button>
          </div>
        )}

        {/* Sort: bookmarked first, then by updated_at desc */}
        {[...conversations].sort((a, b) => {
          if (a.bookmarked && !b.bookmarked) return -1;
          if (!a.bookmarked && b.bookmarked) return 1;
          return 0;
        }).map((conv) => {
          const isActive = conv.conversation_id === activeConversationId;
          return (
            <div
              key={conv.conversation_id}
              onClick={() => onSelect(conv.conversation_id)}
              className={`group flex items-center gap-2 px-2 py-1.5 mx-1 rounded cursor-pointer transition-colors ${
                isActive
                  ? 'bg-gray-700 text-gray-100'
                  : 'text-gray-300 hover:bg-gray-700/50'
              }`}
            >
              {conv.bookmarked ? (
                <Star size={12} className="shrink-0 text-yellow-500 fill-yellow-500" />
              ) : (
                <MessageSquare size={12} className="shrink-0 text-gray-500" />
              )}
              <div className="flex-1 min-w-0">
                <div className="text-[12px] truncate">{conv.title}</div>
                <div className="text-[10px] text-gray-500">
                  {conv.message_count} messages
                </div>
              </div>
              <div className="relative shrink-0">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (menuOpenId === conv.conversation_id) {
                      setMenuOpenId(null);
                      setMenuPosition(null);
                    } else {
                      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
                      const menuH = 110; // approximate menu height
                      const flipUp = rect.bottom + menuH > window.innerHeight;
                      setMenuPosition({
                        top: flipUp ? rect.top - menuH : rect.bottom + 4,
                        left: Math.max(4, rect.right - 144), // 144 = w-36
                      });
                      setMenuOpenId(conv.conversation_id);
                    }
                  }}
                  className="p-1 opacity-0 group-hover:opacity-100 hover:bg-gray-600 rounded text-gray-500 hover:text-gray-300 transition-all"
                  title="More actions"
                  type="button"
                >
                  <MoreHorizontal size={13} />
                </button>
                {menuOpenId === conv.conversation_id && menuPosition && (
                  <div
                    ref={menuRef}
                    className="fixed z-[200] w-36 bg-gray-900 border border-gray-700 rounded-lg shadow-xl overflow-hidden"
                    style={{ top: menuPosition.top, left: menuPosition.left }}
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenId(null);
                        onToggleBookmark?.(conv.conversation_id, !conv.bookmarked);
                      }}
                      className="w-full text-left px-3 py-1.5 text-[11px] text-gray-300 hover:bg-gray-800 flex items-center gap-2 transition-colors"
                      type="button"
                    >
                      <Star size={12} className={conv.bookmarked ? 'text-yellow-500 fill-yellow-500' : ''} />
                      {conv.bookmarked ? 'Unbookmark' : 'Bookmark'}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenId(null);
                        onOpenSettings(conv.conversation_id);
                      }}
                      className="w-full text-left px-3 py-1.5 text-[11px] text-gray-300 hover:bg-gray-800 flex items-center gap-2 transition-colors"
                      type="button"
                    >
                      <Settings size={12} /> Settings
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenId(null);
                        setDeleteTargetId(conv.conversation_id);
                      }}
                      className="w-full text-left px-3 py-1.5 text-[11px] text-red-400 hover:bg-gray-800 flex items-center gap-2 transition-colors"
                      type="button"
                    >
                      <Trash2 size={12} /> Delete
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Click-away overlay for fixed menu */}
      {menuOpenId && (
        <div className="fixed inset-0 z-[199]" onClick={() => { setMenuOpenId(null); setMenuPosition(null); }} />
      )}

      {/* Delete confirmation modal */}
      <ConfirmModal
        isOpen={!!deleteTargetId}
        title="Delete conversation"
        description="This conversation and all its messages will be permanently deleted. This action cannot be undone."
        confirmLabel="Delete"
        variant="danger"
        onConfirm={() => {
          if (deleteTargetId) onDelete(deleteTargetId);
          setDeleteTargetId(null);
        }}
        onCancel={() => setDeleteTargetId(null)}
      />
    </div>
  );
};
