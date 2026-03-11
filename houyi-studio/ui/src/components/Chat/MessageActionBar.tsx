/**
 * MessageActionBar: hover action bar for chat messages.
 *
 * Shows contextual actions based on message role:
 * - User messages: resend, edit, copy, delete
 * - Assistant messages: regenerate, copy, delete
 *
 */
import React from 'react';
import { Copy, Check, Pencil, Trash2, RefreshCw, RotateCcw, Bookmark } from 'lucide-react';
import type { ChatMessage } from '@/types/chat';
import { useChatStore } from '@/stores/useChatStore';
import { ConfirmModal } from '@/components/ConfirmModal';

interface MessageActionBarProps {
  message: ChatMessage;
  onStartEdit?: () => void;
}

export const MessageActionBar: React.FC<MessageActionBarProps> = ({ message, onStartEdit }) => {
  const [copied, setCopied] = React.useState(false);
  const [isDeleting, setIsDeleting] = React.useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = React.useState(false);
  const deleteMessage = useChatStore((s) => s.deleteMessage);
  const regenerateMessage = useChatStore((s) => s.regenerateMessage);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const toggleMessageBookmark = useChatStore((s) => s.toggleMessageBookmark);
  const isStreaming = useChatStore((s) => s.streaming.isStreaming);

  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleResend = () => {
    if (isStreaming) return;
    sendMessage(message.content);
  };

  const handleRegenerate = () => {
    if (isStreaming) return;
    regenerateMessage(message.message_id);
  };

  const handleConfirmDelete = async () => {
    if (isStreaming || isDeleting) return;
    setIsDeleteConfirmOpen(false);
    setIsDeleting(true);
    try {
      await deleteMessage(message.message_id);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="flex items-center gap-0.5 px-1 py-0.5 bg-gray-800 border border-gray-700/50 rounded-md shadow-lg">
      {/* Resend (user only) */}
      {isUser && (
        <button
          onClick={handleResend}
          disabled={isStreaming}
          className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 disabled:opacity-30 transition-colors"
          title="Resend"
          type="button"
        >
          <RotateCcw size={13} />
        </button>
      )}

      {/* Regenerate (assistant only) */}
      {isAssistant && (
        <button
          onClick={handleRegenerate}
          disabled={isStreaming}
          className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 disabled:opacity-30 transition-colors"
          title="Regenerate"
          type="button"
        >
          <RefreshCw size={13} />
        </button>
      )}

      {/* Edit (user only) */}
      {isUser && onStartEdit && (
        <button
          onClick={onStartEdit}
          className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
          title="Edit"
          type="button"
        >
          <Pencil size={13} />
        </button>
      )}

      {/* Copy */}
      <button
        onClick={handleCopy}
        className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
        title="Copy"
        type="button"
      >
        {copied ? <Check size={13} className="text-green-400" /> : <Copy size={13} />}
      </button>

      {/* Bookmark */}
      <button
        onClick={() => toggleMessageBookmark(message.message_id)}
        className={`p-1 rounded transition-colors ${
          message.bookmarked
            ? 'text-yellow-500 hover:bg-gray-700'
            : 'hover:bg-gray-700 text-gray-400 hover:text-yellow-500'
        }`}
        title={message.bookmarked ? 'Remove bookmark' : 'Bookmark'}
        type="button"
      >
        <Bookmark size={13} className={message.bookmarked ? 'fill-yellow-500' : ''} />
      </button>

      {/* Delete (with confirmation) */}
      <button
        onClick={() => setIsDeleteConfirmOpen(true)}
        disabled={isStreaming || isDeleting}
        className={`p-1 rounded transition-colors ${
          isDeleting
            ? 'bg-red-600/20 text-red-400 disabled:opacity-60'
            : 'hover:bg-gray-700 text-gray-400 hover:text-red-400 disabled:opacity-30'
        }`}
        title={isDeleting ? 'Deleting…' : 'Delete'}
        type="button"
      >
        <Trash2 size={13} />
      </button>
      <ConfirmModal
        isOpen={isDeleteConfirmOpen}
        title="Delete message"
        description="This message will be removed from the conversation."
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleConfirmDelete}
        onCancel={() => setIsDeleteConfirmOpen(false)}
      />
    </div>
  );
};
