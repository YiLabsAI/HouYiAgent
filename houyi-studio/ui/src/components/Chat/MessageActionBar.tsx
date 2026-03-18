/**
 * MessageActionBar: hover action bar for chat messages.
 *
 * Shows contextual actions based on message role:
 * - User messages: resend, edit, copy, delete
 * - Assistant messages: regenerate, copy, delete
 *
 */
import React from 'react';
import { Copy, Check, Pencil, Trash2, RefreshCw, RotateCcw, Bookmark, Pin } from 'lucide-react';
import type { ChatMessage } from '@/types/chat';
import { useChatStore } from '@/stores/useChatStore';
import { useConsoleStore } from '@/stores/useConsoleStore';
import { ConfirmModal } from '@/components/ConfirmModal';

interface MessageActionBarProps {
  message: ChatMessage;
  onStartEdit?: () => void;
}

export const MessageActionBar: React.FC<MessageActionBarProps> = ({ message, onStartEdit }) => {
  const [copied, setCopied] = React.useState(false);
  const [isDeleting, setIsDeleting] = React.useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = React.useState(false);
  const [isPinMenuOpen, setIsPinMenuOpen] = React.useState(false);
  const deleteMessage = useChatStore((s) => s.deleteMessage);
  const regenerateMessage = useChatStore((s) => s.regenerateMessage);
  const resendMessage = useChatStore((s) => s.resendMessage);
  const toggleMessageBookmark = useChatStore((s) => s.toggleMessageBookmark);
  const pinMessageToContext = useChatStore((s) => s.pinMessageToContext);
  const updatePinnedContextStatus = useChatStore((s) => s.updatePinnedContextStatus);
  const activePins = useChatStore((s) => s.activePins);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const activeConversation = useChatStore((s) => s.activeConversation);
  const composerUiState = useChatStore((s) => (
    s.activeConversationId ? s.composerUiByConversation[s.activeConversationId] : undefined
  ));
  const isStreaming = useChatStore((s) => s.streaming.isStreaming);
  const runSettings = useConsoleStore((s) => s.runSettings);

  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const activePin = activePins.find((pin) => pin.source_message_id === message.message_id);
  const replaceablePins = activePins.filter((pin) => pin.source_message_id !== message.message_id);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleResend = () => {
    if (isStreaming) return;
    if (!activeConversationId || !activeConversation) return;
    resendMessage(message.content, {
      enable_reasoning: composerUiState?.enableReasoning || undefined,
      enable_web_search: composerUiState?.enableWebSearch || undefined,
      enable_deep_research: composerUiState?.enableDeepResearch || undefined,
      max_tokens: composerUiState?.maxTokensDraft?.trim()
        ? parseInt(composerUiState.maxTokensDraft.trim(), 10)
        : activeConversation.max_tokens ?? undefined,
      stream: activeConversation.stream ?? undefined,
      enable_tool_calls: runSettings.enable_tool_calls,
      tool_call_strategy: runSettings.tool_call_strategy,
    });
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

  const handlePinClick = () => {
    if (activePin) {
      updatePinnedContextStatus(activePin.pin_id, 'archived');
      return;
    }
    if (replaceablePins.length > 0) {
      setIsPinMenuOpen((value) => !value);
      return;
    }
    pinMessageToContext(message.message_id);
  };

  const handleAddPin = () => {
    setIsPinMenuOpen(false);
    pinMessageToContext(message.message_id);
  };

  const handleReplacePin = (pinId: string) => {
    setIsPinMenuOpen(false);
    pinMessageToContext(message.message_id, { replacePinId: pinId });
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

      <div className="relative">
        <button
          onClick={handlePinClick}
          className={`p-1 rounded transition-colors ${
            activePin
              ? 'text-cyan-400 hover:bg-gray-700'
              : 'hover:bg-gray-700 text-gray-400 hover:text-cyan-400'
          }`}
          title={activePin ? 'Archive pin' : 'Pin to context'}
          type="button"
        >
          <Pin size={13} className={activePin ? 'fill-cyan-400' : ''} />
        </button>
        {isPinMenuOpen ? (
          <div className="absolute left-0 top-8 z-10 min-w-[180px] rounded border border-gray-700 bg-gray-900 p-1 shadow-xl">
            <div className="px-2 py-1 text-[10px] text-gray-400">Pin to context</div>
            <button
              type="button"
              onClick={handleAddPin}
              className="flex w-full items-center rounded px-2 py-1.5 text-left text-[11px] text-gray-200 hover:bg-gray-800"
            >
              Add as new pin
            </button>
            {replaceablePins.map((pin) => (
              <button
                key={pin.pin_id}
                type="button"
                onClick={() => handleReplacePin(pin.pin_id)}
                className="flex w-full items-center rounded px-2 py-1.5 text-left text-[11px] text-gray-200 hover:bg-gray-800"
              >
                Replace {pin.title || 'existing pin'}
              </button>
            ))}
          </div>
        ) : null}
      </div>

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
