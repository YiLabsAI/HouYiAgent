/**
 * ChatPage: main chat workspace that replaces the DAG canvas.
 *
 * Layout: full-height flex column with ChatTimeline + Composer.
 * Shows context usage bar when available.
 *
 * BUG-041 flicker fix: messages are wrapped in useDeferredValue so that
 * React prepares the new message DOM tree in the background while keeping
 * the old conversation visible.  This eliminates the per-frame repaint
 * flash that occurs when React replaces many MessageBubble nodes at once.
 */
import React from 'react';
import { useChatStore } from '@/stores/useChatStore';
import { useSettingsStore } from '@/stores/useSettingsStore';
import { ChatTimeline } from './ChatTimeline';
import { Composer } from './Composer';
import { TraceDetailPanel } from './TraceDetailPanel';
import { MessageCircle } from 'lucide-react';

export const ChatPage: React.FC = () => {
  const activeConversation = useChatStore((s) => s.activeConversation);
  const isLoadingConversation = useChatStore((s) => s.isLoadingConversation);
  const streaming = useChatStore((s) => s.streaming);
  const contextUsage = useChatStore((s) => s.contextUsage);
  const error = useChatStore((s) => s.error);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const stopStreaming = useChatStore((s) => s.stopStreaming);
  const clearError = useChatStore((s) => s.clearError);
  const [tracePanelId, setTracePanelId] = React.useState<string | null>(null);

  const fetchSettings = useSettingsStore((s) => s.fetchSettings);
  React.useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  // BUG-041: zero-flicker conversation switch
  // The store replaces activeConversation atomically (old data kept until
  // new data arrives).  ChatTimeline uses useLayoutEffect to scroll to
  // bottom BEFORE the browser paints, preventing the flash where messages
  // appear at scrollTop=0 before jumping to the bottom.
  const messages = React.useMemo(
    () => activeConversation?.messages.filter((m) => m.role !== 'system') || [],
    [activeConversation],
  );

  // Defer heavy DOM updates when switching to a conversation with many messages.
  // This keeps the old timeline visible while React prepares the new tree.
  const deferredMessages = React.useDeferredValue(messages);

  // No active conversation — show empty state
  if (!activeConversation && !isLoadingConversation) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-gray-500 bg-gray-900">
        <MessageCircle size={40} className="mb-4 opacity-30" />
        <p className="text-sm" data-testid="chat-empty-state">Select or create a conversation</p>
        <p className="text-xs mt-1 opacity-60">Use the sidebar to get started</p>
      </div>
    );
  }

  // Loading state — only show spinner if we have NO conversation data at all.
  // When switching between conversations, keep showing the old content until
  // the new data arrives to avoid flash-of-empty-state (BUG-041).
  if (isLoadingConversation && !activeConversation) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 bg-gray-900">
        <div className="text-sm animate-pulse">Loading conversation...</div>
      </div>
    );
  }

  return (
    <div
      className="flex-1 flex flex-col bg-gray-900 min-h-0"
      data-testid="chat-page"
    >
      {/* Context usage bar */}
      {contextUsage && contextUsage.max_context_tokens > 0 && (
        <div className="shrink-0 px-4 py-1.5 border-b border-gray-800 flex items-center gap-2">
          <span className="text-[10px] text-gray-500 shrink-0">Context</span>
          <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden max-w-[200px]">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                contextUsage.used_tokens / contextUsage.max_context_tokens > 0.9
                  ? 'bg-red-500'
                  : contextUsage.used_tokens / contextUsage.max_context_tokens > 0.7
                    ? 'bg-amber-500'
                    : 'bg-blue-500'
              }`}
              style={{
                width: `${Math.min(100, (contextUsage.used_tokens / contextUsage.max_context_tokens) * 100)}%`,
              }}
            />
          </div>
          <span className="text-[10px] text-gray-600 shrink-0 tabular-nums">
            {contextUsage.used_tokens.toLocaleString()} / {contextUsage.max_context_tokens.toLocaleString()} tokens
            ({Math.round((contextUsage.used_tokens / contextUsage.max_context_tokens) * 100)}%)
          </span>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="shrink-0 px-4 py-2 bg-red-900/30 border-b border-red-800/50 flex items-center justify-between">
          <span className="text-[12px] text-red-300">{error}</span>
          <button
            onClick={clearError}
            className="text-[11px] text-red-400 hover:text-red-300 underline"
            type="button"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Message timeline — updated via startTransition to avoid repaint flicker */}
      {/* Only show streaming indicators when the active conversation owns the stream */}
      {(() => {
        const isStreamingHere = streaming.isStreaming && streaming.streamConversationId === activeConversation?.conversation_id;
        return (
          <>
            <ChatTimeline
              messages={deferredMessages}
              streamingMessageId={isStreamingHere ? streaming.messageId : null}
              isWaitingForResponse={isStreamingHere && !streaming.messageId}
              conversationId={activeConversation?.conversation_id ?? null}
              onOpenTrace={setTracePanelId}
            />

            {/* Composer */}
            <Composer
              onSend={(content, options) => {
                const enableSkills: string[] = [];
                if (options?.enableWebSearch) enableSkills.push('web_search');
                if (options?.enableDeepResearch) enableSkills.push('deep_research');
                sendMessage(content, {
                  enable_reasoning: options?.enableReasoning,
                  enable_tool_calls: options?.enableToolCalls,
                  tool_call_strategy: options?.toolCallStrategy,
                  enable_web_search: options?.enableWebSearch,
                  enable_skills: enableSkills.length > 0 ? enableSkills : undefined,
                }, options?.attachments);
              }}
              onStop={stopStreaming}
              isStreaming={isStreamingHere}
            />
          </>
        );
      })()}

      {tracePanelId && (
        <TraceDetailPanel traceId={tracePanelId} onClose={() => setTracePanelId(null)} />
      )}
    </div>
  );
};
