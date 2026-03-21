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
import type { TokenUsage } from '@/types/chat';
import { ChatTimeline } from './ChatTimeline';
import { Composer } from './Composer';
import { ChatInspectorPanel } from './ChatInspectorPanel';
import { TraceDetailPanel } from './TraceDetailPanel';
import { MessageCircle } from 'lucide-react';

const compactionTriggerLabels: Record<string, string> = {
  pre_request_pressure: 'Compacted conversation context for this request',
  overflow_recovery: 'Recovered rolling context capacity',
  manual: 'Compacted conversation context',
  post_turn_background: 'Compacted conversation context in background',
  repo_intent_trim: 'Trimmed request context before send',
};

function resolveConversationStateMeta(
  state: 'healthy' | 'elevated' | 'near_compaction' | 'compacted_recently',
  ratio: number,
) {
  if (state === 'compacted_recently') {
    if (ratio >= 0.9) {
      return {
        barClassName: 'bg-red-500',
        indicatorClassName: 'bg-red-400',
      };
    }
    if (ratio >= 0.7) {
      return {
        barClassName: 'bg-amber-500',
        indicatorClassName: 'bg-amber-400',
      };
    }
  }
  return {
    healthy: {
      barClassName: 'bg-sky-500',
      indicatorClassName: 'bg-sky-400',
    },
    elevated: {
      barClassName: 'bg-amber-500',
      indicatorClassName: 'bg-amber-400',
    },
    near_compaction: {
      barClassName: 'bg-red-500',
      indicatorClassName: 'bg-red-400',
    },
    compacted_recently: {
      barClassName: 'bg-emerald-500',
      indicatorClassName: 'bg-emerald-400',
    },
  }[state];
}

const compactionDismissStorageKey = 'houyi.chat.compactionNoticeDismissals';

function readCompactionDismissals(): Record<string, string> {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(compactionDismissStorageKey);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed as Record<string, string> : {};
  } catch {
    return {};
  }
}

function writeCompactionDismissals(value: Record<string, string>) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(compactionDismissStorageKey, JSON.stringify(value));
  } catch {
  }
}

export const ChatPage: React.FC = () => {
  const activeConversation = useChatStore((s) => s.activeConversation);
  const selectedConversationId = useChatStore((s) => s.activeConversationId);
  const isLoadingConversation = useChatStore((s) => s.isLoadingConversation);
  const streaming = useChatStore((s) => s.streaming);
  const contextUsage = useChatStore((s) => s.contextUsage);
  const latestCompaction = useChatStore((s) => s.latestCompaction);
  const compactionHistory = useChatStore((s) => s.compactionHistory);
  const isLoadingCompactions = useChatStore((s) => s.isLoadingCompactions);
  const restoringCompactionId = useChatStore((s) => s.restoringCompactionId);
  const restoringBackupId = useChatStore((s) => s.restoringBackupId);
  const activePins = useChatStore((s) => s.activePins);
  const agentLoopSummary = useChatStore((s) => s.agentLoopSummary);
  const restoreNotice = useChatStore((s) => s.restoreNotice);
  const error = useChatStore((s) => s.error);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const stopStreaming = useChatStore((s) => s.stopStreaming);
  const fetchCompactions = useChatStore((s) => s.fetchCompactions);
  const restoreCompaction = useChatStore((s) => s.restoreCompaction);
  const restoreBackup = useChatStore((s) => s.restoreBackup);
  const updatePinnedContextStatus = useChatStore((s) => s.updatePinnedContextStatus);
  const clearRestoreNotice = useChatStore((s) => s.clearRestoreNotice);
  const clearError = useChatStore((s) => s.clearError);
  const [tracePanelId, setTracePanelId] = React.useState<string | null>(null);
  const [chatInspectorOpen, setChatInspectorOpen] = React.useState(false);
  const [dismissedCompactions, setDismissedCompactions] = React.useState<Record<string, string>>(
    () => readCompactionDismissals(),
  );

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
  const latestAssistantUsage = React.useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role !== 'assistant') continue;
      const metadata = message.metadata;
      if (metadata && typeof metadata === 'object') {
        const nestedUsage = metadata.usage && typeof metadata.usage === 'object'
          ? metadata.usage as TokenUsage
          : null;
        const merged = {
          ...(nestedUsage || {}),
          ...metadata,
        } as TokenUsage;
        if (Object.keys(merged).length > 0) return merged;
      }
    }
    if (agentLoopSummary.usage || agentLoopSummary.metrics) {
      return {
        ...((agentLoopSummary.usage as TokenUsage | null) || {}),
        ...((agentLoopSummary.metrics as TokenUsage | null) || {}),
      } as TokenUsage;
    }
    return null;
  }, [agentLoopSummary.metrics, agentLoopSummary.usage, messages]);

  // Defer heavy DOM updates when switching to a conversation with many messages.
  // This keeps the old timeline visible while React prepares the new tree.
  const deferredMessages = React.useDeferredValue(messages);
  const isStreamingHere = streaming.isStreaming && streaming.streamConversationId === activeConversation?.conversation_id;
  const timelineMessages = isStreamingHere ? messages : deferredMessages;
  const activeConversationId = activeConversation?.conversation_id ?? null;
  const restoreNoticeBelongsToSelectedConversation = Boolean(
    restoreNotice
    && restoreNotice.conversationId
    && restoreNotice.conversationId === selectedConversationId,
  );
  const conversationContext = activeConversation?.conversation_context_state ?? null;
  const conversationRatio = conversationContext && conversationContext.max_units > 0
    ? conversationContext.used_units / conversationContext.max_units
    : 0;
  const pinViolationCount = Number(latestCompaction?.metrics?.pin_violation_count || 0);
  const compactionTokensBefore = Number(latestCompaction?.metrics?.tokens_before || 0);
  const compactionTokensAfter = Number(latestCompaction?.metrics?.tokens_after || 0);
  const compactionSavedTokens = Math.max(0, compactionTokensBefore - compactionTokensAfter);
  const compactionTriggerLabel = latestCompaction
    ? (compactionTriggerLabels[latestCompaction.trigger] ?? 'Compacted conversation context')
    : 'Compacted conversation context';
  const conversationCapacityLabel = conversationContext
    ? `${conversationContext.used_units.toLocaleString()} / ${conversationContext.max_units.toLocaleString()}`
    : '0 / 0';
  const dismissedCompactionId = activeConversationId
    ? dismissedCompactions[activeConversationId] ?? null
    : null;
  const conversationState = conversationContext
    ? resolveConversationStateMeta(conversationContext.state, conversationRatio)
    : resolveConversationStateMeta('healthy', 0);
  const openChatInspector = () => {
    setTracePanelId(null);
    setChatInspectorOpen(true);
    if (activeConversation?.conversation_id) {
      void fetchCompactions(activeConversation.conversation_id);
    }
  };
  const openTracePanel = (traceId: string) => {
    setChatInspectorOpen(false);
    setTracePanelId(traceId);
  };
  const dismissCompactionNotice = React.useCallback(() => {
    if (!activeConversationId || !latestCompaction?.compaction_id) return;
    setDismissedCompactions((current) => {
      const next = {
        ...current,
        [activeConversationId]: latestCompaction.compaction_id,
      };
      writeCompactionDismissals(next);
      return next;
    });
  }, [activeConversationId, latestCompaction?.compaction_id]);
  const shouldShowCompactionNotice = Boolean(
    latestCompaction && latestCompaction.compaction_id !== dismissedCompactionId,
  );

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
      <div className="shrink-0 border-b border-gray-800 px-4 py-2.5" data-testid="chat-top-rail">
        <div className="flex min-w-0 flex-wrap items-center gap-3 text-[11px] text-gray-300">
          <div className="flex min-w-0 flex-1 items-center gap-3 rounded-md border border-gray-800 bg-gray-900/70 px-3 py-2" data-testid="conversation-summary">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-gray-400">Rolling Context</span>
                <span
                  className={`inline-block h-2 w-2 rounded-full ${conversationState.indicatorClassName}`}
                  title="Rolling context state. Open Inspect for detailed request and compaction status."
                  data-testid="conversation-state-indicator"
                />
              </div>
              <div className="mt-1 flex min-w-0 items-center gap-3">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-gray-800">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${conversationState.barClassName}`}
                    style={{ width: `${Math.min(100, conversationRatio * 100)}%` }}
                  />
                </div>
                <span
                  className="shrink text-[10px] text-gray-500 tabular-nums whitespace-nowrap"
                  title="Conversation rolling context budget. Compaction thresholds are evaluated against this session-level budget when conversation context state is available."
                >
                  {conversationCapacityLabel}
                </span>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                onClick={openChatInspector}
                className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-700"
              >
                Inspect
              </button>
            </div>
          </div>
        </div>
      </div>

      {shouldShowCompactionNotice && latestCompaction && (
        <div
          className="shrink-0 border-b border-gray-800/80 bg-amber-950/20 px-4 py-2"
          data-testid="compaction-notice"
        >
          <div className="flex min-w-0 flex-wrap items-center justify-between gap-3 overflow-x-hidden">
            <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2 text-[11px]">
              <span className="min-w-0 break-words text-amber-300">{compactionTriggerLabel}</span>
              {compactionSavedTokens > 0 && (
                <span
                  className="max-w-full rounded border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] text-amber-200"
                  title={`Latest compaction: ${compactionTokensBefore.toLocaleString()} → ${compactionTokensAfter.toLocaleString()} tokens.`}
                  data-testid="compaction-saved-badge"
                >
                  Latest save {compactionSavedTokens.toLocaleString()} tokens
                </span>
              )}
              <span
                className={`max-w-full rounded border px-1.5 py-0.5 text-[10px] ${
                  pinViolationCount > 0
                    ? 'border-red-500/30 bg-red-500/10 text-red-200'
                    : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'
                }`}
              >
                {pinViolationCount > 0 ? `Pin violations ${pinViolationCount}` : 'Pins protected'}
              </span>
            </div>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={openChatInspector}
                className="rounded border border-gray-700 bg-gray-900/80 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800"
              >
                Details
              </button>
              <button
                type="button"
                onClick={dismissCompactionNotice}
                className="rounded border border-gray-800 bg-gray-950/70 px-2 py-1 text-[10px] text-gray-400 hover:bg-gray-900 hover:text-gray-200"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      {restoreNotice?.undoBackupId && restoreNoticeBelongsToSelectedConversation && (
        <div className="shrink-0 border-b border-cyan-900/40 bg-cyan-950/20 px-4 py-2" data-testid="restore-notice">
          <div className="flex min-w-0 flex-wrap items-center justify-between gap-3 overflow-x-hidden">
            <span className="min-w-0 flex-1 break-words text-[11px] text-cyan-200">{restoreNotice.message}</span>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              {restoreNotice.undoBackupId ? (
                <button
                  type="button"
                  onClick={() => restoreBackup(restoreNotice.undoBackupId as string)}
                  disabled={restoringBackupId === restoreNotice.undoBackupId}
                  className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] text-cyan-200 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {restoringBackupId === restoreNotice.undoBackupId ? 'Undoing restore…' : 'Undo restore'}
                </button>
              ) : null}
              <button
                type="button"
                onClick={clearRestoreNotice}
                className="rounded border border-gray-800 bg-gray-950/70 px-2 py-1 text-[10px] text-gray-400 hover:bg-gray-900 hover:text-gray-200"
              >
                Dismiss
              </button>
            </div>
          </div>
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
        return (
          <>
            <ChatTimeline
              messages={timelineMessages}
              streamingMessageId={isStreamingHere ? streaming.messageId : null}
              isWaitingForResponse={isStreamingHere && !streaming.messageId}
              conversationId={activeConversation?.conversation_id ?? null}
              onOpenTrace={openTracePanel}
            />

            {/* Composer */}
            <Composer
              conversationId={activeConversation?.conversation_id ?? null}
              onSend={(content, options) => {
                const enableSkills: string[] = [];
                if (options?.enableWebSearch) enableSkills.push('houyi_web_search');
                if (options?.enableDeepResearch) enableSkills.push('deep_research');
                sendMessage(content, {
                  enable_reasoning: options?.enableReasoning,
                  enable_tool_calls: options?.enableToolCalls,
                  tool_call_strategy: options?.toolCallStrategy,
                  enable_web_search: options?.enableWebSearch,
                  enable_deep_research: options?.enableDeepResearch,
                  enable_skills: enableSkills.length > 0 ? enableSkills : undefined,
                }, options?.attachments);
              }}
              onStop={stopStreaming}
              isStreaming={isStreamingHere}
            />
          </>
        );
      })()}

      {chatInspectorOpen && (
        <ChatInspectorPanel
          conversationContext={conversationContext}
          contextUsage={contextUsage}
          latestCompaction={latestCompaction}
          compactionHistory={compactionHistory}
          isLoadingCompactions={isLoadingCompactions}
          restoringCompactionId={restoringCompactionId}
          restoringBackupId={restoringBackupId}
          restoreNotice={restoreNotice}
          activePins={activePins}
          usage={latestAssistantUsage}
          onUpdatePinnedContextStatus={updatePinnedContextStatus}
          onRestoreCompaction={restoreCompaction}
          onRestoreBackup={restoreBackup}
          onClearRestoreNotice={clearRestoreNotice}
          onClose={() => setChatInspectorOpen(false)}
        />
      )}
      {tracePanelId && (
        <TraceDetailPanel traceId={tracePanelId} onClose={() => setTracePanelId(null)} />
      )}
    </div>
  );
};
