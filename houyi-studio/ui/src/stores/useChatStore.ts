/**
 * Zustand store for Chat state management.
 *
 * Manages conversation list, active conversation, message streaming,
 * and SSE event handling for Chatbox.
 *
 */
import { create } from 'zustand';
import type {
  Attachment,
  Conversation,
  ConversationSummary,
  ChatMessage,
  ContextUsage,
  CompactionRecord,
  CompactionHistoryItem,
  CreateConversationRequest,
  SendMessageRequest,
  SSEAgentIteration,
  SSEContextCompacted,
  SSEContextStateUpdated,
  SSEMessageDelta,
  SSEMessageFinish,
  SSEMessageComplete,
  SSEMessageError,
  SSEContextUsage,
  SSEToolCallError,
  SSEToolCallResult,
  SSEToolCallStart,
  PinnedContextRecord,
  PinStatus,
} from '@/types/chat';
import { buildVisibleChatError } from '@/utils/chatErrors';

const API_BASE = '/api/chat';

function chatErrorProvider(model: string | null | undefined): 'generic' | 'gemini' {
  const value = String(model || '').toLowerCase();
  if (value.includes('gemini') || value.includes('vertex')) return 'gemini';
  return 'generic';
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const raw = await res.text().catch(() => res.statusText);
    let detail: unknown = raw;
    try {
      detail = raw ? JSON.parse(raw) : raw;
    } catch {
    }
    const visibleError = buildVisibleChatError(
      typeof detail === 'object' && detail
        ? (detail as Record<string, unknown>)
        : {
            error: String(raw || res.statusText),
            status_code: res.status,
          },
      'generic',
    );
    throw new Error(visibleError);
  }
  return res.json();
}

// --- Store types ---

interface StreamingState {
  isStreaming: boolean;
  messageId: string | null;
  contentBuffer: string;
  reasoningBuffer: string;
  abortController: AbortController | null;
  // Which conversation owns it
  streamConversationId: string | null;
  toolMessageIdsByCallId: Record<string, string>;
  suppressVisibleErrors: boolean;
}

type AgentLoopStatus = 'idle' | 'tool_loop' | 'finalizing' | 'done';

interface AgentLoopSummary {
  rounds: number;
  toolCalls: number;
  traceId: string | null;
  usage: Record<string, any> | null;
  metrics: Record<string, any> | null;
  status: AgentLoopStatus;
}

interface RestoreNotice {
  message: string;
  undoBackupId: string | null;
  kind?: 'restore_applied' | 'restore_undone';
  conversationId?: string | null;
}

interface ComposerUiState {
  enableReasoning: boolean;
  enableWebSearch: boolean;
  enableDeepResearch: boolean;
  showAdvanced: boolean;
  maxTokensDraft: string;
}

function buildResendRequestOptions(
  conversation: Conversation,
  composerUi: ComposerUiState | undefined,
  options?: Partial<SendMessageRequest>,
): Partial<SendMessageRequest> {
  const parsedMaxTokens = composerUi?.maxTokensDraft?.trim()
    ? parseInt(composerUi.maxTokensDraft.trim(), 10)
    : NaN;
  const composerMaxTokens = Number.isFinite(parsedMaxTokens) && parsedMaxTokens > 0
    ? parsedMaxTokens
    : undefined;
  const enableSkills = new Set<string>(options?.enable_skills ?? []);
  if (options?.enable_web_search ?? composerUi?.enableWebSearch) {
    enableSkills.add('houyi_web_search');
  }
  if (options?.enable_deep_research ?? composerUi?.enableDeepResearch) {
    enableSkills.add('deep_research');
  }
  return {
    enable_reasoning: options?.enable_reasoning ?? (composerUi?.enableReasoning || undefined),
    enable_web_search: options?.enable_web_search ?? (composerUi?.enableWebSearch || undefined),
    enable_deep_research: options?.enable_deep_research ?? (composerUi?.enableDeepResearch || undefined),
    enable_skills: enableSkills.size > 0 ? Array.from(enableSkills) : undefined,
    max_tokens: options?.max_tokens ?? composerMaxTokens ?? conversation.max_tokens ?? undefined,
    stream: options?.stream ?? conversation.stream ?? undefined,
    enable_tool_calls: options?.enable_tool_calls,
    tool_call_strategy: options?.tool_call_strategy,
    max_tool_iterations: options?.max_tool_iterations,
  };
}

function emptyComposerUiState(): ComposerUiState {
  return {
    enableReasoning: false,
    enableWebSearch: false,
    enableDeepResearch: false,
    showAdvanced: false,
    maxTokensDraft: '',
  };
}

function emptyAgentLoopSummary(): AgentLoopSummary {
  return { rounds: 0, toolCalls: 0, traceId: null, usage: null, metrics: null, status: 'idle' };
}

function deriveLatestCompaction(conversation: Conversation | null): CompactionRecord | null {
  const history = conversation?.metadata?.compaction_history;
  if (!Array.isArray(history) || history.length === 0) return null;
  const latest = history[history.length - 1];
  return latest && typeof latest === 'object' ? latest as CompactionRecord : null;
}

function deriveAgentLoopSummary(conversation: Conversation | null): AgentLoopSummary {
  const allMessages = conversation?.messages ?? [];
  const rounds = allMessages.reduce((maxRound, message) => {
    const roundIndex = Number(message?.metadata?.round_index ?? 0);
    return Number.isFinite(roundIndex) ? Math.max(maxRound, roundIndex) : maxRound;
  }, 0);
  const toolCallIds = new Set<string>();
  allMessages.forEach((message) => {
    if (message.role === 'tool' && typeof message.tool_call_id === 'string' && message.tool_call_id) {
      toolCallIds.add(message.tool_call_id);
    }
    if (message.role !== 'assistant' || !Array.isArray(message.tool_calls)) return;
    message.tool_calls.forEach((toolCall) => {
      if (!toolCall || typeof toolCall !== 'object') return;
      const callId = typeof (toolCall as Record<string, any>).id === 'string'
        ? String((toolCall as Record<string, any>).id)
        : '';
      if (callId) toolCallIds.add(callId);
    });
  });
  const toolCalls = toolCallIds.size;
  const assistantMessages = (conversation?.messages ?? []).filter((message) => message.role === 'assistant');
  for (let index = assistantMessages.length - 1; index >= 0; index -= 1) {
    const metadata = assistantMessages[index]?.metadata;
    if (!metadata || typeof metadata !== 'object') continue;
    const traceId = typeof metadata.trace_id === 'string' ? metadata.trace_id : null;
    const usage = metadata.usage && typeof metadata.usage === 'object' ? metadata.usage as Record<string, any> : null;
    const metrics = {
      finish_reason: metadata.finish_reason,
      budget: metadata.budget,
      first_token_latency_ms: metadata.first_token_latency_ms,
      first_token_ms: metadata.first_token_ms,
      generation_time_ms: metadata.generation_time_ms,
      decode_tokens_per_second: metadata.decode_tokens_per_second,
      end_to_end_tokens_per_second: metadata.end_to_end_tokens_per_second,
      tokens_per_second: metadata.tokens_per_second,
      tool_loop_convergence_reason: metadata.tool_loop_convergence_reason,
      tool_loop_final_stream_skipped: metadata.tool_loop_final_stream_skipped,
      final_stream_status: metadata.final_stream_status,
      final_stream_error_category: metadata.final_stream_error_category,
      final_stream_empty_visible_output: metadata.final_stream_empty_visible_output,
      final_stream_assistant_reasoning_removed_count: metadata.final_stream_assistant_reasoning_removed_count,
      final_stream_assistant_reasoning_only_removed_count: metadata.final_stream_assistant_reasoning_only_removed_count,
      final_stream_assistant_tool_call_carrier_count: metadata.final_stream_assistant_tool_call_carrier_count,
      final_stream_tool_result_projection_count: metadata.final_stream_tool_result_projection_count,
      request_adapter_class: metadata.request_adapter_class,
      request_adapter_strict_message_string_contract: metadata.request_adapter_strict_message_string_contract,
      request_message_count: metadata.request_message_count,
      request_user_message_count: metadata.request_user_message_count,
      request_assistant_message_count: metadata.request_assistant_message_count,
      request_assistant_reasoning_message_count: metadata.request_assistant_reasoning_message_count,
      request_assistant_reasoning_only_message_count: metadata.request_assistant_reasoning_only_message_count,
      request_assistant_tool_call_message_count: metadata.request_assistant_tool_call_message_count,
      request_tool_message_count: metadata.request_tool_message_count,
    };
    const hasMetrics = Object.values(metrics).some((value) => value !== undefined && value !== null);
    if (traceId || usage || hasMetrics) {
      return {
        ...emptyAgentLoopSummary(),
        rounds,
        toolCalls,
        traceId,
        usage,
        metrics: hasMetrics ? metrics : null,
        status: 'done',
      };
    }
  }
  return emptyAgentLoopSummary();
}

function deriveActivePins(conversation: Conversation | null): PinnedContextRecord[] {
  const pins = conversation?.metadata?.pinned_contexts;
  if (!Array.isArray(pins)) return [];
  return pins.filter((pin): pin is PinnedContextRecord => (
    !!pin
    && typeof pin === 'object'
    && typeof (pin as PinnedContextRecord).pin_id === 'string'
    && (pin as PinnedContextRecord).status === 'active'
  ));
}

function applyAuthoritativeContextState(
  state: ChatState,
  conversationId: string,
  conversationContextState: Conversation['conversation_context_state'] | null | undefined,
): Partial<ChatState> {
  if (!state.activeConversation || state.activeConversation.conversation_id !== conversationId) {
    return {};
  }
  return {
    activeConversation: {
      ...state.activeConversation,
      conversation_context_state: conversationContextState ?? null,
    },
  };
}

function mergeMessagesForRefresh(
  currentMessages: ChatMessage[],
  loadedMessages: ChatMessage[],
): ChatMessage[] {
  if (currentMessages.length === 0 || loadedMessages.length === 0) return loadedMessages;
  if (Math.abs(currentMessages.length - loadedMessages.length) > 2) return loadedMessages;

  const merged: ChatMessage[] = [];
  let currentIndex = 0;

  for (const loadedMessage of loadedMessages) {
    const currentMessage = currentMessages[currentIndex];
    if (!currentMessage) {
      merged.push(loadedMessage);
      continue;
    }

    if (currentMessage.role !== loadedMessage.role) {
      merged.push(loadedMessage);
      continue;
    }

    merged.push({
      ...currentMessage,
      ...loadedMessage,
      ui_render_id: currentMessage.ui_render_id || loadedMessage.ui_render_id || loadedMessage.message_id,
      metadata: {
        ...currentMessage.metadata,
        ...loadedMessage.metadata,
      },
    });
    currentIndex += 1;
  }

  return merged;
}

function scheduleConversationRefresh(
  conversationId: string,
  get: () => ChatState,
  preserveRenderedMessages = false,
  delayMs = 250,
) {
  setTimeout(() => {
    const state = get();
    if (state.activeConversationId !== conversationId) return;
    if (state.streaming.isStreaming) return;
    void state.loadConversation(conversationId, undefined, preserveRenderedMessages);
  }, delayMs);
}

interface ChatState {
  // Conversation list
  conversations: ConversationSummary[];
  isLoadingList: boolean;

  // Active conversation
  activeConversationId: string | null;
  activeConversation: Conversation | null;
  isLoadingConversation: boolean;

  // Streaming
  streaming: StreamingState;

  // Context usage (latest)
  contextUsage: ContextUsage | null;

  latestCompaction: CompactionRecord | null;
  compactionHistory: CompactionHistoryItem[];
  isLoadingCompactions: boolean;
  restoringCompactionId: string | null;
  restoringBackupId: string | null;
  activePins: PinnedContextRecord[];
  restoreNotice: RestoreNotice | null;

  // Agent loop summary for current/last streamed assistant response
  agentLoopSummary: AgentLoopSummary;

  // Error
  error: string | null;

  // Search navigation: scroll to a specific message after loading
  scrollToMessageId: string | null;
  composerUiByConversation: Record<string, ComposerUiState>;

  // Actions
  fetchConversations: () => Promise<void>;
  createConversation: (req?: CreateConversationRequest) => Promise<string | undefined>;
  loadConversation: (
    conversationId: string,
    scrollToMessageId?: string,
    preserveRenderedMessages?: boolean,
  ) => Promise<void>;
  updateConversation: (conversationId: string, updates: { title?: string; status?: string; model?: string; system_instructions?: string; temperature?: number | null; max_tokens?: number | null; top_p?: number | null; stream?: boolean | null; bookmarked?: boolean }) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  sendMessage: (content: string, options?: Partial<SendMessageRequest>, files?: File[]) => Promise<void>;
  resendMessage: (content: string, options?: Partial<SendMessageRequest>) => Promise<void>;
  regenerateMessage: (messageId: string) => Promise<void>;
  stopStreaming: () => void;
  clearError: () => void;
  clearScrollTarget: () => void;
  setComposerUiState: (conversationId: string, ui: Partial<ComposerUiState>) => void;

  // Message operations
  editMessage: (messageId: string, content: string) => Promise<void>;
  deleteMessage: (messageId: string) => Promise<void>;
  toggleMessageBookmark: (messageId: string) => Promise<void>;
  pinMessageToContext: (messageId: string, options?: { replacePinId?: string; title?: string }) => Promise<void>;
  updatePinnedContextStatus: (pinId: string, status: PinStatus) => Promise<void>;
  fetchCompactions: (conversationId?: string) => Promise<void>;
  restoreCompaction: (compactionId: string) => Promise<void>;
  restoreBackup: (backupId: string) => Promise<void>;
  clearRestoreNotice: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  // Initial state
  conversations: [],
  isLoadingList: false,
  activeConversationId: null,
  activeConversation: null,
  isLoadingConversation: false,
  streaming: {
    isStreaming: false,
    messageId: null,
    contentBuffer: '',
    reasoningBuffer: '',
    abortController: null,
    streamConversationId: null,
    toolMessageIdsByCallId: {},
    suppressVisibleErrors: false,
  },
  contextUsage: null,
  latestCompaction: null,
  compactionHistory: [],
  isLoadingCompactions: false,
  restoringCompactionId: null,
  restoringBackupId: null,
  activePins: [],
  restoreNotice: null,
  agentLoopSummary: emptyAgentLoopSummary(),
  error: null,
  scrollToMessageId: null,
  composerUiByConversation: {},

  // --- Actions ---

  fetchConversations: async () => {
    set({ isLoadingList: true });
    try {
      const data = await apiFetch<{ conversations: ConversationSummary[] }>('/conversations');
      set({ conversations: data.conversations, isLoadingList: false });
    } catch (e: any) {
      set({ error: e.message, isLoadingList: false });
    }
  },

  createConversation: async (req?: CreateConversationRequest) => {
    set({ error: null });
    try {
      const data = await apiFetch<ConversationSummary>('/conversations', {
        method: 'POST',
        body: JSON.stringify(req || {}),
      });
      // Refresh list and load the new conversation
      await get().fetchConversations();
      await get().loadConversation(data.conversation_id);
      return data.conversation_id;
    } catch (e: any) {
      set({ error: e.message });
      throw e;
    }
  },

  loadConversation: async (
    conversationId: string,
    scrollToMessageId?: string,
    preserveRenderedMessages = false,
  ) => {
    const isSwitch = get().activeConversationId !== conversationId;

    // BUG-041 fix — zero-flicker conversation switch
    //
    // Strategy: during a switch, update ONLY activeConversationId (for
    // sidebar highlight) in the first set().  Do NOT touch any state that
    // ChatPage subscribes to.  ChatPage keeps rendering the OLD conversation.
    // When the new data arrives, replace everything in a single atomic set().
    //
    // DO NOT abort the SSE stream — let it finish in the background so the
    // backend persists the full response.  handleSSEEvent's conversationId
    // guard will silently discard delta events for the old conversation.
    // When the stream finishes, sendMessage's loadConversation call will
    // reload the completed data.
    if (isSwitch) {
      // Preserve streaming state — the SSE stream continues in the background.
      // handleSSEEvent will keep accumulating the buffer even though the UI
      // is showing a different conversation.
      set({
        activeConversationId: conversationId,
        error: null,
      });
    }

    try {
      // Parallel fetch: conversation + context-usage
      const [data, usageResult, compactionsResult] = await Promise.all([
        apiFetch<Conversation>(`/conversations/${conversationId}`),
        apiFetch<{ usage: any }>(`/conversations/${conversationId}/context-usage`).catch(() => null),
        apiFetch<{ items: CompactionHistoryItem[] }>(`/conversations/${conversationId}/compactions`).catch(() => null),
      ]);

      // Stale-response guard: user clicked another conversation while fetching
      if (get().activeConversationId !== conversationId) return;

      const currentStreaming = get().streaming;
      const shouldAdoptServerStreamingState = Boolean(
        data.active_streaming_state
        && (!currentStreaming.isStreaming || currentStreaming.streamConversationId === conversationId),
      );
      const streaming = shouldAdoptServerStreamingState
        ? {
            ...currentStreaming,
            isStreaming: true,
            messageId: data.active_streaming_state?.message_id ?? currentStreaming.messageId,
            streamConversationId: conversationId,
            abortController:
              currentStreaming.streamConversationId === conversationId
                ? currentStreaming.abortController
                : null,
            toolMessageIdsByCallId:
              currentStreaming.streamConversationId === conversationId
                ? currentStreaming.toolMessageIdsByCallId
                : {},
          }
        : currentStreaming;

      let conversationData = data;
      if (
        streaming.isStreaming &&
        streaming.streamConversationId === conversationId &&
        (
          streaming.messageId
          || streaming.contentBuffer
          || streaming.reasoningBuffer
          || Object.keys(streaming.toolMessageIdsByCallId).length > 0
        )
      ) {
        const messages = [...data.messages];
        const lastMsg = messages[messages.length - 1];
        const isStreamRender = data.stream !== false;
        const displayContent = isStreamRender ? streaming.contentBuffer : '';
        const displayReasoning = isStreamRender ? (streaming.reasoningBuffer || null) : null;
        const shouldReplaceExistingAssistant = Boolean(
          lastMsg
          && lastMsg.role === 'assistant'
          && streaming.messageId
          && lastMsg.message_id === streaming.messageId,
        );

        if (shouldReplaceExistingAssistant) {
          messages[messages.length - 1] = {
            ...lastMsg,
            content: displayContent,
            reasoning_content: displayReasoning,
          };
        } else {
          messages.push({
            message_id: streaming.messageId || `tmp-stream-${Date.now()}`,
            role: 'assistant' as const,
            content: displayContent,
            reasoning_content: displayReasoning,
            metadata: {},
            created_at: Date.now() / 1000,
          });
        }
        conversationData = { ...data, messages };
      }

      if (
        preserveRenderedMessages
        && !streaming.isStreaming
        && get().activeConversation?.conversation_id === conversationId
      ) {
        const currentConversation = get().activeConversation;
        if (currentConversation) {
          conversationData = {
            ...data,
            messages: mergeMessagesForRefresh(currentConversation.messages, data.messages),
          };
        }
      }

      // Single atomic set — ChatPage renders exactly once with the new data
      set({
        activeConversation: conversationData,
        isLoadingConversation: false,
        streaming,
        contextUsage: usageResult?.usage ?? null,
        latestCompaction: deriveLatestCompaction(conversationData),
        compactionHistory: compactionsResult?.items ?? [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        activePins: deriveActivePins(conversationData),
        agentLoopSummary: deriveAgentLoopSummary(conversationData),
        scrollToMessageId: scrollToMessageId ?? null,
        error: null,
        restoreNotice: get().restoreNotice?.conversationId === conversationId
          ? get().restoreNotice
          : null,
      });
    } catch (e: any) {
      if (get().activeConversationId !== conversationId) return;
      set({
        error: e.message,
        isLoadingConversation: false,
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        activeConversation: null,
      });
    }
  },

  updateConversation: async (conversationId: string, updates) => {
    set({ error: null });
    try {
      const summary = await apiFetch<Record<string, any>>(`/conversations/${conversationId}`, {
        method: 'PATCH',
        body: JSON.stringify(updates),
      });
      await get().fetchConversations();
      // Merge metadata from summary into active conversation WITHOUT
      // replacing messages (PATCH returns summary only, no messages).
      const current = get().activeConversation;
      if (current && get().activeConversationId === conversationId) {
        set({
          activeConversation: {
            ...current,
            title: summary.title ?? current.title,
            model: summary.model ?? current.model,
            system_instructions: summary.system_instructions ?? current.system_instructions,
            temperature: summary.temperature !== undefined ? summary.temperature : current.temperature,
            max_tokens: summary.max_tokens !== undefined ? summary.max_tokens : current.max_tokens,
            top_p: summary.top_p !== undefined ? summary.top_p : current.top_p,
            stream: summary.stream !== undefined ? summary.stream : current.stream,
            status: summary.status ?? current.status,
          },
        });
      }
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  deleteConversation: async (conversationId: string) => {
    set({ error: null });
    try {
      const wasActive = get().activeConversationId === conversationId;
      await apiFetch(`/conversations/${conversationId}`, { method: 'DELETE' });
      await get().fetchConversations();

      // Auto-select adjacent conversation to avoid flash-of-empty-state.
      // If the deleted conversation was active, pick the next one in the list.
      if (wasActive) {
        const remaining = get().conversations;
        if (remaining.length > 0) {
          await get().loadConversation(remaining[0].conversation_id);
        } else {
          set({ activeConversationId: null, activeConversation: null });
        }
      }
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  sendMessage: async (content: string, options?: Partial<SendMessageRequest>, files?: File[]) => {
    const { activeConversationId, activeConversation, streaming } = get();
    if (!activeConversationId || !activeConversation) {
      set({ error: 'No active conversation' });
      return;
    }
    if (streaming.isStreaming) {
      set({ error: 'A response is already streaming. Stop it before sending another message.' });
      return;
    }

    // Convert File[] to base64 Attachment[] before sending.
    // Uses FileReader.readAsDataURL which handles large files safely
    // (btoa + String.fromCharCode can stack-overflow on big buffers).
    let attachments: Attachment[] | undefined;
    if (files && files.length > 0) {
      attachments = await Promise.all(
        files.map((file) => new Promise<Attachment>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            resolve({
              filename: file.name,
              mime_type: file.type || 'application/octet-stream',
              data: reader.result as string,
              size: file.size,
            });
          };
          reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
          reader.readAsDataURL(file);
        })),
      );
    }

    const abortController = new AbortController();
    const optimisticUserRenderId = `ui-user-${Date.now()}`;

    // Optimistically add user message to UI
    const userMsg: ChatMessage = {
      message_id: `tmp-${Date.now()}`,
      ui_render_id: optimisticUserRenderId,
      role: 'user',
      content,
      attachments,
      metadata: {},
      created_at: Date.now() / 1000,
    };

    set((state) => ({
      activeConversation: state.activeConversation
        ? { ...state.activeConversation, messages: [...state.activeConversation.messages, userMsg] }
        : null,
      activePins: deriveActivePins(state.activeConversation),
      streaming: {
        isStreaming: true,
        messageId: null,
        contentBuffer: '',
        reasoningBuffer: '',
        abortController,
        streamConversationId: activeConversationId,
        toolMessageIdsByCallId: {},
        suppressVisibleErrors: false,
      },
      latestCompaction: null,
      agentLoopSummary: emptyAgentLoopSummary(),
      error: null,
    }));

    let sawTerminalEvent = false;

    try {
      const response = await fetch(
        `${API_BASE}/conversations/${activeConversationId}/messages`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content, ...options, attachments }),
          signal: abortController.signal,
        },
      );

      if (!response.ok) {
        const detail = await response.text().catch(() => response.statusText);
        throw new Error(`API ${response.status}: ${detail}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';
      let eventType = '';
      let eventData = '';

      const streamConversationId = activeConversationId;
      const processEvent = () => {
        if (eventType && eventData) {
          try {
            const data = JSON.parse(eventData);
            handleSSEEvent(eventType, data, set, get, streamConversationId);
            if (eventType === 'message.complete' || eventType === 'message.error' || eventType === 'message.aborted') {
              sawTerminalEvent = true;
            }
          } catch {
            // Skip malformed events
          }
        }
        eventType = '';
        eventData = '';
      };

      let streamDone = false;
      while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) {
          streamDone = true;
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6).trim();
          } else if (line === '') {
            processEvent();
          }
        }
      }

      // Process any remaining buffered data after stream ends
      if (buffer.trim()) {
        const remainingLines = buffer.split('\n');
        for (const line of remainingLines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6).trim();
          } else if (line === '') {
            processEvent();
          }
        }
      }
      // Flush any final event without trailing newline
      processEvent();

      set((state) => ({
        streaming: { ...state.streaming, isStreaming: false, abortController: null, streamConversationId: null },
      }));

      // Reload conversation to get persisted state + refresh sidebar counts.
      // Only reload the UI if the user is still viewing this conversation;
      // otherwise just refresh the sidebar list so message counts update.
      if (get().activeConversationId === streamConversationId) {
        void get().loadConversation(streamConversationId, undefined, true);
        scheduleConversationRefresh(streamConversationId, get, true);
      }
      void get().fetchConversations();
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User-initiated abort — expected
      } else if (sawTerminalEvent) {
        // The backend already produced a terminal SSE event for this turn.
        // Ignore transport noise after completion so stale timeout/network
        // messages do not overwrite a successful response.
      } else {
        // Only show error if user is still on this conversation
        if (get().activeConversationId === activeConversationId) {
          set({ error: e.message });
        }
      }
    } finally {
      // Always clear streaming state — the stream is done regardless of
      // which conversation the user is currently viewing.
      set((state) => ({
        streaming: { ...state.streaming, isStreaming: false, abortController: null, streamConversationId: null },
      }));
    }
  },

  stopStreaming: () => {
    const { streaming } = get();
    if (streaming.abortController) {
      streaming.abortController.abort();
    }
    set((state) => ({
      streaming: {
        ...state.streaming,
        isStreaming: false,
        messageId: null,
        contentBuffer: '',
        reasoningBuffer: '',
        abortController: null,
        streamConversationId: null,
        toolMessageIdsByCallId: {},
        suppressVisibleErrors: false,
      },
    }));
  },

  resendMessage: async (content: string, options?: Partial<SendMessageRequest>) => {
    const { activeConversationId, activeConversation, streaming, composerUiByConversation } = get();
    if (!activeConversationId || !activeConversation) {
      set({ error: 'No active conversation' });
      return;
    }
    if (streaming.isStreaming) {
      set({ error: 'A response is already streaming. Stop it before sending another message.' });
      return;
    }

    const abortController = new AbortController();
    const optimisticUserRenderId = `ui-user-${Date.now()}`;
    const userMsg: ChatMessage = {
      message_id: `tmp-${Date.now()}`,
      ui_render_id: optimisticUserRenderId,
      role: 'user',
      content,
      metadata: {},
      created_at: Date.now() / 1000,
    };
    const requestOptions = buildResendRequestOptions(
      activeConversation,
      composerUiByConversation[activeConversationId],
      options,
    );
    set((state) => ({
      activeConversation: state.activeConversation
        ? { ...state.activeConversation, messages: [...state.activeConversation.messages, userMsg] }
        : null,
      activePins: deriveActivePins(state.activeConversation),
      streaming: {
        isStreaming: true,
        messageId: null,
        contentBuffer: '',
        reasoningBuffer: '',
        abortController,
        streamConversationId: activeConversationId,
        toolMessageIdsByCallId: {},
        suppressVisibleErrors: true,
      },
      latestCompaction: null,
      agentLoopSummary: emptyAgentLoopSummary(),
      error: null,
    }));

    let sawTerminalEvent = false;

    try {
      const response = await fetch(
        `${API_BASE}/conversations/${activeConversationId}/messages`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content, ...requestOptions }),
          signal: abortController.signal,
        },
      );

      if (!response.ok) {
        const detail = await response.text().catch(() => response.statusText);
        throw new Error(`API ${response.status}: ${detail}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';
      let eventType = '';
      let eventData = '';
      const streamConversationId = activeConversationId;

      const processEvent = () => {
        if (eventType && eventData) {
          try {
            const data = JSON.parse(eventData);
            handleSSEEvent(eventType, data, set, get, streamConversationId);
            if (eventType === 'message.complete' || eventType === 'message.error' || eventType === 'message.aborted') {
              sawTerminalEvent = true;
            }
          } catch {
          }
        }
        eventType = '';
        eventData = '';
      };

      let streamDone = false;
      while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) {
          streamDone = true;
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6).trim();
          } else if (line === '') {
            processEvent();
          }
        }
      }

      if (buffer.trim()) {
        const remainingLines = buffer.split('\n');
        for (const line of remainingLines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6).trim();
          } else if (line === '') {
            processEvent();
          }
        }
      }
      processEvent();

      const shouldReloadConversationAfterStream = (() => {
        const state = get();
        if (state.activeConversationId !== activeConversationId) return false;
        return state.agentLoopSummary.toolCalls > 0
          || state.agentLoopSummary.status === 'finalizing'
          || Object.keys(state.streaming.toolMessageIdsByCallId).length > 0;
      })();

      set((state) => ({
        streaming: {
          ...state.streaming,
          isStreaming: false,
          abortController: null,
          streamConversationId: null,
          suppressVisibleErrors: false,
        },
      }));
      if (shouldReloadConversationAfterStream) {
        await get().loadConversation(activeConversationId, undefined, true);
      }
      void get().fetchConversations();
    } catch (e: any) {
      if (e.name !== 'AbortError' && !sawTerminalEvent && !get().streaming.suppressVisibleErrors) {
        if (get().activeConversationId === activeConversationId) {
          set({ error: e.message });
        }
      }
    } finally {
      set((state) => ({
        streaming: {
          ...state.streaming,
          isStreaming: false,
          abortController: null,
          streamConversationId: null,
          suppressVisibleErrors: false,
        },
      }));
    }
  },

  clearError: () => set({ error: null }),

  clearScrollTarget: () => set({ scrollToMessageId: null }),

  setComposerUiState: (conversationId, ui) => set((state) => ({
    composerUiByConversation: {
      ...state.composerUiByConversation,
      [conversationId]: {
        ...(state.composerUiByConversation[conversationId] ?? emptyComposerUiState()),
        ...ui,
      },
    },
  })),

  fetchCompactions: async (conversationId) => {
    const targetConversationId = conversationId ?? get().activeConversationId;
    if (!targetConversationId) return;
    set({ isLoadingCompactions: true, error: null });
    try {
      const data = await apiFetch<{ items: CompactionHistoryItem[] }>(
        `/conversations/${targetConversationId}/compactions`,
      );
      if (get().activeConversationId === targetConversationId) {
        set({
          compactionHistory: data.items,
          isLoadingCompactions: false,
        });
      }
    } catch (e: any) {
      if (get().activeConversationId === targetConversationId) {
        set({ error: e.message, isLoadingCompactions: false });
      }
    }
  },

  restoreCompaction: async (compactionId) => {
    const { activeConversationId } = get();
    if (!activeConversationId) return;
    set({ restoringCompactionId: compactionId, error: null });
    try {
      const data = await apiFetch<{
        status: string;
        backup_id: string;
        restored_compaction_id: string;
        restore_point_backup_id?: string | null;
        conversation_context_state?: Conversation['conversation_context_state'] | null;
        conversation: Conversation;
      }>(
        `/conversations/${activeConversationId}/compactions/${compactionId}/restore`,
        { method: 'POST' },
      );
      const restoredConversation = data.conversation;
      const restoredMessages = restoredConversation.messages ?? [];
      const latestMessageId = restoredMessages.length > 0
        ? restoredMessages[restoredMessages.length - 1]?.message_id ?? null
        : null;
      set((state) => applyAuthoritativeContextState(
        state,
        activeConversationId,
        data.conversation_context_state ?? restoredConversation.conversation_context_state,
      ));
      await get().loadConversation(activeConversationId, latestMessageId ?? undefined);
      set({
        restoringCompactionId: null,
        restoreNotice: {
          message: 'Restored snapshot. You can undo this restore using the restore-point backup created immediately before the restore.',
          undoBackupId: typeof data.restore_point_backup_id === 'string' && data.restore_point_backup_id
            ? data.restore_point_backup_id
            : null,
          kind: 'restore_applied',
          conversationId: activeConversationId,
        },
      });
    } catch (e: any) {
      set({ error: e.message, restoringCompactionId: null });
    }
  },

  restoreBackup: async (backupId) => {
    const { activeConversationId } = get();
    if (!activeConversationId) return;
    set({ restoringBackupId: backupId, error: null });
    try {
      const data = await apiFetch<{
        status: string;
        backup_id: string;
        conversation_context_state?: Conversation['conversation_context_state'] | null;
        conversation: Conversation;
      }>(
        `/conversations/${activeConversationId}/backups/${backupId}/restore`,
        { method: 'POST' },
      );
      const restoredConversation = data.conversation;
      const restoredMessages = restoredConversation.messages ?? [];
      const latestMessageId = restoredMessages.length > 0
        ? restoredMessages[restoredMessages.length - 1]?.message_id ?? null
        : null;
      set((state) => applyAuthoritativeContextState(
        state,
        activeConversationId,
        data.conversation_context_state ?? restoredConversation.conversation_context_state,
      ));
      await get().loadConversation(activeConversationId, latestMessageId ?? undefined);
      set({
        restoringBackupId: null,
        restoreNotice: {
          message: 'Undo restore completed. Returned to the conversation state captured by the restore-point backup.',
          undoBackupId: null,
          kind: 'restore_undone',
          conversationId: activeConversationId,
        },
      });
    } catch (e: any) {
      set({ error: e.message, restoringBackupId: null });
    }
  },

  clearRestoreNotice: () => set({ restoreNotice: null }),

  // --- Message operations ---

  editMessage: async (messageId: string, content: string) => {
    const { activeConversationId } = get();
    if (!activeConversationId) return;
    set({ error: null });
    try {
      const data = await apiFetch<{
        conversation_id: string;
        conversation_context_state?: Conversation['conversation_context_state'] | null;
      }>(`/conversations/${activeConversationId}/messages/${messageId}`, {
        method: 'PUT',
        body: JSON.stringify({ content }),
      });
      set((state) => applyAuthoritativeContextState(
        state,
        data.conversation_id ?? activeConversationId,
        data.conversation_context_state,
      ));
      await get().loadConversation(activeConversationId);
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  deleteMessage: async (messageId: string) => {
    const { activeConversationId, activeConversation } = get();
    if (!activeConversationId || !activeConversation) return;
    set({ error: null });

    // Optimistic removal from UI
    set({
      activeConversation: {
        ...activeConversation,
        messages: activeConversation.messages.filter((m) => m.message_id !== messageId),
      },
    });

    try {
      const data = await apiFetch<{
        conversation_id: string;
        conversation_context_state?: Conversation['conversation_context_state'] | null;
      }>(`/conversations/${activeConversationId}/messages/${messageId}`, {
        method: 'DELETE',
      });
      set((state) => applyAuthoritativeContextState(
        state,
        data.conversation_id ?? activeConversationId,
        data.conversation_context_state,
      ));
      await get().loadConversation(activeConversationId);
      await get().fetchConversations();
    } catch (e: any) {
      // Revert on failure
      set({ error: e.message });
      await get().loadConversation(activeConversationId);
    }
  },

  regenerateMessage: async (messageId: string) => {
    const { activeConversationId, activeConversation, streaming } = get();
    if (!activeConversationId || !activeConversation) return;
    if (streaming.isStreaming) {
      set({ error: 'A response is already streaming. Stop it before regenerating another message.' });
      return;
    }

    const abortController = new AbortController();
    set(() => ({
      activePins: deriveActivePins(activeConversation as Conversation),
      streaming: {
        isStreaming: true,
        messageId: null,
        contentBuffer: '',
        reasoningBuffer: '',
        abortController,
        streamConversationId: activeConversationId,
        toolMessageIdsByCallId: {},
        suppressVisibleErrors: false,
      },
      latestCompaction: null,
      agentLoopSummary: emptyAgentLoopSummary(),
      error: null,
    }));

    try {
      const response = await fetch(
        `${API_BASE}/conversations/${activeConversationId}/messages/${messageId}/regenerate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: abortController.signal,
        },
      );

      if (!response.ok) {
        const detail = await response.text().catch(() => response.statusText);
        throw new Error(`API ${response.status}: ${detail}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';
      let regenEventType = '';
      let regenEventData = '';

      const regenConversationId = activeConversationId;
      const processRegenEvent = () => {
        if (regenEventType && regenEventData) {
          try {
            const data = JSON.parse(regenEventData);
            handleSSEEvent(regenEventType, data, set, get, regenConversationId);
          } catch {
            // Skip malformed events
          }
        }
        regenEventType = '';
        regenEventData = '';
      };

      let regenStreamDone = false;
      while (!regenStreamDone) {
        const { done, value } = await reader.read();
        if (done) {
          regenStreamDone = true;
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            regenEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            regenEventData = line.slice(6).trim();
          } else if (line === '') {
            processRegenEvent();
          }
        }
      }

      // Process any remaining buffered data after stream ends
      if (buffer.trim()) {
        const remainingLines = buffer.split('\n');
        for (const line of remainingLines) {
          if (line.startsWith('event: ')) {
            regenEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            regenEventData = line.slice(6).trim();
          } else if (line === '') {
            processRegenEvent();
          }
        }
      }
      // Flush any final event without trailing newline
      processRegenEvent();

      void get().fetchConversations();
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User-initiated abort
      } else {
        if (get().activeConversationId === activeConversationId) {
          set({ error: e.message });
        }
      }
    } finally {
      set((state) => ({
        streaming: {
          ...state.streaming,
          isStreaming: false,
          abortController: null,
          streamConversationId: null,
        },
      }));
    }
  },

  toggleMessageBookmark: async (messageId: string) => {
    const { activeConversationId, activeConversation } = get();
    if (!activeConversationId || !activeConversation) return;

    const msg = activeConversation.messages.find((m) => m.message_id === messageId);
    if (!msg) return;

    const newBookmarked = !msg.bookmarked;

    // Optimistic update
    set((state) => {
      if (!state.activeConversation) return {};
      const messages = state.activeConversation.messages.map((m) =>
        m.message_id === messageId ? { ...m, bookmarked: newBookmarked } : m,
      );
      return { activeConversation: { ...state.activeConversation, messages } };
    });

    try {
      const resp = await fetch(
        `${API_BASE}/conversations/${activeConversationId}/messages/${messageId}/bookmark`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bookmarked: newBookmarked }),
        },
      );
      if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
      }
    } catch {
      // Revert on failure
      set((state) => {
        if (!state.activeConversation) return {};
        const messages = state.activeConversation.messages.map((m) =>
          m.message_id === messageId ? { ...m, bookmarked: !newBookmarked } : m,
        );
        return { activeConversation: { ...state.activeConversation, messages } };
      });
    }
  },

  pinMessageToContext: async (messageId, options) => {
    const { activeConversationId } = get();
    if (!activeConversationId) return;
    set({ error: null });
    try {
      await apiFetch(`/conversations/${activeConversationId}/messages/${messageId}/pin-context`, {
        method: 'POST',
        body: JSON.stringify({
          replace_pin_id: options?.replacePinId,
          title: options?.title,
        }),
      });
      await get().loadConversation(activeConversationId);
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  updatePinnedContextStatus: async (pinId, status) => {
    const { activeConversationId } = get();
    if (!activeConversationId) return;
    set({ error: null });
    try {
      await apiFetch(`/conversations/${activeConversationId}/pins/${pinId}`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      });
      await get().loadConversation(activeConversationId);
    } catch (e: any) {
      set({ error: e.message });
    }
  },
}));

// --- SSE event handler ---

function handleSSEEvent(
  eventType: string,
  data: any,
  set: (fn: (state: ChatState) => Partial<ChatState>) => void,
  _get: () => ChatState,
  streamConversationId?: string,
) {
  const stringifyPayload = (payload: unknown): string => {
    if (payload == null) return '';
    if (typeof payload === 'string') return payload;
    try {
      return JSON.stringify(payload, null, 2);
    } catch {
      return String(payload);
    }
  };

  // Check if user is viewing the conversation that owns this stream.
  const isViewingStream = !streamConversationId || _get().activeConversationId === streamConversationId;

  switch (eventType) {
    case 'context.usage': {
      if (!isViewingStream) break; // Skip UI update if viewing another conversation
      const evt = data as SSEContextUsage;
      set(() => ({ contextUsage: evt.usage }));
      break;
    }

    case 'context.compacted': {
      if (!isViewingStream) break;
      const evt = data as SSEContextCompacted;
      set(() => ({ latestCompaction: evt.compaction }));
      break;
    }

    case 'context.state.updated': {
      if (!isViewingStream) break;
      const evt = data as SSEContextStateUpdated;
      set((state) => {
        const activeConversation = state.activeConversation;
        if (!activeConversation || activeConversation.conversation_id !== evt.conversation_id) {
          return {};
        }
        return {
          activeConversation: {
            ...activeConversation,
            conversation_context_state: evt.conversation_context_state,
          },
        };
      });
      break;
    }

    case 'message.delta': {
      const evt = data as SSEMessageDelta;
      set((state) => {
        const newContent = state.streaming.contentBuffer + (evt.content || '');
        const newReasoning = state.streaming.reasoningBuffer + (evt.reasoning_content || '');

        // Always update the buffer so content is preserved when switching back.
        const updatedStreaming = {
          ...state.streaming,
          messageId: evt.message_id,
          contentBuffer: newContent,
          reasoningBuffer: newReasoning,
        };

        // Only update activeConversation messages if user is viewing this stream's conversation.
        if (!isViewingStream) {
          return { streaming: updatedStreaming };
        }

        // Determine if streaming render is enabled (conversation > global default)
        const isStreamRender = state.activeConversation?.stream !== false;

        // Update the last message in conversation (assistant placeholder)
        let updatedConversation = state.activeConversation;
        if (updatedConversation) {
          const messages = [...updatedConversation.messages];
          const targetId = evt.message_id || state.streaming.messageId;
          const existingIndex = targetId
            ? messages.findIndex((m) => m.role === 'assistant' && m.message_id === targetId)
            : -1;

          // Content to show in UI: full buffer if streaming, empty placeholder if buffered
          const displayContent = isStreamRender ? newContent : '';
          const displayReasoning = isStreamRender ? (newReasoning || null) : null;

          if (existingIndex >= 0) {
            messages[existingIndex] = {
              ...messages[existingIndex],
              content: displayContent,
              reasoning_content: displayReasoning,
            };
          } else {
            messages.push({
              message_id: evt.message_id,
              role: 'assistant',
              content: displayContent,
              reasoning_content: displayReasoning,
              metadata: {},
              created_at: Date.now() / 1000,
            });
          }

          updatedConversation = { ...updatedConversation, messages };
        }

        return {
          activeConversation: updatedConversation,
          streaming: updatedStreaming,
        };
      });
      break;
    }

    case 'message.finish': {
      // Flush buffered content to UI (for buffered render mode)
      set((state) => {
        let updatedConversation = state.activeConversation;
        const isStreamRender = updatedConversation?.stream !== false;

        // In buffered mode, render the full content now
        if (!isStreamRender && updatedConversation) {
          const messages = [...updatedConversation.messages];
          const evt = data as SSEMessageFinish;
          const targetId = evt.message_id || state.streaming.messageId;
          const existingIndex = targetId
            ? messages.findIndex((m) => m.role === 'assistant' && m.message_id === targetId)
            : -1;
          if (existingIndex >= 0) {
            messages[existingIndex] = {
              ...messages[existingIndex],
              content: state.streaming.contentBuffer,
              reasoning_content: state.streaming.reasoningBuffer || null,
            };
            updatedConversation = { ...updatedConversation, messages };
          }
        }

        return {
          activeConversation: updatedConversation,
        };
      });
      break;
    }

    case 'agent.iteration': {
      if (!isViewingStream) break;
      const evt = data as SSEAgentIteration;
      set((state) => ({
        agentLoopSummary: {
          ...state.agentLoopSummary,
          rounds: Math.max(state.agentLoopSummary.rounds, evt.round_index || 0),
          traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
          status: 'tool_loop',
        },
      }));
      break;
    }

    case 'agent.finalizing': {
      if (!isViewingStream) break;
      const evt = data as { trace_id?: string };
      set((state) => ({
        agentLoopSummary: {
          ...state.agentLoopSummary,
          traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
          status: 'finalizing',
        },
      }));
      break;
    }

    case 'tool_call.start': {
      const evt = data as SSEToolCallStart;
      set((state) => {
        const callId = evt.tool_call_id || '';
        const existingMessageId = callId ? state.streaming.toolMessageIdsByCallId[callId] : undefined;
        const toolMessageId = existingMessageId || `tmp-tool-${callId || Date.now()}`;
        const toolName = evt.requested_tool_name || evt.tool_name || 'tool';
        const argsText = stringifyPayload(evt.arguments);
        const toolCallArguments = evt.arguments == null
          ? ''
          : typeof evt.arguments === 'string'
            ? evt.arguments
            : stringifyPayload(evt.arguments);
        const nextToolMap = callId
          ? { ...state.streaming.toolMessageIdsByCallId, [callId]: toolMessageId }
          : state.streaming.toolMessageIdsByCallId;
        const assistantMessageId = evt.message_id
          || state.streaming.messageId
          || `tmp-assistant-stream-${Date.now()}`;

        if (!isViewingStream || !state.activeConversation) {
          return {
            streaming: {
              ...state.streaming,
              messageId: state.streaming.messageId || assistantMessageId,
              toolMessageIdsByCallId: nextToolMap,
            },
          };
        }

        const messages = [...state.activeConversation.messages];
        const assistantToolCall = {
          id: callId || `tool-call-${toolMessageId}`,
          type: 'function',
          function: {
            name: toolName,
            arguments: toolCallArguments,
          },
        };
        const existingAssistantIndex = messages.findIndex((m) => m.role === 'assistant' && m.message_id === assistantMessageId);
        if (existingAssistantIndex < 0) {
          messages.push({
            message_id: assistantMessageId,
            role: 'assistant',
            content: '',
            reasoning_content: null,
            tool_calls: [assistantToolCall],
            metadata: {},
            created_at: Date.now() / 1000,
          });
        } else {
          const existingAssistant = messages[existingAssistantIndex];
          const existingToolCalls = Array.isArray(existingAssistant.tool_calls)
            ? existingAssistant.tool_calls
            : [];
          const alreadyTracked = existingToolCalls.some((toolCall) => (
            toolCall
            && typeof toolCall === 'object'
            && typeof (toolCall as Record<string, any>).id === 'string'
            && String((toolCall as Record<string, any>).id) === assistantToolCall.id
          ));
          if (!alreadyTracked) {
            messages[existingAssistantIndex] = {
              ...existingAssistant,
              tool_calls: [...existingToolCalls, assistantToolCall],
            };
          }
        }
        const existingIndex = messages.findIndex((m) => m.message_id === toolMessageId);
        const toolMsg: ChatMessage = {
          message_id: toolMessageId,
          role: 'tool',
          content: argsText,
          name: toolName,
          tool_call_id: evt.tool_call_id ?? null,
          metadata: {
            tool_status: 'running',
            round_index: evt.round_index,
            parallel_group_id: evt.parallel_group_id,
            duration_ms: evt.duration_ms,
          },
          created_at: Date.now() / 1000,
        };
        if (existingIndex >= 0) {
          messages[existingIndex] = { ...messages[existingIndex], ...toolMsg };
        } else {
          messages.push(toolMsg);
        }

        return {
          activeConversation: { ...state.activeConversation, messages },
          streaming: {
            ...state.streaming,
            messageId: state.streaming.messageId || assistantMessageId,
            toolMessageIdsByCallId: nextToolMap,
          },
          agentLoopSummary: {
            ...state.agentLoopSummary,
            toolCalls: state.agentLoopSummary.toolCalls + 1,
            traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
            status: 'tool_loop',
          },
        };
      });
      break;
    }

    case 'tool_call.result':
    case 'tool_call.error': {
      const evt = eventType === 'tool_call.result'
        ? (data as SSEToolCallResult)
        : (data as SSEToolCallError);
      set((state) => {
        if (!isViewingStream || !state.activeConversation) {
          return {};
        }

        const callId = evt.tool_call_id || '';
        const mappedToolMessageId = callId ? state.streaming.toolMessageIdsByCallId[callId] : undefined;
        const fallbackToolMessage = state.activeConversation.messages
          .slice()
          .reverse()
          .find((message) => {
            if (message.role !== 'tool') return false;
            if (callId && message.tool_call_id === callId) return true;
            if (evt.tool_name && message.name !== evt.tool_name) return false;
            const rawStatus = String(message.metadata?.tool_status || '').trim().toLowerCase();
            return rawStatus === 'running' || rawStatus === 'pending';
          });
        const toolMessageId = mappedToolMessageId || fallbackToolMessage?.message_id;
        if (!toolMessageId) {
          return {};
        }

        const payload = eventType === 'tool_call.result'
          ? stringifyPayload((evt as SSEToolCallResult).result)
          : stringifyPayload((evt as SSEToolCallError).error);
        const messages = state.activeConversation.messages.map((m) => {
          if (m.message_id !== toolMessageId) return m;
          return {
            ...m,
            name: evt.requested_tool_name ?? evt.tool_name ?? m.name,
            content: payload,
            metadata: {
              ...m.metadata,
              tool_status: eventType === 'tool_call.result' ? 'ok' : 'error',
              parallel_group_id: evt.parallel_group_id ?? m.metadata?.parallel_group_id,
              round_index: evt.round_index ?? m.metadata?.round_index,
              duration_ms: evt.duration_ms ?? m.metadata?.duration_ms,
            },
          };
        });

        return {
          activeConversation: { ...state.activeConversation, messages },
          streaming: {
            ...state.streaming,
            toolMessageIdsByCallId: callId && !mappedToolMessageId
              ? { ...state.streaming.toolMessageIdsByCallId, [callId]: toolMessageId }
              : state.streaming.toolMessageIdsByCallId,
          },
          agentLoopSummary: {
            ...state.agentLoopSummary,
            traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
            status: 'tool_loop',
          },
        };
      });
      break;
    }

    case 'message.complete': {
      const evt = data as SSEMessageComplete;
      set((state) => {
        const traceId = evt.metadata?.trace_id ?? state.agentLoopSummary.traceId;
        const usage = evt.metadata?.usage ?? state.agentLoopSummary.usage;
        const metrics = {
          finish_reason: evt.metadata?.finish_reason,
          budget: evt.metadata?.budget,
          first_token_latency_ms: evt.metadata?.first_token_latency_ms,
          first_token_ms: evt.metadata?.first_token_ms,
          generation_time_ms: evt.metadata?.generation_time_ms,
          decode_tokens_per_second: evt.metadata?.decode_tokens_per_second,
          end_to_end_tokens_per_second: evt.metadata?.end_to_end_tokens_per_second,
          tokens_per_second: evt.metadata?.tokens_per_second,
          tool_loop_convergence_reason: evt.metadata?.tool_loop_convergence_reason,
          tool_loop_final_stream_skipped: evt.metadata?.tool_loop_final_stream_skipped,
          final_stream_status: evt.metadata?.final_stream_status,
          final_stream_error_category: evt.metadata?.final_stream_error_category,
          final_stream_empty_visible_output: evt.metadata?.final_stream_empty_visible_output,
          final_stream_assistant_reasoning_removed_count: evt.metadata?.final_stream_assistant_reasoning_removed_count,
          final_stream_assistant_reasoning_only_removed_count: evt.metadata?.final_stream_assistant_reasoning_only_removed_count,
          final_stream_assistant_tool_call_carrier_count: evt.metadata?.final_stream_assistant_tool_call_carrier_count,
          final_stream_tool_result_projection_count: evt.metadata?.final_stream_tool_result_projection_count,
          request_adapter_class: evt.metadata?.request_adapter_class,
          request_adapter_strict_message_string_contract: evt.metadata?.request_adapter_strict_message_string_contract,
          request_message_count: evt.metadata?.request_message_count,
          request_user_message_count: evt.metadata?.request_user_message_count,
          request_assistant_message_count: evt.metadata?.request_assistant_message_count,
          request_assistant_reasoning_message_count: evt.metadata?.request_assistant_reasoning_message_count,
          request_assistant_reasoning_only_message_count: evt.metadata?.request_assistant_reasoning_only_message_count,
          request_assistant_tool_call_message_count: evt.metadata?.request_assistant_tool_call_message_count,
          request_tool_message_count: evt.metadata?.request_tool_message_count,
        };

        if (!isViewingStream || !state.activeConversation) {
          return {};
        }

        const messages = state.activeConversation.messages.map((m) => {
          if (m.message_id === evt.message_id) {
            return {
              ...m,
              metadata: {
                ...m.metadata,
                ...evt.metadata,
              },
            };
          }
          if (m.role !== 'tool') return m;
          const rawStatus = String(m.metadata?.tool_status || '').trim().toLowerCase();
          if (rawStatus !== 'running' && rawStatus !== 'pending') return m;
          return {
            ...m,
            metadata: {
              ...m.metadata,
              tool_status: 'completed',
            },
          };
        });

        return {
          activeConversation: { ...state.activeConversation, messages },
          error: null,
          streaming: {
            ...state.streaming,
            isStreaming: false,
            suppressVisibleErrors: false,
          },
          agentLoopSummary: {
            ...state.agentLoopSummary,
            traceId,
            usage,
            metrics,
          },
        };
      });
      break;
    }

    case 'message.error': {
      const evt = data as SSEMessageError;
      set((state) => {
        // Remove empty assistant placeholder from UI (LLM failed before producing content)
        let updatedConversation = state.activeConversation;
        let hasVisibleAssistantError = false;
        if (updatedConversation) {
          const messages = updatedConversation.messages;
          const last = messages[messages.length - 1];
          if (last && last.role === 'assistant' && last.content) {
            hasVisibleAssistantError = true;
          }
          if (last && last.role === 'assistant' && !last.content && !last.reasoning_content) {
            updatedConversation = {
              ...updatedConversation,
              messages: messages.slice(0, -1),
            };
          }
        }
        return {
          activeConversation: updatedConversation,
          error: hasVisibleAssistantError
            ? null
            : buildVisibleChatError(
              {
                error: String(evt.error || 'Unknown streaming error'),
                error_code: evt.error_code,
                public_message: evt.public_message,
                retryable: evt.retryable,
                status_code: evt.status_code,
                provider_code: evt.provider_code,
              },
              chatErrorProvider(state.activeConversation?.model),
            ),
          streaming: { ...state.streaming, isStreaming: false },
        };
      });
      break;
    }

    case 'message.aborted': {
      set((state) => ({
        streaming: { ...state.streaming, isStreaming: false },
      }));
      break;
    }
  }
}
