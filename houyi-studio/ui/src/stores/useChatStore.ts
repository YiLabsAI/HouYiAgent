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
  CreateConversationRequest,
  SendMessageRequest,
  SSEAgentIteration,
  SSEMessageDelta,
  SSEMessageFinish,
  SSEMessageComplete,
  SSEMessageError,
  SSEContextUsage,
  SSEToolCallError,
  SSEToolCallResult,
  SSEToolCallStart,
} from '@/types/chat';

// --- API helpers ---

const API_BASE = '/api/chat';

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
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
  // Which conversation owns this stream.  Used to resume UI when switching back.
  streamConversationId: string | null;
  toolMessageIdsByCallId: Record<string, string>;
}

interface AgentLoopSummary {
  rounds: number;
  toolCalls: number;
  traceId: string | null;
  usage: Record<string, any> | null;
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

  // Agent loop summary for current/last streamed assistant response
  agentLoopSummary: AgentLoopSummary;

  // Error
  error: string | null;

  // Search navigation: scroll to a specific message after loading
  scrollToMessageId: string | null;

  // Actions
  fetchConversations: () => Promise<void>;
  createConversation: (req?: CreateConversationRequest) => Promise<string>;
  loadConversation: (conversationId: string, scrollToMessageId?: string) => Promise<void>;
  updateConversation: (conversationId: string, updates: { title?: string; status?: string; model?: string; system_instructions?: string; temperature?: number | null; max_tokens?: number | null; top_p?: number | null; stream?: boolean | null; bookmarked?: boolean }) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  sendMessage: (content: string, options?: Partial<SendMessageRequest>, files?: File[]) => Promise<void>;
  stopStreaming: () => void;
  clearError: () => void;
  clearScrollTarget: () => void;

  // Message operations
  editMessage: (messageId: string, content: string) => Promise<void>;
  deleteMessage: (messageId: string) => Promise<void>;
  regenerateMessage: (messageId: string) => Promise<void>;
  toggleMessageBookmark: (messageId: string) => Promise<void>;
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
  },
  contextUsage: null,
  agentLoopSummary: { rounds: 0, toolCalls: 0, traceId: null, usage: null },
  error: null,
  scrollToMessageId: null,

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

  loadConversation: async (conversationId: string, scrollToMessageId?: string) => {
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
      const [data, usageResult] = await Promise.all([
        apiFetch<Conversation>(`/conversations/${conversationId}`),
        apiFetch<{ usage: any }>(`/conversations/${conversationId}/context-usage`).catch(() => null),
      ]);

      // Stale-response guard: user clicked another conversation while fetching
      if (get().activeConversationId !== conversationId) return;

      // If the SSE stream is still running for this conversation, inject the
      // in-progress assistant message from the buffer so the user sees the
      // partial content immediately when switching back.
      const { streaming } = get();
      let conversationData = data;
      if (
        streaming.isStreaming &&
        streaming.streamConversationId === conversationId &&
        streaming.contentBuffer
      ) {
        const messages = [...data.messages];
        const lastMsg = messages[messages.length - 1];
        const isStreamRender = data.stream !== false;
        const displayContent = isStreamRender ? streaming.contentBuffer : '';
        const displayReasoning = isStreamRender ? (streaming.reasoningBuffer || null) : null;

        if (lastMsg && lastMsg.role === 'assistant') {
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

      // Single atomic set — ChatPage renders exactly once with the new data
      set({
        activeConversation: conversationData,
        isLoadingConversation: false,
        contextUsage: usageResult?.usage ?? null,
        scrollToMessageId: scrollToMessageId ?? null,
      });
    } catch (e: any) {
      if (get().activeConversationId !== conversationId) return;
      set({ error: e.message, isLoadingConversation: false, activeConversation: null });
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
    const { activeConversationId, activeConversation } = get();
    if (!activeConversationId || !activeConversation) {
      set({ error: 'No active conversation' });
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

    // Optimistically add user message to UI
    const userMsg: ChatMessage = {
      message_id: `tmp-${Date.now()}`,
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
      streaming: {
        isStreaming: true,
        messageId: null,
        contentBuffer: '',
        reasoningBuffer: '',
        abortController,
        streamConversationId: activeConversationId,
        toolMessageIdsByCallId: {},
      },
      agentLoopSummary: { rounds: 0, toolCalls: 0, traceId: null, usage: null },
      error: null,
    }));

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
          } catch {
            // Skip malformed events
          }
        }
        eventType = '';
        eventData = '';
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

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

      // Reload conversation to get persisted state + refresh sidebar counts.
      // Only reload the UI if the user is still viewing this conversation;
      // otherwise just refresh the sidebar list so message counts update.
      if (get().activeConversationId === streamConversationId) {
        await get().loadConversation(streamConversationId);
      }
      await get().fetchConversations();
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User-initiated abort — expected
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
      },
    }));
  },

  clearError: () => set({ error: null }),

  clearScrollTarget: () => set({ scrollToMessageId: null }),

  // --- Message operations ---

  editMessage: async (messageId: string, content: string) => {
    const { activeConversationId } = get();
    if (!activeConversationId) return;
    set({ error: null });
    try {
      await apiFetch(`/conversations/${activeConversationId}/messages/${messageId}`, {
        method: 'PUT',
        body: JSON.stringify({ content }),
      });
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
      await apiFetch(`/conversations/${activeConversationId}/messages/${messageId}`, {
        method: 'DELETE',
      });
      await get().loadConversation(activeConversationId);
      await get().fetchConversations();
    } catch (e: any) {
      // Revert on failure
      set({ error: e.message });
      await get().loadConversation(activeConversationId);
    }
  },

  regenerateMessage: async (messageId: string) => {
    const { activeConversationId, activeConversation } = get();
    if (!activeConversationId || !activeConversation) return;

    const abortController = new AbortController();
    set(() => ({
      streaming: {
        isStreaming: true,
        messageId: null,
        contentBuffer: '',
        reasoningBuffer: '',
        abortController,
        streamConversationId: activeConversationId,
        toolMessageIdsByCallId: {},
      },
      agentLoopSummary: { rounds: 0, toolCalls: 0, traceId: null, usage: null },
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

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

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

      if (get().activeConversationId === activeConversationId) {
        await get().loadConversation(activeConversationId);
      }
      await get().fetchConversations();
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
          streaming: { ...state.streaming, isStreaming: false },
        };
      });
      break;
    }

    case 'agent.iteration': {
      const evt = data as SSEAgentIteration;
      set((state) => ({
        agentLoopSummary: {
          ...state.agentLoopSummary,
          rounds: Math.max(state.agentLoopSummary.rounds, evt.round_index || 0),
          traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
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
        const toolName = evt.tool_name || 'tool';
        const argsText = stringifyPayload(evt.arguments);
        const nextToolMap = callId
          ? { ...state.streaming.toolMessageIdsByCallId, [callId]: toolMessageId }
          : state.streaming.toolMessageIdsByCallId;

        if (!isViewingStream || !state.activeConversation) {
          return {
            streaming: { ...state.streaming, toolMessageIdsByCallId: nextToolMap },
            agentLoopSummary: {
              ...state.agentLoopSummary,
              toolCalls: state.agentLoopSummary.toolCalls + 1,
              traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
            },
          };
        }

        const messages = [...state.activeConversation.messages];
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
          streaming: { ...state.streaming, toolMessageIdsByCallId: nextToolMap },
          agentLoopSummary: {
            ...state.agentLoopSummary,
            toolCalls: state.agentLoopSummary.toolCalls + 1,
            traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
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
          return {
            agentLoopSummary: {
              ...state.agentLoopSummary,
              traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
            },
          };
        }

        const callId = evt.tool_call_id || '';
        const toolMessageId = callId ? state.streaming.toolMessageIdsByCallId[callId] : undefined;
        if (!toolMessageId) {
          return {
            agentLoopSummary: {
              ...state.agentLoopSummary,
              traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
            },
          };
        }

        const payload = eventType === 'tool_call.result'
          ? stringifyPayload((evt as SSEToolCallResult).result)
          : stringifyPayload((evt as SSEToolCallError).error);
        const messages = state.activeConversation.messages.map((m) => {
          if (m.message_id !== toolMessageId) return m;
          return {
            ...m,
            content: payload,
            metadata: {
              ...m.metadata,
              tool_status: eventType === 'tool_call.result' ? 'ok' : 'error',
            },
          };
        });

        return {
          activeConversation: { ...state.activeConversation, messages },
          agentLoopSummary: {
            ...state.agentLoopSummary,
            traceId: evt.trace_id ?? state.agentLoopSummary.traceId,
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

        if (!isViewingStream || !state.activeConversation) {
          return {
            agentLoopSummary: {
              ...state.agentLoopSummary,
              traceId,
              usage,
            },
          };
        }

        const messages = state.activeConversation.messages.map((m) => {
          if (m.message_id !== evt.message_id) return m;
          return {
            ...m,
            metadata: {
              ...m.metadata,
              ...evt.metadata,
            },
          };
        });

        return {
          activeConversation: { ...state.activeConversation, messages },
          agentLoopSummary: {
            ...state.agentLoopSummary,
            traceId,
            usage,
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
        if (updatedConversation) {
          const messages = updatedConversation.messages;
          const last = messages[messages.length - 1];
          if (last && last.role === 'assistant' && !last.content && !last.reasoning_content) {
            updatedConversation = {
              ...updatedConversation,
              messages: messages.slice(0, -1),
            };
          }
        }
        return {
          activeConversation: updatedConversation,
          error: `LLM Error: ${evt.error}`,
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
