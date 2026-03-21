import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

let latestChatTimelineProps: Record<string, unknown> | null = null;

type MockChatStore = {
  activeConversationId?: string | null;
  activeConversation: {
    conversation_id: string;
    conversation_context_state?: {
      conversation_id: string;
      used_units: number;
      max_units: number;
      state: 'healthy' | 'elevated' | 'near_compaction' | 'compacted_recently';
      last_compacted_at?: number | null;
      last_compaction_delta?: number | null;
      updated_at: number;
    } | null;
    messages: Array<{ message_id: string; role: string; content: string; metadata: Record<string, unknown>; created_at: number }>;
  } | null;
  isLoadingConversation: boolean;
  streaming: {
    isStreaming: boolean;
    messageId: string | null;
    streamConversationId: string | null;
  };
  contextUsage: {
    model?: string;
    used_tokens: number;
    max_context_tokens: number;
    reserved_output_tokens?: number;
    available_tokens?: number;
    available_input_tokens?: number;
    block_breakdown?: Record<string, number>;
    drop_reasons?: Record<string, string>;
  } | null;
  latestCompaction: {
    compaction_id?: string;
    trigger: string;
    summary: string;
    metrics: { messages_compacted: number; tokens_before?: number; tokens_after?: number; pin_violation_count?: number };
  } | null;
  compactionHistory: Array<{
    compaction: {
      compaction_id: string;
      trigger: string;
      summary: string;
      metrics?: { messages_compacted?: number };
    };
    diff: {
      current_message_count: number;
      backup_message_count: number | null;
      current_visible_message_count: number;
      backup_visible_message_count: number | null;
      removed_message_ids: string[];
      added_message_ids: string[];
      source_message_ids: string[];
      source_message_previews?: Array<{
        message_id: string;
        role: string;
        name?: string | null;
        created_at?: number;
        preview: string;
      }>;
      added_message_previews?: Array<{
        message_id: string;
        role: string;
        name?: string | null;
        created_at?: number;
        preview: string;
      }>;
    };
    backup: {
      backup_id: string;
      conversation_id: string;
      trigger: string;
      created_at: number;
      path: string;
      record_id?: string | null;
      metadata: Record<string, unknown>;
    } | null;
  }>;
  isLoadingCompactions: boolean;
  restoringCompactionId: string | null;
  restoringBackupId?: string | null;
  restoreNotice?: { message: string; undoBackupId: string | null; conversationId?: string | null } | null;
  activePins: Array<{
    pin_id: string;
    source_message_id: string;
    title?: string;
    content?: string;
  }>;
  agentLoopSummary: {
    rounds: number;
    toolCalls: number;
    traceId: string | null;
    usage: Record<string, unknown> | null;
    metrics?: Record<string, unknown> | null;
  };
  error: string | null;
  sendMessage: ReturnType<typeof vi.fn>;
  stopStreaming: ReturnType<typeof vi.fn>;
  fetchCompactions: ReturnType<typeof vi.fn>;
  restoreCompaction: ReturnType<typeof vi.fn>;
  restoreBackup?: ReturnType<typeof vi.fn>;
  updatePinnedContextStatus: ReturnType<typeof vi.fn>;
  clearRestoreNotice?: ReturnType<typeof vi.fn>;
  clearError: ReturnType<typeof vi.fn>;
};

const { mockUseChatStore, mockUseSettingsStore } = vi.hoisted(() => ({
  mockUseChatStore: vi.fn(),
  mockUseSettingsStore: vi.fn(),
}));

vi.mock('@/stores/useChatStore', () => ({
  useChatStore: mockUseChatStore,
}));

vi.mock('@/stores/useSettingsStore', () => ({
  useSettingsStore: mockUseSettingsStore,
}));

vi.mock('@/components/Chat/ChatTimeline', () => ({
  ChatTimeline: (props: Record<string, unknown>) => {
    latestChatTimelineProps = props;
    return <div data-testid="chat-timeline" />;
  },
}));

vi.mock('@/components/Chat/Composer', () => ({
  Composer: () => <div data-testid="chat-composer" />,
}));

vi.mock('@/components/Chat/TraceDetailPanel', () => ({
  TraceDetailPanel: () => null,
}));

import { ChatPage } from '@/components/Chat/ChatPage';

const compactionDismissStorageKey = 'houyi.chat.compactionNoticeDismissals';

describe('ChatPage', () => {
  beforeEach(() => {
    latestChatTimelineProps = null;
    if (typeof window.localStorage?.removeItem === 'function') {
      window.localStorage.removeItem(compactionDismissStorageKey);
    }
    const store: MockChatStore = {
      activeConversationId: 'conv-1',
      activeConversation: {
        conversation_id: 'conv-1',
        conversation_context_state: {
          conversation_id: 'conv-1',
          used_units: 1200,
          max_units: 272000,
          state: 'healthy',
          updated_at: 1,
        },
        messages: [
          { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
        ],
      },
      isLoadingConversation: false,
      streaming: {
        isStreaming: false,
        messageId: null,
        streamConversationId: null,
      },
      contextUsage: null,
      latestCompaction: null,
      compactionHistory: [],
      isLoadingCompactions: false,
      restoringCompactionId: null,
      restoringBackupId: null,
      restoreNotice: null,
      activePins: [],
      agentLoopSummary: {
        rounds: 0,
        toolCalls: 0,
        traceId: null,
        usage: null,
        metrics: null,
      },
      error: null,
      sendMessage: vi.fn(),
      stopStreaming: vi.fn(),
      fetchCompactions: vi.fn(),
      restoreCompaction: vi.fn(),
      restoreBackup: vi.fn(),
      updatePinnedContextStatus: vi.fn(),
      clearRestoreNotice: vi.fn(),
      clearError: vi.fn(),
    };

    mockUseChatStore.mockReset();
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => (
      selector ? selector(store) : store
    ));

    mockUseSettingsStore.mockReset();
    mockUseSettingsStore.mockImplementation((selector?: (state: { fetchSettings: ReturnType<typeof vi.fn> }) => unknown) => {
      const settingsStore = { fetchSettings: vi.fn() };
      return selector ? selector(settingsStore) : settingsStore;
    });
  });

  it('only shows top restore notice while undo is still available', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversationId: 'conv-1',
        activeConversation: {
          conversation_id: 'conv-1',
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: null,
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: {
          message: 'Undo restore completed. Returned to the snapshot you were viewing before the restore.',
          undoBackupId: null,
        },
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.queryByTestId('restore-notice')).not.toBeInTheDocument();
  });

  it('hides the top restore notice as soon as the selected conversation changes', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore & { activeConversationId: string | null }) => unknown) => {
      const store = {
        activeConversationId: 'conv-2',
        activeConversation: {
          conversation_id: 'conv-1',
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: null,
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: {
          message: 'Restored snapshot. You can undo this restore from the previous state backup.',
          undoBackupId: 'backup-undo-1',
          conversationId: 'conv-1',
        },
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.queryByTestId('restore-notice')).not.toBeInTheDocument();
  });

  it('renders compaction notice', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: null,
        latestCompaction: {
          compaction_id: 'cmp-1',
          trigger: 'repo_intent_trim',
          summary: 'Kept only the newest repo-scoped turns.',
          metrics: { messages_compacted: 2, tokens_before: 4000, tokens_after: 2800, pin_violation_count: 0 },
        },
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.getByTestId('compaction-notice')).toBeInTheDocument();
    expect(screen.getByText('Trimmed request context before send')).toBeInTheDocument();
    expect(screen.getByText('Latest save 1,200 tokens')).toBeInTheDocument();
    expect(screen.getByTestId('compaction-saved-badge')).toHaveAttribute('title', 'Latest compaction: 4,000 → 2,800 tokens.');
    expect(screen.getByText('Pins protected')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Details' })).toBeInTheDocument();
    expect(screen.queryByText('Kept only the newest repo-scoped turns.')).not.toBeInTheDocument();
    expect(screen.queryByText('2 msgs')).not.toBeInTheDocument();
  });

  it('renders latest compaction delta with reduced rolling usage', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversationId: 'conv-1',
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 271995,
            max_units: 272000,
            state: 'compacted_recently',
            last_compacted_at: 1710000000,
            last_compaction_delta: 5,
            updated_at: 1710000001,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: null,
        latestCompaction: {
          compaction_id: 'cmp-1',
          trigger: 'overflow_recovery',
          summary: 'Recovered context budget',
          metrics: {
            messages_compacted: 1,
            tokens_before: 272000,
            tokens_after: 271995,
            pin_violation_count: 0,
          },
        },
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.getByText('Recovered rolling context capacity')).toBeInTheDocument();
    expect(screen.getByText('Latest save 5 tokens')).toBeInTheDocument();
    expect(screen.getByText('271,995 / 272,000')).toBeInTheDocument();
    expect(screen.getByTestId('conversation-state-indicator')).toHaveClass('bg-red-400');
  });

  it('dismisses compaction notice', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: null,
        latestCompaction: {
          compaction_id: 'cmp-1',
          trigger: 'pre_request_pressure',
          summary: 'Prepared active context without dropping pinned content.',
          metrics: { messages_compacted: 1, tokens_before: 4096, tokens_after: 2784, pin_violation_count: 0 },
        },
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    const { rerender } = render(<ChatPage />);
    expect(screen.getByTestId('compaction-notice')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByTestId('compaction-notice')).not.toBeInTheDocument();

    rerender(<ChatPage />);
    expect(screen.queryByTestId('compaction-notice')).not.toBeInTheDocument();
  });

  it('passes live messages to ChatTimeline while the active conversation is streaming', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
            { message_id: 'a1', role: 'assistant', content: 'partial', metadata: {}, created_at: 2 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: true,
          messageId: 'a1',
          streamConversationId: 'conv-1',
        },
        contextUsage: null,
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(latestChatTimelineProps).toMatchObject({
      messages: [
        { message_id: 'u1', role: 'user', content: 'hello' },
        { message_id: 'a1', role: 'assistant', content: 'partial' },
      ],
      streamingMessageId: 'a1',
      isWaitingForResponse: false,
      conversationId: 'conv-1',
    });
  });

  it('hides compaction notice without data', () => {
    render(<ChatPage />);

    expect(screen.queryByTestId('compaction-notice')).not.toBeInTheDocument();
  });

  it('renders compact top rail', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 125306,
            max_units: 272000,
            state: 'elevated',
            updated_at: 1,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: {
          used_tokens: 3200,
          max_context_tokens: 8000,
          block_breakdown: {
            current_turn: 300,
            recent: 1900,
            pinned: 600,
          },
          drop_reasons: {
            older_summary: 'budget_exceeded',
            memory: 'boundary_excluded',
          },
        },
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: 'trace-1',
          usage: {
            prompt_tokens: 1200,
            completion_tokens: 180,
            cached_prompt_tokens: 240,
            cached_prompt_tokens_reported: true,
            usage_confidence: 'reported',
          },
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.getByTestId('chat-top-rail')).toBeInTheDocument();
    expect(screen.getByTestId('conversation-summary')).toBeInTheDocument();
    expect(screen.getByText('Rolling Context')).toBeInTheDocument();
    expect(screen.getByText('125,306 / 272,000')).toBeInTheDocument();
    expect(screen.getByTestId('conversation-state-indicator')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Inspect' })).toBeEnabled();
    expect(screen.queryByText('2 dropped')).not.toBeInTheDocument();
    expect(screen.queryByText('Last Request Plan')).not.toBeInTheDocument();
    expect(screen.queryByText('Trace')).not.toBeInTheDocument();
  });

  it('keeps request-specific cache status out of the top rail', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 125306,
            max_units: 272000,
            state: 'elevated',
            updated_at: 1,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: {
          used_tokens: 3200,
          max_context_tokens: 8000,
          block_breakdown: {
            current_turn: 300,
            recent: 1900,
          },
          drop_reasons: {
            older_summary: 'budget_exceeded',
          },
        },
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: 'trace-1',
          usage: {
            prompt_tokens: 1200,
            completion_tokens: 180,
            cached_prompt_tokens: 0,
            cached_prompt_tokens_reported: true,
            cache_hit: false,
            cache_hit_reported: true,
            usage_confidence: 'reported',
          },
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.getByTestId('conversation-summary')).toBeInTheDocument();
    expect(screen.queryByText('Context trimmed · Cache hit')).not.toBeInTheDocument();
    expect(screen.queryByText('Context trimmed · No cache hit')).not.toBeInTheDocument();
  });

  it('keeps top rail summary cards shrinkable in narrow layouts', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversationId: 'conv-1',
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 228,
            max_units: 272000,
            state: 'healthy',
            updated_at: 1,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: {
          used_tokens: 246,
          max_context_tokens: 131072,
          block_breakdown: {},
          drop_reasons: {},
        },
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: {
            cache_hit: false,
            cache_hit_reported: true,
          },
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.getByTestId('chat-top-rail').firstElementChild).toHaveClass('min-w-0');
    expect(screen.getByTestId('conversation-summary')).toHaveClass('min-w-0');
  });

  it('keeps the top rail focused on rolling context even when request capacity differs', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversationId: 'conv-1',
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 504,
            max_units: 272000,
            state: 'healthy',
            updated_at: 1,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: {
          model: 'minimax-m1',
          used_tokens: 16163,
          max_context_tokens: 1000000,
          block_breakdown: {},
          drop_reasons: {},
        },
        latestCompaction: {
          compaction_id: 'cmp-1',
          trigger: 'manual',
          summary: 'trimmed old context',
          metrics: {
            messages_compacted: 6,
            tokens_before: 800000,
            tokens_after: 72089,
          },
        },
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: {
            cache_hit: true,
            cache_hit_reported: true,
          },
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(within(screen.getByTestId('conversation-summary')).getByText('504 / 272,000')).toBeInTheDocument();
    expect(screen.queryByText('minimax-m1')).not.toBeInTheDocument();
    expect(screen.queryByText('16,163 / 1,000,000')).not.toBeInTheDocument();
    expect(screen.getByText('Latest save 727,911 tokens')).toBeInTheDocument();
  });

  it('shows inspect as the only this-turn details entry point', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 125306,
            max_units: 272000,
            state: 'elevated',
            updated_at: 1,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: {
          used_tokens: 3200,
          max_context_tokens: 8000,
          reserved_output_tokens: 1024,
          available_tokens: 3776,
          available_input_tokens: 3576,
          block_breakdown: {
            system: 186,
            pinned: 420,
            current_turn: 78,
            summary: 640,
          },
          drop_reasons: {
            older_summary: 'budget_exceeded',
            memory: 'boundary_excluded',
          },
        },
        latestCompaction: {
          trigger: 'repo_intent_trim',
          summary: 'Kept only the newest repo-scoped turns.',
          metrics: {
            messages_compacted: 2,
            tokens_before: 3840,
            tokens_after: 2560,
            pin_violation_count: 0,
          },
        },
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: 'trace-1',
          usage: {
            prompt_tokens: 1200,
            completion_tokens: 180,
            reasoning_tokens: 120,
            reasoning_tokens_reported: true,
            answer_tokens: 60,
            answer_tokens_reported: true,
            cached_prompt_tokens: 240,
            cached_prompt_tokens_reported: true,
            cache_hit: true,
            cache_hit_reported: true,
            usage_confidence: 'reported',
          },
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.getByRole('button', { name: 'Inspect' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Plan' })).not.toBeInTheDocument();
    expect(screen.queryByTestId('last-request-plan-details')).not.toBeInTheDocument();
  });

  it('opens chat inspector drawer', () => {
    const fetchCompactions = vi.fn();
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 125306,
            max_units: 272000,
            state: 'elevated',
            updated_at: 1,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: {
          used_tokens: 3200,
          max_context_tokens: 8000,
          reserved_output_tokens: 1024,
          available_tokens: 4792,
          available_input_tokens: 4575,
          block_breakdown: {
            system: 186,
            pinned: 420,
            current_turn: 78,
            summary: 640,
          },
          drop_reasons: {
            older_summary: 'budget_exceeded',
            memory: 'boundary_excluded',
          },
        },
        latestCompaction: {
          compaction_id: 'cmp-1',
          trigger: 'repo_intent_trim',
          summary: 'Kept only the newest repo-scoped turns.',
          metrics: { messages_compacted: 2, tokens_before: 3840, tokens_after: 2560, pin_violation_count: 0 },
        },
        compactionHistory: [
          {
            compaction: {
              compaction_id: 'cmp-1',
              trigger: 'repo_intent_trim',
              summary: 'Kept only the newest repo-scoped turns.',
              metrics: { messages_compacted: 2 },
            },
            diff: {
              current_message_count: 5,
              backup_message_count: 7,
              current_visible_message_count: 5,
              backup_visible_message_count: 7,
              removed_message_ids: ['u1', 'u2'],
              added_message_ids: ['a3'],
              source_message_ids: ['u1', 'u2'],
              source_message_previews: [
                {
                  message_id: 'u1',
                  role: 'user',
                  preview: 'Removed preview one',
                },
                {
                  message_id: 'u2',
                  role: 'assistant',
                  preview: 'Removed preview two',
                },
                {
                  message_id: 'u3',
                  role: 'tool',
                  preview: 'Removed preview three',
                },
              ],
              added_message_previews: [
                {
                  message_id: 'a3',
                  role: 'assistant',
                  preview: 'Added preview one',
                },
              ],
            },
            backup: {
              backup_id: 'backup-1',
              conversation_id: 'conv-1',
              trigger: 'repo_intent_trim',
              created_at: 1,
              path: 'conv-1--backup-1.json',
              record_id: 'cmp-1',
              metadata: {},
            },
          },
        ],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [
          {
            pin_id: 'pin-1',
            source_message_id: 'u1',
            title: 'Pinned rule',
            content: 'Deploy to staging first.',
          },
        ],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: {
            prompt_tokens: 2593,
            completion_tokens: 1024,
            reasoning_tokens: 0,
            reasoning_tokens_reported: false,
            answer_tokens: 1024,
            answer_tokens_reported: false,
            cached_prompt_tokens: 0,
            cached_prompt_tokens_reported: false,
            cache_hit: false,
            cache_hit_reported: false,
            usage_confidence: 'reported',
          },
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions,
        restoreCompaction: vi.fn(),
        restoreBackup: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));

    const detailsPanel = screen.getByTestId('chat-inspector-panel');
    expect(detailsPanel).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Chat Inspector')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Conversation Context')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Request Context')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Token Accounting')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Compaction')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Compaction history')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Latest compaction details')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Show trimmed details')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Trimmed to fit request budget')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('The request hit budget limits before this block could be included.')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Excluded by planning boundary')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('This block was outside the active planning boundary for the current request.')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Removed preview')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('1 more preview item(s)')).toBeInTheDocument();
    expect(within(detailsPanel).getByText(/Full snapshot content is preserved in backup/i)).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Active pins')).toBeInTheDocument();
    expect(within(detailsPanel).getByText(/Add or replace pins from a message's pin menu/i)).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Deploy to staging first.')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Protection')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('2,593')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Provider reported overall usage')).toBeInTheDocument();
    expect(within(detailsPanel).getAllByText('Not reported').length).toBeGreaterThan(0);
    expect(within(detailsPanel).getByText('Unknown')).toBeInTheDocument();
    expect(within(detailsPanel).getByText('Pins protected')).toBeInTheDocument();

    fireEvent.click(within(detailsPanel).getByRole('button', { name: 'Show trimmed details' }));
    expect(within(detailsPanel).getByText('Omitted context block')).toBeInTheDocument();
    expect(within(detailsPanel).getByText(/Internal reference: older_summary/i)).toBeInTheDocument();

    expect(fetchCompactions).toHaveBeenCalledWith('conv-1');
  });

  it('shows pinned active status in top rail', () => {
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversation: {
          conversation_id: 'conv-1',
          conversation_context_state: {
            conversation_id: 'conv-1',
            used_units: 1000,
            max_units: 272000,
            state: 'healthy',
            updated_at: 1,
          },
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: {
          used_tokens: 1200,
          max_context_tokens: 8000,
          block_breakdown: {
            pinned: 300,
            current_turn: 120,
          },
          drop_reasons: {},
        },
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: null,
        activePins: [
          {
            pin_id: 'pin-1',
            source_message_id: 'u1',
            title: 'Pinned rule',
            content: 'Deploy to staging first.',
          },
        ],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        updatePinnedContextStatus: vi.fn(),
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);
    expect(screen.getByTestId('conversation-summary')).toBeInTheDocument();
    expect(screen.queryByText('1 pinned')).not.toBeInTheDocument();
  });

  it('shows restore notice with undo action', () => {
    const restoreBackup = vi.fn();
    const clearRestoreNotice = vi.fn();
    mockUseChatStore.mockImplementation((selector?: (state: MockChatStore) => unknown) => {
      const store: MockChatStore = {
        activeConversationId: 'conv-1',
        activeConversation: {
          conversation_id: 'conv-1',
          messages: [
            { message_id: 'u1', role: 'user', content: 'hello', metadata: {}, created_at: 1 },
          ],
        },
        isLoadingConversation: false,
        streaming: {
          isStreaming: false,
          messageId: null,
          streamConversationId: null,
        },
        contextUsage: null,
        latestCompaction: null,
        compactionHistory: [],
        isLoadingCompactions: false,
        restoringCompactionId: null,
        restoringBackupId: null,
        restoreNotice: {
          message: 'Restored snapshot. You can undo this restore from the previous state backup.',
          undoBackupId: 'backup-undo-1',
          conversationId: 'conv-1',
        },
        activePins: [],
        agentLoopSummary: {
          rounds: 0,
          toolCalls: 0,
          traceId: null,
          usage: null,
          metrics: null,
        },
        error: null,
        sendMessage: vi.fn(),
        stopStreaming: vi.fn(),
        fetchCompactions: vi.fn(),
        restoreCompaction: vi.fn(),
        restoreBackup,
        updatePinnedContextStatus: vi.fn(),
        clearRestoreNotice,
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    const notice = screen.getByTestId('restore-notice');
    fireEvent.click(within(notice).getByRole('button', { name: 'Undo restore' }));
    expect(restoreBackup).toHaveBeenCalledWith('backup-undo-1');
    fireEvent.click(within(notice).getByRole('button', { name: 'Dismiss' }));
    expect(clearRestoreNotice).toHaveBeenCalled();
  });
});
