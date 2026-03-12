import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

type MockChatStore = {
  activeConversation: {
    conversation_id: string;
    messages: Array<{ message_id: string; role: string; content: string; metadata: Record<string, unknown>; created_at: number }>;
  } | null;
  isLoadingConversation: boolean;
  streaming: {
    isStreaming: boolean;
    messageId: string | null;
    streamConversationId: string | null;
  };
  contextUsage: {
    used_tokens: number;
    max_context_tokens: number;
  } | null;
  latestCompaction: {
    trigger: string;
    summary: string;
    metrics: { messages_compacted: number };
  } | null;
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
  ChatTimeline: () => <div data-testid="chat-timeline" />,
}));

vi.mock('@/components/Chat/Composer', () => ({
  Composer: () => <div data-testid="chat-composer" />,
}));

vi.mock('@/components/Chat/TraceDetailPanel', () => ({
  TraceDetailPanel: () => null,
}));

vi.mock('@/components/Chat/AgentLoopSummary', () => ({
  AgentLoopSummary: () => null,
}));

import { ChatPage } from '@/components/Chat/ChatPage';

describe('ChatPage', () => {
  beforeEach(() => {
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
      latestCompaction: null,
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

  it('renders compaction notice when latest compaction is available', () => {
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
          trigger: 'repo_intent_trim',
          summary: 'Kept only the newest repo-scoped turns.',
          metrics: { messages_compacted: 2 },
        },
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
        clearError: vi.fn(),
      };
      return selector ? selector(store) : store;
    });

    render(<ChatPage />);

    expect(screen.getByTestId('compaction-notice')).toBeInTheDocument();
    expect(screen.getByText('Context compacted via repo_intent_trim')).toBeInTheDocument();
    expect(screen.getByText('Kept only the newest repo-scoped turns.')).toBeInTheDocument();
    expect(screen.getByText('2 msgs')).toBeInTheDocument();
  });

  it('does not render compaction notice when no compaction is present', () => {
    render(<ChatPage />);

    expect(screen.queryByTestId('compaction-notice')).not.toBeInTheDocument();
  });
});
