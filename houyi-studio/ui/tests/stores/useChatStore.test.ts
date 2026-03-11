import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useChatStore } from '@/stores/useChatStore';
import type { Conversation } from '@/types/chat';

describe('useChatStore', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    useChatStore.setState({
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
      agentLoopSummary: { rounds: 0, toolCalls: 0, traceId: null, usage: null, metrics: null },
      error: null,
      scrollToMessageId: null,
    });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('restores a streaming assistant when switching back with reasoning buffer only', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'u1',
          role: 'user',
          content: 'hello',
          metadata: {},
          created_at: 1,
        },
      ],
      model: 'demo',
      system_instructions: '',
      temperature: null,
      max_tokens: null,
      top_p: null,
      stream: true,
      bookmarked: false,
      metadata: {},
      created_at: 1,
      updated_at: 1,
      schema_version: 1,
    };

    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(conversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      streaming: {
        ...state.streaming,
        isStreaming: true,
        messageId: 'assistant-1',
        contentBuffer: '',
        reasoningBuffer: 'thinking...',
        streamConversationId: 'conv-1',
        toolMessageIdsByCallId: { call_1: 'tmp-tool-call_1' },
      },
    }));

    await useChatStore.getState().loadConversation('conv-1');

    const activeConversation = useChatStore.getState().activeConversation;
    expect(activeConversation).not.toBeNull();
    expect(activeConversation?.messages).toHaveLength(2);
    expect(activeConversation?.messages[1]).toMatchObject({
      message_id: 'assistant-1',
      role: 'assistant',
      content: '',
      reasoning_content: 'thinking...',
    });
  });

  it('restores an empty streaming assistant when switching back before any text arrives', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'u1',
          role: 'user',
          content: 'hello',
          metadata: {},
          created_at: 1,
        },
      ],
      model: 'demo',
      system_instructions: '',
      temperature: null,
      max_tokens: null,
      top_p: null,
      stream: true,
      bookmarked: false,
      metadata: {},
      created_at: 1,
      updated_at: 1,
      schema_version: 1,
    };

    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(conversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      streaming: {
        ...state.streaming,
        isStreaming: true,
        messageId: 'assistant-1',
        contentBuffer: '',
        reasoningBuffer: '',
        streamConversationId: 'conv-1',
        toolMessageIdsByCallId: {},
      },
    }));

    await useChatStore.getState().loadConversation('conv-1');

    const activeConversation = useChatStore.getState().activeConversation;
    expect(activeConversation).not.toBeNull();
    expect(activeConversation?.messages).toHaveLength(2);
    expect(activeConversation?.messages[1]).toMatchObject({
      message_id: 'assistant-1',
      role: 'assistant',
      content: '',
      reasoning_content: null,
    });
  });

  it('appends a temporary streaming assistant instead of overwriting the previous assistant when messageId is not available yet', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'u1',
          role: 'user',
          content: 'hello',
          metadata: {},
          created_at: 1,
        },
        {
          message_id: 'a-old',
          role: 'assistant',
          content: 'previous answer',
          metadata: {},
          created_at: 2,
        },
      ],
      model: 'demo',
      system_instructions: '',
      temperature: null,
      max_tokens: null,
      top_p: null,
      stream: true,
      bookmarked: false,
      metadata: {},
      created_at: 1,
      updated_at: 2,
      schema_version: 1,
    };

    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(conversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      streaming: {
        ...state.streaming,
        isStreaming: true,
        messageId: null,
        contentBuffer: '',
        reasoningBuffer: 'thinking...',
        streamConversationId: 'conv-1',
        toolMessageIdsByCallId: {},
      },
    }));

    await useChatStore.getState().loadConversation('conv-1');

    const activeConversation = useChatStore.getState().activeConversation;
    expect(activeConversation).not.toBeNull();
    expect(activeConversation?.messages).toHaveLength(3);
    expect(activeConversation?.messages[1]).toMatchObject({
      message_id: 'a-old',
      role: 'assistant',
      content: 'previous answer',
    });
    expect(activeConversation?.messages[2]).toMatchObject({
      role: 'assistant',
      content: '',
      reasoning_content: 'thinking...',
    });
  });

  it('derives agent loop summary without double-counting assistant tool_calls and persisted tool messages', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'u1',
          role: 'user',
          content: 'run tools',
          metadata: {},
          created_at: 1,
        },
        {
          message_id: 'a-carrier',
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call-1',
              function: { name: 'houyi_find_files', arguments: '{"pattern":"*.md"}' },
            },
            {
              id: 'call-2',
              function: { name: 'houyi_read_file', arguments: '{"path":"README.md"}' },
            },
          ],
          metadata: { round_index: 1, trace_id: 'trace-1' },
          created_at: 2,
        },
        {
          message_id: 't1',
          role: 'tool',
          content: '{"matches":["README.md"]}',
          name: 'houyi_find_files',
          tool_call_id: 'call-1',
          metadata: { round_index: 1, trace_id: 'trace-1' },
          created_at: 3,
        },
        {
          message_id: 't2',
          role: 'tool',
          content: '{"content":"# README"}',
          name: 'houyi_read_file',
          tool_call_id: 'call-2',
          metadata: { round_index: 1, trace_id: 'trace-1' },
          created_at: 4,
        },
        {
          message_id: 'a-final',
          role: 'assistant',
          content: 'done',
          metadata: { trace_id: 'trace-1', usage: { total_tokens: 42 } },
          created_at: 5,
        },
      ],
      model: 'demo',
      system_instructions: '',
      temperature: null,
      max_tokens: null,
      top_p: null,
      stream: true,
      bookmarked: false,
      metadata: {},
      created_at: 1,
      updated_at: 5,
      schema_version: 1,
    };

    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(conversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
    }));

    await useChatStore.getState().loadConversation('conv-1');

    expect(useChatStore.getState().agentLoopSummary).toMatchObject({
      rounds: 1,
      toolCalls: 2,
      traceId: 'trace-1',
    });
  });

  it('merges refreshed messages after stream', async () => {
    const loadedConversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'user-1',
          role: 'user',
          content: 'hello',
          metadata: { persisted: true },
          created_at: 1,
        },
        {
          message_id: 'assistant-1',
          role: 'assistant',
          content: 'done',
          metadata: { finish_reason: 'stop' },
          created_at: 2,
        },
      ],
      model: 'demo',
      system_instructions: '',
      temperature: null,
      max_tokens: null,
      top_p: null,
      stream: true,
      bookmarked: false,
      metadata: {},
      created_at: 1,
      updated_at: 2,
      schema_version: 1,
    };

    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(loadedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: {
        ...loadedConversation,
        messages: [
          {
            message_id: 'tmp-1',
            ui_render_id: 'ui-user-1',
            role: 'user',
            content: 'hello',
            metadata: {},
            created_at: 1,
          },
          {
            message_id: 'assistant-1',
            role: 'assistant',
            content: 'done',
            metadata: {},
            created_at: 2,
          },
        ],
      },
    }));

    await useChatStore.getState().loadConversation('conv-1', undefined, true);

    const activeConversation = useChatStore.getState().activeConversation;
    expect(activeConversation?.messages).toHaveLength(2);
    expect(activeConversation?.messages[0]).toMatchObject({
      message_id: 'user-1',
      ui_render_id: 'ui-user-1',
      role: 'user',
      content: 'hello',
      metadata: { persisted: true },
    });
    expect(activeConversation?.messages[1]).toMatchObject({
      message_id: 'assistant-1',
      role: 'assistant',
      content: 'done',
      metadata: { finish_reason: 'stop' },
    });
  });

  it('rejects sending a second message while another response is streaming', async () => {
    const fetchMock = vi.fn();
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: {
        conversation_id: 'conv-1',
        title: 'Chat',
        status: 'active',
        messages: [],
        model: 'demo',
        system_instructions: '',
        temperature: null,
        max_tokens: null,
        top_p: null,
        stream: true,
        bookmarked: false,
        metadata: {},
        created_at: 1,
        updated_at: 1,
        schema_version: 1,
      },
      streaming: {
        ...state.streaming,
        isStreaming: true,
        streamConversationId: 'conv-2',
      },
    }));

    await useChatStore.getState().sendMessage('hello');

    expect(fetchMock).not.toHaveBeenCalled();
    expect(useChatStore.getState().error).toBe(
      'A response is already streaming. Stop it before sending another message.',
    );
  });

  it('rejects regenerating while another response is streaming', async () => {
    const fetchMock = vi.fn();
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: {
        conversation_id: 'conv-1',
        title: 'Chat',
        status: 'active',
        messages: [
          {
            message_id: 'assistant-1',
            role: 'assistant',
            content: 'done',
            metadata: {},
            created_at: 1,
          },
        ],
        model: 'demo',
        system_instructions: '',
        temperature: null,
        max_tokens: null,
        top_p: null,
        stream: true,
        bookmarked: false,
        metadata: {},
        created_at: 1,
        updated_at: 1,
        schema_version: 1,
      },
      streaming: {
        ...state.streaming,
        isStreaming: true,
        streamConversationId: 'conv-2',
      },
    }));

    await useChatStore.getState().regenerateMessage('assistant-1');

    expect(fetchMock).not.toHaveBeenCalled();
    expect(useChatStore.getState().error).toBe(
      'A response is already streaming. Stop it before regenerating another message.',
    );
  });
});
