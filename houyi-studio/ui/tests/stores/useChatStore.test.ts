import { waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useChatStore } from '@/stores/useChatStore';
import type { ChatMessage, Conversation } from '@/types/chat';

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
        suppressVisibleErrors: false,
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
        status: 'idle',
      },
      error: null,
      scrollToMessageId: null,
      composerUiByConversation: {},
    });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('restores streaming assistant from reasoning buffer', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'u1',
          role: 'user',
          content: 'search the web',
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

  it('restores empty streaming assistant before text arrives', async () => {
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

  it('appends temporary assistant before message id', async () => {
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

  it('derives agent loop summary without double count', async () => {
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

  it('hydrates finish reason from assistant metadata', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'user-1',
          role: 'user',
          content: 'hello',
          metadata: {},
          created_at: 1,
        },
        {
          message_id: 'assistant-1',
          role: 'assistant',
          content: 'done',
          metadata: {
            trace_id: 'trace-finish',
            finish_reason: 'length',
            generation_time_ms: 321,
          },
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
    }));

    await useChatStore.getState().loadConversation('conv-1');

    expect(useChatStore.getState().agentLoopSummary).toMatchObject({
      traceId: 'trace-finish',
      metrics: {
        finish_reason: 'length',
        generation_time_ms: 321,
      },
    });
  });

  it('hydrates legacy first token and decode throughput metrics from assistant metadata', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'user-1',
          role: 'user',
          content: 'hello',
          metadata: {},
          created_at: 1,
        },
        {
          message_id: 'assistant-1',
          role: 'assistant',
          content: 'done',
          metadata: {
            trace_id: 'trace-legacy-metrics',
            first_token_ms: 411,
            decode_tokens_per_second: 73,
          },
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
    }));

    await useChatStore.getState().loadConversation('conv-1');

    expect(useChatStore.getState().agentLoopSummary).toMatchObject({
      traceId: 'trace-legacy-metrics',
      metrics: {
        first_token_ms: 411,
        decode_tokens_per_second: 73,
      },
    });
  });

  it('hydrates final stream phase metrics from assistant metadata', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'user-1',
          role: 'user',
          content: 'hello',
          metadata: {},
          created_at: 1,
        },
        {
          message_id: 'assistant-1',
          role: 'assistant',
          content: '',
          reasoning_content: 'thinking only',
          metadata: {
            trace_id: 'trace-final-stream',
            tool_loop_convergence_reason: 'needs_final_stream',
            tool_loop_final_stream_skipped: false,
            request_adapter_class: 'SiliconFlowAdapter',
            request_adapter_strict_message_string_contract: true,
            request_message_count: 4,
            request_user_message_count: 1,
            request_assistant_message_count: 2,
            request_assistant_reasoning_message_count: 1,
            request_assistant_reasoning_only_message_count: 1,
            request_assistant_tool_call_message_count: 1,
            request_tool_message_count: 1,
            final_stream_status: 'error',
            final_stream_error_category: 'timeout',
            final_stream_empty_visible_output: false,
            final_stream_assistant_reasoning_removed_count: 2,
            final_stream_assistant_reasoning_only_removed_count: 1,
            final_stream_assistant_tool_call_carrier_count: 1,
            final_stream_tool_result_projection_count: 1,
            finish_reason: 'error',
          },
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
    }));

    await useChatStore.getState().loadConversation('conv-1');

    expect(useChatStore.getState().agentLoopSummary).toMatchObject({
      traceId: 'trace-final-stream',
      metrics: {
        finish_reason: 'error',
        tool_loop_convergence_reason: 'needs_final_stream',
        tool_loop_final_stream_skipped: false,
        request_adapter_class: 'SiliconFlowAdapter',
        request_adapter_strict_message_string_contract: true,
        request_message_count: 4,
        request_user_message_count: 1,
        request_assistant_message_count: 2,
        request_assistant_reasoning_message_count: 1,
        request_assistant_reasoning_only_message_count: 1,
        request_assistant_tool_call_message_count: 1,
        request_tool_message_count: 1,
        final_stream_status: 'error',
        final_stream_error_category: 'timeout',
        final_stream_empty_visible_output: false,
        final_stream_assistant_reasoning_removed_count: 2,
        final_stream_assistant_reasoning_only_removed_count: 1,
        final_stream_assistant_tool_call_carrier_count: 1,
        final_stream_tool_result_projection_count: 1,
      },
    });
  });

  it('hydrates latest compaction from history', async () => {
    const conversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'user-1',
          role: 'user',
          content: 'hello repo',
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
      metadata: {
        compaction_history: [
          {
            compaction_id: 'cmp-1',
            trigger: 'repo_intent_trim',
            summary: 'Dropped older repo setup messages.',
            source_message_ids: ['u1'],
            pinned_message_ids: [],
            retained_refs: [],
            metrics: { messages_compacted: 1, tokens_before: 1200, tokens_after: 400 },
            created_at: 1,
            metadata: { reason: 'repo_intent_recent_window' },
          },
          {
            compaction_id: 'cmp-2',
            trigger: 'repo_intent_trim',
            summary: 'Kept only the newest repo-scoped turns.',
            source_message_ids: ['u2', 'u3'],
            pinned_message_ids: [],
            retained_refs: [],
            metrics: { messages_compacted: 2, tokens_before: 2000, tokens_after: 700 },
            created_at: 2,
            metadata: { reason: 'repo_intent_recent_window' },
          },
        ],
      },
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
    }));

    await useChatStore.getState().loadConversation('conv-1');

    expect(useChatStore.getState().latestCompaction).toMatchObject({
      compaction_id: 'cmp-2',
      trigger: 'repo_intent_trim',
      summary: 'Kept only the newest repo-scoped turns.',
      metrics: {
        messages_compacted: 2,
        tokens_before: 2000,
        tokens_after: 700,
      },
    });
  });


  it('loads compaction history on load', async () => {
    const conversation: Conversation = {
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
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({
          items: [
            {
              compaction: {
                compaction_id: 'cmp-1',
                trigger: 'manual',
                summary: 'Compacted old history',
                source_message_ids: ['u1', 'u2'],
                pinned_message_ids: [],
                retained_refs: [],
                metrics: { messages_compacted: 2, tokens_before: 2400, tokens_after: 1200 },
                created_at: 2,
                metadata: {},
              },
              backup: {
                backup_id: 'backup-1',
                conversation_id: 'conv-1',
                trigger: 'manual',
                created_at: 2,
                path: 'conv-1--backup-1.json',
                record_id: 'cmp-1',
                metadata: {},
              },
              diff: {
                source_message_ids: ['u1', 'u2'],
                backup_message_count: 7,
                current_message_count: 5,
                backup_visible_message_count: 7,
                current_visible_message_count: 5,
                removed_message_ids: ['u1', 'u2'],
                added_message_ids: [],
              },
            },
          ],
        }), {
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

    expect(useChatStore.getState().compactionHistory).toHaveLength(1);
    expect(useChatStore.getState().compactionHistory[0]?.diff.removed_message_ids).toEqual(['u1', 'u2']);
  });

  it('restores backup and clears undo target', async () => {
    const restoredConversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Before restore',
      status: 'active',
      messages: [
        {
          message_id: 'u-final',
          role: 'assistant',
          content: 'back to latest',
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
      updated_at: 3,
      schema_version: 1,
    };

    global.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/backups/backup-undo-1/restore') && init?.method === 'POST') {
        return new Response(JSON.stringify({
          status: 'restored',
          backup_id: 'backup-undo-1',
          conversation: restoredConversation,
        }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(restoredConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      restoreNotice: {
        message: 'Restored snapshot. You can undo this restore from the previous state backup.',
        undoBackupId: 'backup-undo-1',
        kind: 'restore_applied',
        conversationId: 'conv-1',
      },
    }));

    await useChatStore.getState().restoreBackup('backup-undo-1');

    expect(useChatStore.getState().restoringBackupId).toBeNull();
    expect(useChatStore.getState().activeConversation?.title).toBe('Before restore');
    expect(useChatStore.getState().restoreNotice).toMatchObject({
      undoBackupId: null,
      conversationId: 'conv-1',
      kind: 'restore_undone',
    });
    expect(useChatStore.getState().restoreNotice?.message).toContain('Undo restore completed');
    expect(useChatStore.getState().scrollToMessageId).toBe('u-final');
  });

  it('hydrates conversation context state on load', async () => {
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
      conversation_context_state: {
        conversation_id: 'conv-1',
        used_units: 125306,
        max_units: 272000,
        state: 'elevated',
        last_compacted_at: null,
        last_compaction_delta: null,
        updated_at: 2,
      },
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
    }));

    await useChatStore.getState().loadConversation('conv-1');

    expect(useChatStore.getState().activeConversation?.conversation_context_state).toEqual({
      conversation_id: 'conv-1',
      used_units: 125306,
      max_units: 272000,
      state: 'elevated',
      last_compacted_at: null,
      last_compaction_delta: null,
      updated_at: 2,
    });
  });

  it('loads active stream state', async () => {
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
      active_streaming_state: {
        conversation_id: 'conv-1',
        message_id: 'assistant-1',
        request_id: 'req-1',
        status: 'streaming',
        started_at: 2,
        updated_at: 2,
      },
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
    }));

    await useChatStore.getState().loadConversation('conv-1');

    expect(useChatStore.getState().streaming).toMatchObject({
      isStreaming: true,
      messageId: 'assistant-1',
      streamConversationId: 'conv-1',
    });
    expect(useChatStore.getState().activeConversation?.messages.at(-1)).toMatchObject({
      message_id: 'assistant-1',
      role: 'assistant',
      content: '',
    });
  });

  it('updates latest compaction from live event', async () => {
    const compactionRecord = {
      compaction_id: 'cmp-live-1',
      trigger: 'repo_intent_trim',
      summary: 'Compacted older turns',
      source_message_ids: ['u1', 'u2'],
      pinned_message_ids: [],
      retained_refs: ['README.md'],
      metrics: { messages_compacted: 2, tokens_before: 1800, tokens_after: 650 },
      created_at: 10,
      metadata: { reason: 'repo_intent_recent_window' },
    };
    const persistedConversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'user-1',
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
      model: 'demo',
      system_instructions: '',
      temperature: null,
      max_tokens: null,
      top_p: null,
      stream: true,
      bookmarked: false,
      metadata: {
        compaction_history: [compactionRecord],
      },
      created_at: 1,
      updated_at: 2,
      schema_version: 1,
    };
    const encoder = new TextEncoder();
    const streamBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: context.compacted\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-1', compaction: compactionRecord })}\n\n`
          + 'event: message.finish\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-1',
            model: 'demo',
            finish_reason: 'stop',
            total_chunks: 0,
            content_length: 0,
            timestamp: 10,
          })}\n\n`,
        ));
        controller.close();
      },
    });

    global.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages') && init?.method === 'POST') {
        return new Response(streamBody, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(persistedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
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
    }));

    await useChatStore.getState().sendMessage('hello');

    await waitFor(() => {
      expect(useChatStore.getState().latestCompaction).toMatchObject({
        compaction_id: 'cmp-live-1',
        trigger: 'repo_intent_trim',
      });
      expect(useChatStore.getState().activeConversation?.conversation_context_state).toMatchObject({
        state: 'compacted_recently',
        last_compacted_at: 10,
        last_compaction_delta: 1150,
      });
    });
  });

  it('refreshes latest compaction after stream when compaction lands late', async () => {
    const encoder = new TextEncoder();
    const firstConversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'assistant-1',
          role: 'assistant',
          content: 'done',
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
    const secondConversation: Conversation = {
      ...firstConversation,
      metadata: {
        compaction_history: [
          {
            compaction_id: 'cmp-late-1',
            trigger: 'post_turn_background',
            summary: 'Compacted older turns after the response completed.',
            source_message_ids: ['u1', 'u2'],
            pinned_message_ids: [],
            retained_refs: [],
            metrics: {
              messages_compacted: 2,
              tokens_before: 1900,
              tokens_after: 700,
            },
            created_at: 3,
            metadata: {},
          },
        ],
      },
      updated_at: 3,
    };
    const streamBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: message.finish\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-1',
            model: 'demo',
            finish_reason: 'stop',
            total_chunks: 0,
            content_length: 0,
            timestamp: 10,
          })}\n\n`,
        ));
        controller.close();
      },
    });
    let conversationFetches = 0;
    const setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout');

    global.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages') && init?.method === 'POST') {
        return new Response(streamBody, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        conversationFetches += 1;
        return new Response(
          JSON.stringify(conversationFetches >= 2 ? secondConversation : firstConversation),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        );
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
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
    }));

    await useChatStore.getState().sendMessage('hello');
    expect(useChatStore.getState().latestCompaction).toBeNull();
    expect(setTimeoutSpy).toHaveBeenCalled();

    await useChatStore.getState().loadConversation('conv-1', undefined, true);

    expect(useChatStore.getState().latestCompaction).toMatchObject({
      compaction_id: 'cmp-late-1',
      trigger: 'post_turn_background',
    });
  });

  it('rejects send while streaming', async () => {
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

  it('loads context state into conversation', async () => {
    const encoder = new TextEncoder();
    const conversationPayload: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [],
      model: 'deepseek-chat',
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
    const streamBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: message.error\n'
          + `data: ${JSON.stringify({ error: '401 unauthorized' })}\n\n`,
        ));
        controller.close();
      },
    });

    global.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages') && init?.method === 'POST') {
        return new Response(streamBody, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify({
          ...conversationPayload,
          messages: [
            {
              message_id: 'u1',
              role: 'user',
              content: 'hello',
              metadata: {},
              created_at: 1,
            },
          ],
        }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().sendMessage('hello');

    await waitFor(() => {
      expect(useChatStore.getState().error).toBe(
        'The model request failed. Please retry in a moment.',
      );
    });
  });

  it('rejects regenerate while streaming', async () => {
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

  it('regenerate clears streaming without reloading conversation after completion', async () => {
    const conversationPayload: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'assistant-1',
          role: 'assistant',
          content: 'previous',
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

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: message.delta\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', content: 'done' })}\n\n`
          + 'event: message.finish\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', model: 'demo' })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages/assistant-1/regenerate')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().regenerateMessage('assistant-1');

    await waitFor(() => {
      expect(useChatStore.getState().streaming.isStreaming).toBe(false);
      expect(useChatStore.getState().streaming.streamConversationId).toBeNull();
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/chat/conversations/conv-1/messages/assistant-1/regenerate',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/chat/conversations',
      expect.anything(),
    );
    expect(useChatStore.getState().activeConversation?.messages).toEqual([
      ...conversationPayload.messages,
      expect.objectContaining({
        message_id: 'assistant-2',
        role: 'assistant',
        content: 'done',
      }),
    ]);
  });

  it('regenerate keeps streaming until complete', async () => {
    const conversationPayload: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        {
          message_id: 'assistant-1',
          role: 'assistant',
          content: 'previous',
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

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        controller.enqueue(encoder.encode(
          'event: message.delta\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', content: 'done' })}\n\n`
          + 'event: message.finish\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', model: 'demo' })}\n\n`,
        ));
        await Promise.resolve();
        expect(useChatStore.getState().streaming.isStreaming).toBe(true);
        controller.enqueue(encoder.encode(
          'event: message.complete\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop' } })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages/assistant-1/regenerate')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().regenerateMessage('assistant-1');

    await waitFor(() => {
      expect(useChatStore.getState().streaming.isStreaming).toBe(false);
    });
  });

  it('resend avoids conversation reload after completion', async () => {
    const conversationPayload: Conversation = {
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

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        controller.enqueue(encoder.encode(
          'event: message.delta\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', content: 'done' })}\n\n`
          + 'event: message.finish\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', model: 'demo' })}\n\n`,
        ));
        await Promise.resolve();
        expect(useChatStore.getState().streaming.isStreaming).toBe(true);
        controller.enqueue(encoder.encode(
          'event: message.complete\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop' } })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        throw new Error('resend should not reload conversation');
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().resendMessage('hello');

    await waitFor(() => {
      expect(useChatStore.getState().streaming.isStreaming).toBe(false);
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('ignores transport errors after message.complete has already landed', async () => {
    const conversationPayload: Conversation = {
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

    const encoder = new TextEncoder();
    let readCount = 0;
    const reader = {
      read: vi.fn(async () => {
        readCount += 1;
        if (readCount === 1) {
          return {
            done: false,
            value: encoder.encode(
              'event: message.delta\n'
              + `data: ${JSON.stringify({ message_id: 'assistant-2', content: 'done' })}\n\n`
              + 'event: message.complete\n'
              + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop' } })}\n\n`,
            ),
          };
        }
        throw new Error('The model request timed out before the response completed.');
      }),
    };

    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return {
          ok: true,
          body: {
            getReader: () => reader,
          },
        } as unknown as Response;
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify({
          ...conversationPayload,
          messages: [
            ...conversationPayload.messages,
            {
              message_id: 'assistant-2',
              role: 'assistant',
              content: 'done',
              metadata: { finish_reason: 'stop' },
              created_at: 2,
            },
          ],
        }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
      error: 'old timeout',
    }));

    await useChatStore.getState().sendMessage('hello');

    await waitFor(() => {
      expect(useChatStore.getState().streaming.isStreaming).toBe(false);
    });
    expect(useChatStore.getState().error).toBeNull();
    expect(useChatStore.getState().activeConversation?.messages.some((m) => m.message_id === 'assistant-2')).toBe(true);
  });

  it('resend optimistically appends a new user message before response arrives', async () => {
    const conversationPayload: Conversation = {
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

    let resolveStream: (() => void) | undefined;
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        resolveStream = () => {
          controller.enqueue(encoder.encode(
            'event: message.complete\n'
            + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop' } })}\n\n`,
          ));
          controller.close();
        };
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
      composerUiByConversation: {
        'conv-1': {
          enableReasoning: true,
          enableWebSearch: true,
          enableDeepResearch: false,
          showAdvanced: true,
          maxTokensDraft: '1024',
        },
      },
    }));

    const resendPromise = useChatStore.getState().resendMessage('hello');

    expect(useChatStore.getState().activeConversation?.messages).toHaveLength(2);
    expect(useChatStore.getState().activeConversation?.messages[0]).toMatchObject({
      message_id: 'u1',
      role: 'user',
      content: 'hello',
    });
    expect(useChatStore.getState().activeConversation?.messages[1]).toMatchObject({
      role: 'user',
      content: 'hello',
    });
    const requestCall = fetchMock.mock.calls.find((call) => String(call[0]).endsWith('/api/chat/conversations/conv-1/messages'));
    expect(requestCall).toBeTruthy();
    const requestInit = ((requestCall as unknown as [RequestInfo | URL, RequestInit | undefined] | undefined)?.[1]);
    expect(requestInit).toBeTruthy();
    expect(JSON.parse(String(requestInit?.body))).toMatchObject({
      content: 'hello',
      enable_reasoning: true,
      enable_web_search: true,
      enable_skills: ['houyi_web_search'],
      max_tokens: 1024,
      stream: true,
    });

    const finishStream = resolveStream;
    if (typeof finishStream === 'function') {
      finishStream();
    }
    await resendPromise;
  });

  it('tool_call.start creates an assistant anchor before tool messages', async () => {
    const conversationPayload: Conversation = {
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
    const persistedConversation: Conversation = {
      ...conversationPayload,
      messages: [
        ...conversationPayload.messages,
        {
          message_id: 'assistant-2',
          role: 'assistant',
          content: '',
          metadata: { finish_reason: 'stop' },
          created_at: 2,
        },
        {
          message_id: 'tool-1',
          role: 'tool',
          content: '{\n  "entries": [\n    "README.md"\n  ]\n}',
          name: 'houyi_list_dir',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'ok', round_index: 1 },
          created_at: 2,
        },
      ],
      updated_at: 2,
    };

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: tool_call.start\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-2',
            tool_call_id: 'call-1',
            tool_name: 'houyi_list_dir',
            arguments: { path: '.' },
            round_index: 1,
          })}\n\n`
          + 'event: message.complete\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop' } })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(persistedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().resendMessage('hello');

    const messages = useChatStore.getState().activeConversation?.messages ?? [];
    const assistantIndex = messages.findIndex((m) => m.role === 'assistant');
    const toolIndex = messages.findIndex((m) => m.role === 'tool');
    const toolMessages = messages.filter((m) => m.role === 'tool');
    expect(assistantIndex).toBeGreaterThanOrEqual(0);
    expect(toolIndex).toBeGreaterThanOrEqual(0);
    expect(toolMessages[0].message_id).toBe('tool-1');
  });

  it('prefers requested_tool_name for tool activity display', async () => {
    const conversationPayload: Conversation = {
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

    const persistedConversation: Conversation = {
      ...conversationPayload,
      messages: [
        ...conversationPayload.messages,
        {
          message_id: 'assistant-2',
          role: 'assistant',
          content: 'done',
          metadata: { finish_reason: 'stop', trace_id: 'trace-1' },
          created_at: 2,
        },
        {
          message_id: 'tool-1',
          role: 'tool',
          content: '{\n  "ok": true,\n  "path": "README.md"\n}',
          name: 'houyi_read_file',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'ok', round_index: 1 },
          created_at: 2,
        },
      ],
      updated_at: 2,
    };

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: tool_call.start\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-2',
            tool_call_id: 'call-1',
            tool_name: 'houyi_read_file',
            arguments: { path: 'README.md' },
            round_index: 1,
          })}\n\n`
          + 'event: tool_call.result\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-2',
            tool_call_id: 'call-1',
            tool_name: 'houyi_read_file',
            result: { ok: true, path: 'README.md' },
            round_index: 1,
          })}\n\n`
          + 'event: agent.finalizing\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', trace_id: 'trace-1' })}\n\n`
          + 'event: message.delta\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', seq: 1, content: 'done' })}\n\n`
          + 'event: message.complete\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop', trace_id: 'trace-1' } })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(persistedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().resendMessage('hello');

    const messages = useChatStore.getState().activeConversation?.messages ?? [];
    const assistant = messages.find((m) => m.role === 'assistant' && m.message_id === 'assistant-2');
    const toolMessages = messages.filter((m) => m.role === 'tool');
    expect(assistant?.content).toBe('done');
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0]).toMatchObject({
      name: 'houyi_read_file',
      tool_call_id: 'call-1',
    });
  });

  it('finalizes running tool activity on message.complete when tool result is missing', async () => {
    const conversationPayload: Conversation = {
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

    const persistedConversation: Conversation = {
      ...conversationPayload,
      messages: [
        ...conversationPayload.messages,
        {
          message_id: 'assistant-2',
          role: 'assistant',
          content: 'done',
          metadata: { finish_reason: 'stop', trace_id: 'trace-1' },
          created_at: 2,
        },
        {
          message_id: 'tool-1',
          role: 'tool',
          content: '{\n  "path": "."\n}',
          name: 'houyi_list_dir',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'completed', round_index: 1 },
          created_at: 2,
        },
      ],
      updated_at: 2,
    };

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: tool_call.start\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-2',
            tool_call_id: 'call-1',
            tool_name: 'houyi_list_dir',
            arguments: { path: '.' },
            round_index: 1,
          })}\n\n`
          + 'event: message.delta\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', seq: 1, content: 'done' })}\n\n`
          + 'event: message.complete\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop', trace_id: 'trace-1' } })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(persistedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().resendMessage('hello');

    const toolMessage = useChatStore.getState().activeConversation?.messages.find(
      (message) => message.role === 'tool' && message.tool_call_id === 'call-1',
    );
    expect(toolMessage?.metadata?.tool_status).toBe('completed');
    expect(useChatStore.getState().streaming.isStreaming).toBe(false);
  });

  it('preserves loaded assistant content when refresh returns tool messages before the final assistant', async () => {
    const currentMessages: ChatMessage[] = [
      {
        message_id: 'u1',
        role: 'user',
        content: 'hello',
        metadata: {},
        created_at: 1,
      },
      {
        message_id: 'assistant-2',
        role: 'assistant',
        content: '',
        metadata: { trace_id: 'trace-1' },
        created_at: 2,
      },
      {
        message_id: 'tmp-tool-1',
        role: 'tool',
        content: '{"path":"README.md"}',
        name: 'houyi_read_file',
        tool_call_id: 'call-1',
        metadata: { tool_status: 'running', round_index: 1 },
        created_at: 2,
      },
    ];

    const loadedConversation: Conversation = {
      conversation_id: 'conv-1',
      title: 'Chat',
      status: 'active',
      messages: [
        currentMessages[0],
        {
          message_id: 'tool-1',
          role: 'tool',
          content: '{"path":"README.md"}',
          name: 'houyi_read_file',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'ok', round_index: 1 },
          created_at: 2,
        },
        {
          message_id: 'assistant-2',
          role: 'assistant',
          content: 'final summary content',
          metadata: { trace_id: 'trace-1', finish_reason: 'stop' },
          created_at: 3,
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
      updated_at: 3,
      schema_version: 1,
    };

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(loadedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: {
        ...loadedConversation,
        messages: currentMessages,
      },
    }));

    await useChatStore.getState().loadConversation('conv-1', undefined, true);

    const assistant = useChatStore.getState().activeConversation?.messages.find((m) => m.message_id === 'assistant-2');
    expect(assistant?.content).toBe('final summary content');
  });

  it('resend reloads conversation after tool loop completion', async () => {
    const conversationPayload: Conversation = {
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

    const reloadedConversation: Conversation = {
      ...conversationPayload,
      messages: [
        ...conversationPayload.messages,
        {
          message_id: 'assistant-2',
          role: 'assistant',
          content: 'done',
          metadata: { trace_id: 'trace-1' },
          created_at: 2,
        },
        {
          message_id: 'tool-1',
          role: 'tool',
          content: '{"ok":true}',
          name: 'houyi_read_file',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'ok', round_index: 1 },
          created_at: 2,
        },
      ],
      updated_at: 2,
    };

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: tool_call.start\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-2',
            tool_call_id: 'call-1',
            tool_name: 'houyi_read_file',
            arguments: { path: 'README.md' },
            round_index: 1,
          })}\n\n`
          + 'event: agent.finalizing\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', trace_id: 'trace-1' })}\n\n`
          + 'event: message.delta\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', seq: 1, content: 'done' })}\n\n`
          + 'event: message.complete\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop', trace_id: 'trace-1' } })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(reloadedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: conversationPayload,
    }));

    await useChatStore.getState().resendMessage('hello');

    expect(
      fetchMock.mock.calls.some(
        ([input]) => String(input).endsWith('/api/chat/conversations/conv-1'),
      ),
    ).toBe(true);
    expect(
      fetchMock.mock.calls.some(
        ([input]) => String(input).endsWith('/api/chat/conversations/conv-1/context-usage'),
      ),
    ).toBe(true);
  });

  it('reconciles tool results when tool call id mapping is missing', async () => {
    const conversationPayload: Conversation = {
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

    const persistedConversation: Conversation = {
      ...conversationPayload,
      messages: [
        ...conversationPayload.messages,
        {
          message_id: 'assistant-2',
          role: 'assistant',
          content: 'done',
          metadata: { finish_reason: 'stop', trace_id: 'trace-1' },
          created_at: 2,
        },
        {
          message_id: 'tool-1',
          role: 'tool',
          content: '{\n  "entries": [\n    "README.md"\n  ]\n}',
          name: 'houyi_list_dir',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'ok', round_index: 1 },
          created_at: 2,
        },
      ],
      updated_at: 2,
    };

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(
          'event: tool_call.result\n'
          + `data: ${JSON.stringify({
            message_id: 'assistant-2',
            tool_call_id: 'call-1',
            tool_name: 'houyi_list_dir',
            result: { entries: ['README.md'] },
            round_index: 1,
          })}\n\n`
          + 'event: message.complete\n'
          + `data: ${JSON.stringify({ message_id: 'assistant-2', metadata: { finish_reason: 'stop', trace_id: 'trace-1' } })}\n\n`,
        ));
        controller.close();
      },
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/chat/conversations/conv-1/messages')) {
        return new Response(stream, { status: 200 });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/context-usage')) {
        return new Response(JSON.stringify({ usage: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1/compactions')) {
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations/conv-1')) {
        return new Response(JSON.stringify(persistedConversation), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat/conversations')) {
        return new Response(JSON.stringify({ conversations: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    global.fetch = fetchMock as typeof fetch;

    useChatStore.setState((state) => ({
      ...state,
      activeConversationId: 'conv-1',
      activeConversation: persistedConversation,
      streaming: {
        ...state.streaming,
        isStreaming: true,
        messageId: 'assistant-2',
        streamConversationId: 'conv-1',
        toolMessageIdsByCallId: {},
      },
    }));

    await useChatStore.getState().resendMessage('hello');

    const toolMessage = useChatStore.getState().activeConversation?.messages.find((m) => m.role === 'tool');
    expect(toolMessage).toBeTruthy();
    expect(toolMessage?.metadata?.tool_status).toBe('ok');
    expect(String(toolMessage?.content || '')).toContain('README.md');
  });
});
