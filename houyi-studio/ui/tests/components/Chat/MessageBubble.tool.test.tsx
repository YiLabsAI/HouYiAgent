import { act, fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { MessageBubble } from '@/components/Chat/MessageBubble';

vi.mock('@/stores/useChatStore', async () => {
  const actual = await vi.importActual<typeof import('@/stores/useChatStore')>('@/stores/useChatStore');
  return {
    ...actual,
    useChatStore: Object.assign(actual.useChatStore, {
      getState: actual.useChatStore.getState,
      setState: actual.useChatStore.setState,
    }),
  };
});

const { useChatStore } = await import('@/stores/useChatStore');

describe('MessageBubble(tool)', () => {
  it('renders ToolCallBubble for tool role message', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'tool-msg-1',
          role: 'tool',
          content: '{"result":"ok"}',
          name: 'houyi_read_file',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'ok' },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText('houyi_read_file')).toBeInTheDocument();
    expect(screen.getByText('ok')).toBeInTheDocument();
  });

  it('does not render empty non-streaming assistant placeholder', () => {
    const { container } = render(
      <MessageBubble
        message={{
          message_id: 'assistant-empty-1',
          role: 'assistant',
          content: '',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(container.firstChild).toBeNull();
  });

  it('renders embedded tool steps and trace action on assistant message', () => {
    const onOpenTrace = vi.fn();
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-1',
          role: 'assistant',
          content: 'done',
          metadata: { trace_id: 'trace-123', usage: { total_tokens: 42 } },
          created_at: Date.now() / 1000,
        }}
        toolSteps={[
          {
            message_id: 'tool-step-1',
            role: 'tool',
            content: '{"result":"ok"}',
            name: 'houyi_read_file',
            metadata: { tool_status: 'ok', round_index: 1 },
            created_at: Date.now() / 1000,
          },
        ]}
        onOpenTrace={onOpenTrace}
      />,
    );

    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
    expect(screen.getByText('Rounds 1')).toBeInTheDocument();
    expect(screen.queryByText(/Stats/)).not.toBeInTheDocument();
    expect(screen.getByText('Tokens: 42')).toBeInTheDocument();

    fireEvent.click(screen.getByText('View trace'));
    expect(onOpenTrace).toHaveBeenCalledWith('trace-123');

    fireEvent.click(screen.getByText('Show steps'));
    expect(screen.getByText('houyi_read_file')).toBeInTheDocument();
  });

  it('renders assistant latency and throughput stats', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-stats-1',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: { prompt_tokens: 20, completion_tokens: 40, total_tokens: 60 },
            first_token_latency_ms: 187.4,
            tokens_per_second: 22.25,
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.queryByText(/Stats/)).not.toBeInTheDocument();
    expect(screen.getByText('First token 187 ms')).toBeInTheDocument();
    expect(screen.getByText('22 tokens/s')).toBeInTheDocument();
    expect(screen.getByText(/Tokens: 60\s*↑20\s*↓40/)).toBeInTheDocument();
  });

  it('renders user input token stats', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'user-stats-1',
          role: 'user',
          content: 'hello world',
          metadata: {
            usage: { input_tokens: 12, prompt_tokens: 12, total_tokens: 12 },
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getAllByText('You')).toHaveLength(1);
    expect(screen.getByText('Tokens: 12')).toBeInTheDocument();
  });

  it('does not render user token badge when only prompt_tokens exists', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'user-stats-prompt-only',
          role: 'user',
          content: 'hello world',
          metadata: {
            usage: { prompt_tokens: 12, total_tokens: 12 },
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.queryByText('Tokens: 12')).not.toBeInTheDocument();
  });

  it('keeps assistant bubble when content is empty but tool steps exist', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-empty-steps',
          role: 'assistant',
          content: '',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
        toolSteps={[
          {
            message_id: 'tool-step-empty-1',
            role: 'tool',
            content: '{"ok":true}',
            name: 'houyi_grep',
            metadata: { tool_status: 'ok', round_index: 1 },
            created_at: Date.now() / 1000,
          },
        ]}
      />,
    );

    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
    expect(screen.getByText('Show steps')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Show steps'));
    expect(screen.getByText('houyi_grep')).toBeInTheDocument();
  });

  it('strips inline tool marker payload from assistant content', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-inline-marker',
          role: 'assistant',
          content: '<|tool_calls_section_begin|><|tool_call_begin|>functions.houyi_grep<|tool_call_end|><|tool_calls_section_end|>',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
        toolSteps={[
          {
            message_id: 'tool-step-marker-1',
            role: 'tool',
            content: '{"ok":true}',
            name: 'houyi_grep',
            metadata: { tool_status: 'ok', round_index: 1 },
            created_at: Date.now() / 1000,
          },
        ]}
      />,
    );

    expect(screen.queryByText(/tool_calls_section_begin/i)).not.toBeInTheDocument();
    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
  });

  it('strips xml-style tool_call payload markers from assistant content', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-xml-tool-marker',
          role: 'assistant',
          content: '<tool_call>houyi_list_dir<arg_key>path</arg_key><arg_value>/tmp</arg_value></tool_call>',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
        toolSteps={[
          {
            message_id: 'tool-step-xml-1',
            role: 'tool',
            content: '{"ok":true}',
            name: 'houyi_list_dir',
            metadata: { tool_status: 'ok', round_index: 1 },
            created_at: Date.now() / 1000,
          },
        ]}
      />,
    );

    expect(screen.queryByText(/<tool_call>/i)).not.toBeInTheDocument();
    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
  });

  it('does not sanitize normal assistant text that only mentions parallel_tool_calls', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-normal-text-with-tool-word',
          role: 'assistant',
          content: 'Conclusion:\nif (parallel_tool_calls) {\n  return true;\n}',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText(/Conclusion:/)).toBeInTheDocument();
    expect(screen.getByText(/parallel_tool_calls/)).toBeInTheDocument();
    expect(screen.getByText(/return true/)).toBeInTheDocument();
  });

  it('strips leaked think wrappers and incomplete tool_call fragment from reasoning', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-reasoning-leak',
          role: 'assistant',
          content: 'Final answer',
          reasoning_content: '<think>plan\n<tool_call>houyi_search_text </think>',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
      />,
    );

    fireEvent.click(screen.getByText('Thinking'));
    expect(screen.queryByText(/<tool_call>/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/<\/think>/i)).not.toBeInTheDocument();
    expect(screen.getByText(/houyi_search_text/)).toBeInTheDocument();
  });

  it('strips leaked think wrappers and incomplete tool_call fragment from content', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-content-leak',
          role: 'assistant',
          content: '<tool_call>houyi_search_text </think>',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.queryByText(/<tool_call>/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/<\/think>/i)).not.toBeInTheDocument();
    expect(screen.getByText(/houyi_search_text/)).toBeInTheDocument();
  });

  it('hides tool summary while streaming and shows collapsed steps after completion', () => {
    const baseMessage = {
      message_id: 'assistant-stream-steps',
      role: 'assistant' as const,
      content: '',
      metadata: {},
      created_at: Date.now() / 1000,
    };
    const steps = [
      {
        message_id: 'tool-step-stream-1',
        role: 'tool' as const,
        content: '{"ok":true}',
        name: 'houyi_read_file',
        metadata: { tool_status: 'ok', round_index: 1 },
        created_at: Date.now() / 1000,
      },
    ];

    const { rerender } = render(
      <MessageBubble
        message={baseMessage}
        toolSteps={steps}
        isLastMessage
        isStreaming
      />,
    );

    expect(screen.queryByText('Tool calls 1')).not.toBeInTheDocument();
    expect(screen.queryByText('houyi_read_file')).not.toBeInTheDocument();

    rerender(
      <MessageBubble
        message={baseMessage}
        toolSteps={steps}
        isLastMessage
        isStreaming={false}
      />,
    );

    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
    expect(screen.getByText('Show steps')).toBeInTheDocument();
  });

  it('keeps the streaming thinking panel scrolled to the latest text', () => {
    let lastAssignedScrollTop = 0;
    const scrollHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLDivElement.prototype, 'scrollHeight');
    const scrollTopDescriptor = Object.getOwnPropertyDescriptor(HTMLDivElement.prototype, 'scrollTop');

    Object.defineProperty(HTMLDivElement.prototype, 'scrollHeight', {
      configurable: true,
      get() {
        return 480;
      },
    });
    Object.defineProperty(HTMLDivElement.prototype, 'scrollTop', {
      configurable: true,
      get() {
        return lastAssignedScrollTop;
      },
      set(value: number) {
        lastAssignedScrollTop = value;
      },
    });

    useChatStore.setState((state) => ({
      ...state,
      streaming: {
        ...state.streaming,
        messageId: 'assistant-stream-reasoning',
        reasoningBuffer: 'line 1\nline 2\nline 3',
      },
    }));

    const { rerender } = render(
      <MessageBubble
        message={{
          message_id: 'assistant-stream-reasoning',
          role: 'assistant',
          content: '',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
        isStreaming
      />,
    );

    try {
      act(() => {
        useChatStore.setState((state) => ({
          ...state,
          streaming: {
            ...state.streaming,
            messageId: 'assistant-stream-reasoning',
            reasoningBuffer: 'line 1\nline 2\nline 3\nline 4',
          },
        }));

        rerender(
          <MessageBubble
            message={{
              message_id: 'assistant-stream-reasoning',
              role: 'assistant',
              content: '',
              metadata: {},
              created_at: Date.now() / 1000,
            }}
            isStreaming
          />,
        );
      });

      expect(lastAssignedScrollTop).toBe(480);
    } finally {
      if (scrollHeightDescriptor) {
        Object.defineProperty(HTMLDivElement.prototype, 'scrollHeight', scrollHeightDescriptor);
      }
      if (scrollTopDescriptor) {
        Object.defineProperty(HTMLDivElement.prototype, 'scrollTop', scrollTopDescriptor);
      }
    }
  });
});
