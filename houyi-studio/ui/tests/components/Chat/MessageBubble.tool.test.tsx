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

    expect(screen.getAllByText('houyi_read_file').length).toBeGreaterThan(0);
    expect(screen.getByText('ok')).toBeInTheDocument();
  });

  it('renders tool duration and parallel group metadata', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'tool-msg-meta-1',
          role: 'tool',
          content: '{"result":"ok"}',
          name: 'web_search',
          tool_call_id: 'call-meta-1',
          metadata: {
            tool_status: 'ok',
            duration_ms: 1485,
            parallel_group_id: 'round_1',
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText('Duration 1.5s')).toBeInTheDocument();
    expect(screen.getByText('Parallel round_1')).toBeInTheDocument();
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

  it('does not render literal tool-call carrier marker as assistant content', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-tool-carrier-1',
          role: 'assistant',
          content: '[tool call]<tool_call>houyi_read_file</tool_call>',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.queryByText('[tool call]')).not.toBeInTheDocument();
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

    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    expect(screen.getAllByText('done').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Rounds 1').length).toBeGreaterThan(0);
    expect(screen.queryByText(/Stats/)).not.toBeInTheDocument();
    expect(screen.getByText('Tokens: 42')).toBeInTheDocument();

    fireEvent.click(screen.getAllByText('View trace')[0]);
    expect(onOpenTrace).toHaveBeenCalledWith('trace-123');

    expect(screen.queryByText('houyi_read_file')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
    expect(screen.getByText('houyi_read_file')).toBeInTheDocument();
  });

  it('renders tool activity with meta input output and raw payload sections', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-tools-structured',
          role: 'assistant',
          content: 'done',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
        toolSteps={[
          {
            message_id: 'tool-step-shell-1',
            role: 'tool',
            content: JSON.stringify({
              data: {
                command: 'find . -name "skill.md" 2>/dev/null | head -20',
                cwd: '/Users/von/workspace/HouYiAgent',
                timeout_seconds: 30,
                retry_count: 0,
                stdout: './skills/skill.md',
                stderr: '',
                returncode: 0,
                timed_out: false,
                message: 'command completed',
              },
              success: true,
            }),
            name: 'houyi_shell_exec',
            metadata: { tool_status: 'done', duration_ms: 406, round_index: 2 },
            created_at: Date.now() / 1000,
          },
        ]}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
    fireEvent.click(screen.getByRole('button', { name: /houyi_shell_exec/i }));
    expect(screen.getByText('Meta')).toBeInTheDocument();
    expect(screen.getByText('Input')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
    expect(screen.getByText('Raw payload')).toBeInTheDocument();
    expect(screen.getByText('Command')).toBeInTheDocument();
    expect(screen.getByText('Working directory')).toBeInTheDocument();
    expect(screen.getByText('Stdout')).toBeInTheDocument();
    expect(screen.getByText('/Users/von/workspace/HouYiAgent')).toBeInTheDocument();
    expect(screen.getByText('./skills/skill.md')).toBeInTheDocument();
    expect(screen.getAllByText('406 ms').length).toBeGreaterThan(0);
  });

  it('formats structured tool payload into readable output sections', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-tools-find-files',
          role: 'assistant',
          content: '',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
        toolSteps={[
          {
            message_id: 'tool-step-find-1',
            role: 'tool',
            content: JSON.stringify({
              data: {
                matches: ['a.ts', 'b.ts'],
                root: '/repo',
                pattern: 'skill.md',
              },
              success: true,
              _truncated: true,
              _truncated_message: '...[truncated]...',
            }),
            name: 'houyi_find_files',
            metadata: { tool_status: 'done' },
            created_at: Date.now() / 1000,
          },
        ]}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
    fireEvent.click(screen.getByRole('button', { name: /houyi_find_files/i }));
    expect(screen.getByText('matches: 2')).toBeInTheDocument();
    expect(screen.getByText('Matches')).toBeInTheDocument();
    expect(screen.getByText('Truncated')).toBeInTheDocument();
    expect(screen.getByText('Truncation note')).toBeInTheDocument();
  });

  it('renders assistant latency and throughput stats', () => {
    vi.useFakeTimers();
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-stats-1',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: { prompt_tokens: 34, completion_tokens: 54, total_tokens: 88 },
            first_token_latency_ms: 187,
            end_to_end_tokens_per_second: 22.1,
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText('Tokens: 88 ↑34 ↓54')).toBeInTheDocument();
    fireEvent.mouseEnter(screen.getByText(/Tokens:/i));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.queryByText(/Tokens 88/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Prompt 34/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Completion 54/i)).not.toBeInTheDocument();
    expect(screen.getByText(/First token\s+187 ms/i)).toBeInTheDocument();
    expect(screen.getByText(/Throughput\s+22 tokens\/s/i)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it('restores tooltip timing stats from usage metadata fallback fields', () => {
    vi.useFakeTimers();
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-stats-usage-fallback',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: {
              prompt_tokens: 59413,
              completion_tokens: 518,
              total_tokens: 59931,
              first_token_latency_ms: 142,
              end_to_end_tokens_per_second: 31.4,
            },
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText('Tokens: 59931 ↑59413 ↓518')).toBeInTheDocument();
    fireEvent.mouseEnter(screen.getByText(/Tokens:/i));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText(/First token\s+142 ms/i)).toBeInTheDocument();
    expect(screen.getByText(/Throughput\s+31 tokens\/s/i)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it('does not borrow tooltip metrics from unrelated active conversation summary state', () => {
    vi.useFakeTimers();
    useChatStore.setState((state) => ({
      ...state,
      activeConversation: {
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
            message_id: 'assistant-summary-fallback',
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
      } as any,
      agentLoopSummary: {
        rounds: 0,
        toolCalls: 0,
        traceId: 'trace-1',
        usage: {
          prompt_tokens: 59413,
          completion_tokens: 518,
          total_tokens: 59931,
        },
        metrics: {
          first_token_latency_ms: 142,
          end_to_end_tokens_per_second: 31.4,
        },
        status: 'done',
      },
    }));

    render(
      <MessageBubble
        message={{
          message_id: 'assistant-other-conversation',
          role: 'assistant',
          content: 'done',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.queryByText('Tokens: 59931 ↑59413 ↓518')).not.toBeInTheDocument();
    vi.useRealTimers();
  });

  it('keeps token tooltip available even when only usage metadata exists', () => {
    vi.useFakeTimers();
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-usage-only',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: { prompt_tokens: 7022, completion_tokens: 4, total_tokens: 7026 },
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText('Tokens: 7026 ↑7022 ↓4')).toBeInTheDocument();
    fireEvent.mouseEnter(screen.getByText(/Tokens:/i));
    act(() => {
      vi.advanceTimersByTime(100);
    });

    expect(screen.getByText(/Tokens 7026/i)).toBeInTheDocument();
    expect(screen.getAllByText(/↑7022/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/↓4/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Prompt 7022/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Completion 4/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/First token/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Throughput/i)).not.toBeInTheDocument();
    vi.useRealTimers();
  });

  it('keeps the metrics tooltip above the last message row', () => {
    vi.useFakeTimers();
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-stats-last',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
            first_token_latency_ms: 120,
          },
          created_at: Date.now() / 1000,
        }}
        isLastMessage
      />,
    );

    fireEvent.mouseEnter(screen.getByText(/Tokens:/i));
    act(() => {
      vi.advanceTimersByTime(100);
    });

    expect(screen.getByText(/First token\s+120 ms/i).closest('div.pointer-events-none')).toHaveClass('fixed');
    vi.useRealTimers();
  });

  it('uses metadata first token and decode throughput fields for tooltip metrics', () => {
    vi.useFakeTimers();
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-metadata-throughput-1',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 },
            first_token_ms: 529,
            decode_tokens_per_second: 96,
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    fireEvent.mouseEnter(screen.getByText(/Tokens:/i));
    act(() => {
      vi.advanceTimersByTime(100);
    });

    const tooltip = screen.getByText(/First token\s+529 ms/i).closest('div.pointer-events-none');
    expect(tooltip).toBeInTheDocument();
    expect(screen.getByText(/Throughput\s+96 tokens\/s/i)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it('keeps timing tooltip available when metrics exist without token totals', () => {
    vi.useFakeTimers();
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-metrics-only',
          role: 'assistant',
          content: 'done',
          metadata: {
            first_token_ms: 411,
            decode_tokens_per_second: 73,
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText('Metrics')).toBeInTheDocument();
    fireEvent.mouseEnter(screen.getByText('Metrics'));
    act(() => {
      vi.advanceTimersByTime(100);
    });

    expect(screen.getByText(/First token\s+411 ms/i)).toBeInTheDocument();
    expect(screen.getByText(/Throughput\s+73 tokens\/s/i)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it('wraps long assistant token labels inside the meta row', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-long-token-label',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: {
              prompt_tokens: 37594,
              completion_tokens: 414,
              total_tokens: 38008,
            },
            trace_id: 'trace-long-token-label',
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    const tokenLabel = screen.getByText(/Tokens: 38008/i);
    expect(tokenLabel).toHaveClass('break-all');
    expect(tokenLabel).not.toHaveClass('whitespace-nowrap');
  });

  it('renders reasoning budget metadata', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-budget-1',
          role: 'assistant',
          content: 'done',
          metadata: {
            usage: {
              prompt_tokens: 20,
              completion_tokens: 40,
              total_tokens: 60,
              reasoning_tokens: 12,
              answer_tokens: 28,
              cached_prompt_tokens: 0,
              usage_confidence: 'reported',
            },
            finish_reason: 'length',
            budget: {
              answer_reserve: 512,
              max_tokens_guardrail_applied: true,
            },
          },
          created_at: Date.now() / 1000,
        }}
      />,
    );
    expect(screen.queryByText('Reserve 512')).not.toBeInTheDocument();
    expect(screen.getByText('Guardrail')).toBeInTheDocument();
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

    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
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
    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
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
    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
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

  it('converts bracket tool replay text into tool activity when no tool step event survived', () => {
    render(
      <MessageBubble
        message={{
          message_id: 'assistant-bracket-tool-replay',
          role: 'assistant',
          content: '我帮你搜索本地名为 readme.md 或类似的文件：\n\n[tool:houyi_shell_exec] {"command":"find . -name README.md"}',
          metadata: {},
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText(/我帮你搜索本地名为 readme.md 或类似的文件/i)).toBeInTheDocument();
    expect(screen.queryByText(/\[tool:houyi_shell_exec\]/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
    expect(screen.getByText('houyi_shell_exec')).toBeInTheDocument();
    expect(screen.getByText(/find \. -name README\.md/i)).toBeInTheDocument();
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

  it('shows inline running tool activity while streaming and expands details on demand', () => {
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
        metadata: { tool_status: 'running', round_index: 1, duration_ms: 350 },
        created_at: Date.now() / 1000,
      },
    ];

    render(
      <MessageBubble
        message={baseMessage}
        toolSteps={steps}
        isLastMessage
        isStreaming
      />,
    );

    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    expect(screen.getAllByText('running').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Duration 350ms').length).toBeGreaterThan(0);
    expect(screen.queryByText('houyi_read_file')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
    expect(screen.getByText('houyi_read_file')).toBeInTheDocument();
  });

  it('keeps inline tool activity collapsed after streaming completes until manually expanded', () => {
    const baseMessage = {
      message_id: 'assistant-stream-steps-complete',
      role: 'assistant' as const,
      content: '',
      metadata: {},
      created_at: Date.now() / 1000,
    };
    const steps = [
      {
        message_id: 'tool-step-stream-complete-1',
        role: 'tool' as const,
        content: '{"ok":true}',
        name: 'houyi_read_file',
        metadata: { tool_status: 'ok', round_index: 1, duration_ms: 420 },
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

    rerender(
      <MessageBubble
        message={baseMessage}
        toolSteps={steps}
        isLastMessage
        isStreaming={false}
      />,
    );

    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    expect(screen.getAllByText('Duration 420ms').length).toBeGreaterThan(0);
    expect(screen.queryByText('houyi_read_file')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
    expect(screen.getByText('houyi_read_file')).toBeInTheDocument();
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
