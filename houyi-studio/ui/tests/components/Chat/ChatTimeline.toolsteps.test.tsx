import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatTimeline } from '@/components/Chat/ChatTimeline';

describe('ChatTimeline tool-step association', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}

        unobserve() {}

        disconnect() {}
      },
    );
  });

  it('attaches tool messages between assistant turns to the next assistant message', () => {
    const now = Date.now() / 1000;
    render(
      <ChatTimeline
        conversationId="c1"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={[
          {
            message_id: 'u1',
            role: 'user',
            content: 'find files',
            metadata: {},
            created_at: now,
          },
          {
            message_id: 'a-placeholder',
            role: 'assistant',
            content: '',
            tool_calls: [{ id: 'call-1' }],
            metadata: {},
            created_at: now + 1,
          },
          {
            message_id: 't1',
            role: 'tool',
            name: 'houyi_read_file',
            tool_call_id: 'call-1',
            content: '{"ok":true}',
            metadata: { tool_status: 'ok', round_index: 1 },
            created_at: now + 2,
          },
          {
            message_id: 'a-final',
            role: 'assistant',
            content: 'done',
            metadata: {},
            created_at: now + 3,
          },
        ]}
      />,
    );

    expect(screen.getByText('done')).toBeInTheDocument();
    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Show steps'));
    expect(screen.getByText('houyi_read_file')).toBeInTheDocument();
  });

  it('shows steps when only assistant tool_calls placeholder exists', () => {
    const now = Date.now() / 1000;
    render(
      <ChatTimeline
        conversationId="c2"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={[
          {
            message_id: 'u2',
            role: 'user',
            content: 'search web',
            metadata: {},
            created_at: now,
          },
          {
            message_id: 'a2-placeholder',
            role: 'assistant',
            content: '',
            tool_calls: [
              {
                id: 'call-2',
                function: {
                  name: 'web_search',
                  arguments: '{"query":"houyi"}',
                },
              },
            ],
            metadata: {},
            created_at: now + 1,
          },
          {
            message_id: 'a2-final',
            role: 'assistant',
            content: 'done',
            metadata: {},
            created_at: now + 2,
          },
        ]}
      />,
    );

    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Show steps'));
    expect(screen.getByText('web_search')).toBeInTheDocument();
  });

  it('shows steps when assistant tool_calls carrier has non-empty content', () => {
    const now = Date.now() / 1000;
    render(
      <ChatTimeline
        conversationId="c3"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={[
          {
            message_id: 'u3',
            role: 'user',
            content: 'locate docs',
            metadata: {},
            created_at: now,
          },
          {
            message_id: 'a3-carrier',
            role: 'assistant',
            content: 'I will search now.',
            tool_calls: [
              {
                id: 'call-3',
                function: {
                  name: 'houyi_find_files',
                  arguments: '{"pattern":"*.md"}',
                },
              },
            ],
            metadata: {},
            created_at: now + 1,
          },
          {
            message_id: 'a3-final',
            role: 'assistant',
            content: 'Found docs.',
            metadata: {},
            created_at: now + 2,
          },
        ]}
      />,
    );

    expect(screen.getByText('Found docs.')).toBeInTheDocument();
    expect(screen.getByText('Tool calls 1')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Show steps'));
    expect(screen.getByText('houyi_find_files')).toBeInTheDocument();
  });

  it('renders safely when assistant reasoning payload is non-string', () => {
    const now = Date.now() / 1000;
    render(
      <ChatTimeline
        conversationId="c4"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={[
          {
            message_id: 'u4',
            role: 'user',
            content: 'analyze',
            metadata: {},
            created_at: now,
          },
          {
            message_id: 'a4',
            role: 'assistant',
            content: 'done',
            reasoning_content: { step: 'internal' } as unknown as string,
            metadata: {},
            created_at: now + 1,
          },
        ]}
      />,
    );

    expect(screen.getByText('done')).toBeInTheDocument();
    expect(screen.getByText('Thinking')).toBeInTheDocument();
  });

  it('hides tool summary while assistant is actively streaming', () => {
    const now = Date.now() / 1000;
    render(
      <ChatTimeline
        conversationId="c5"
        streamingMessageId="a5"
        isWaitingForResponse={false}
        messages={[
          {
            message_id: 'u5',
            role: 'user',
            content: 'find skill file',
            metadata: {},
            created_at: now,
          },
          {
            message_id: 'a5',
            role: 'assistant',
            content: 'searching...',
            metadata: {},
            created_at: now + 1,
          },
          {
            message_id: 't5',
            role: 'tool',
            name: 'houyi_find_files',
            tool_call_id: 'call-5',
            content: '{"matches":0}',
            metadata: { tool_status: 'ok', round_index: 1 },
            created_at: now + 2,
          },
        ]}
      />,
    );

    expect(screen.queryByText('Tool calls 1')).not.toBeInTheDocument();
    expect(screen.queryByText('Show steps')).not.toBeInTheDocument();
    expect(screen.queryByText('Hide steps')).not.toBeInTheDocument();
  });

  it('rerenders safely when an empty assistant placeholder stops streaming', () => {
    const now = Date.now() / 1000;
    const messages = [
      {
        message_id: 'u6',
        role: 'user' as const,
        content: 'hello',
        metadata: {},
        created_at: now,
      },
      {
        message_id: 'a6',
        role: 'assistant' as const,
        content: '',
        metadata: {},
        created_at: now + 1,
      },
    ];

    const { rerender } = render(
      <ChatTimeline
        conversationId="c6"
        streamingMessageId="a6"
        isWaitingForResponse={false}
        messages={messages}
      />, 
    );

    rerender(
      <ChatTimeline
        conversationId="c6"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={messages}
      />,
    );

    expect(screen.getByText('hello')).toBeInTheDocument();
  });
});
