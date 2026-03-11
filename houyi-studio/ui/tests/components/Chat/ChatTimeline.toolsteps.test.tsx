import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatTimeline } from '@/components/Chat/ChatTimeline';

function buildMessages(count: number) {
  const now = Date.now() / 1000;
  return Array.from({ length: count }, (_, index) => ({
    message_id: `m-${index + 1}`,
    role: index % 2 === 0 ? 'user' as const : 'assistant' as const,
    content: `message ${index + 1}`,
    metadata: {},
    created_at: now + index,
  }));
}

function installScrollableTimelineLayout() {
  const scrollTopAssignments: number[] = [];
  const scrollTopValues = new WeakMap<Element, number>();

  const scrollTopDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollTop');
  const scrollHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollHeight');
  const clientHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientHeight');
  const rectSpy = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.dataset.testid === 'chat-timeline') {
      return {
        x: 0,
        y: 0,
        top: 0,
        left: 0,
        right: 320,
        bottom: 400,
        width: 320,
        height: 400,
        toJSON: () => ({}),
      } as DOMRect;
    }

    const messageId = this.dataset.messageId;
    if (messageId) {
      const numericId = Number(messageId.replace('m-', ''));
      const top = (numericId % 10) * 40;
      return {
        x: 0,
        y: top,
        top,
        left: 0,
        right: 320,
        bottom: top + 32,
        width: 320,
        height: 32,
        toJSON: () => ({}),
      } as DOMRect;
    }

    return {
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      width: 0,
      height: 0,
      toJSON: () => ({}),
    } as DOMRect;
  });

  Object.defineProperty(HTMLElement.prototype, 'scrollTop', {
    configurable: true,
    get() {
      return scrollTopValues.get(this) ?? 0;
    },
    set(value: number) {
      scrollTopValues.set(this, value);
      if ((this as HTMLElement).dataset.testid === 'chat-timeline') {
        scrollTopAssignments.push(value);
      }
    },
  });

  Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
    configurable: true,
    get() {
      return (this as HTMLElement).dataset.testid === 'chat-timeline' ? 2400 : 0;
    },
  });

  Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
    configurable: true,
    get() {
      return (this as HTMLElement).dataset.testid === 'chat-timeline' ? 400 : 0;
    },
  });

  return {
    scrollTopAssignments,
    restore() {
      rectSpy.mockRestore();
      if (scrollTopDescriptor) {
        Object.defineProperty(HTMLElement.prototype, 'scrollTop', scrollTopDescriptor);
      } else {
        delete (HTMLElement.prototype as unknown as Record<string, unknown>).scrollTop;
      }
      if (scrollHeightDescriptor) {
        Object.defineProperty(HTMLElement.prototype, 'scrollHeight', scrollHeightDescriptor);
      } else {
        delete (HTMLElement.prototype as unknown as Record<string, unknown>).scrollHeight;
      }
      if (clientHeightDescriptor) {
        Object.defineProperty(HTMLElement.prototype, 'clientHeight', clientHeightDescriptor);
      } else {
        delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientHeight;
      }
    },
  };
}

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
            metadata: {
              tool_status: 'ok',
              round_index: 1,
              parallel_group_id: 'round_1',
              duration_ms: 1485,
            },
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

    expect(screen.getAllByText('done').length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    expect(screen.queryByText('houyi_read_file')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
    expect(screen.getByText('houyi_read_file')).toBeInTheDocument();
    expect(screen.getAllByText('Duration 1.5s').length).toBeGreaterThan(0);
    expect(screen.getByText('Parallel round_1')).toBeInTheDocument();
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

    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
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
    expect(screen.queryByText('I will search now.')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tool activity 1' }));
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

    expect(screen.getByRole('button', { name: 'Tool activity 1' })).toBeInTheDocument();
    expect(screen.queryByText('houyi_find_files')).not.toBeInTheDocument();
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

  it('renders only the most recent 120 messages until Show more is clicked', () => {
    render(
      <ChatTimeline
        conversationId="c7"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={buildMessages(160)}
      />,
    );

    expect(screen.getAllByTestId('message-bubble')).toHaveLength(120);
    expect(screen.queryByText('message 1')).not.toBeInTheDocument();
    expect(screen.getByText('message 160')).toBeInTheDocument();
    expect(screen.getByTestId('chat-timeline-show-more')).toBeInTheDocument();
    expect(screen.getByText('Showing 120 / 160')).toBeInTheDocument();
  });

  it('keeps Show more visible until all older history is explicitly revealed', () => {
    render(
      <ChatTimeline
        conversationId="c8"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={buildMessages(260)}
      />,
    );

    const showMore = screen.getByTestId('chat-timeline-show-more');
    fireEvent.click(showMore);

    expect(screen.getAllByTestId('message-bubble')).toHaveLength(240);
    expect(screen.getByTestId('chat-timeline-show-more')).toBeInTheDocument();
    expect(screen.getByText('Showing 240 / 260')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('chat-timeline-show-more'));

    expect(screen.getAllByTestId('message-bubble')).toHaveLength(260);
    expect(screen.queryByTestId('chat-timeline-show-more')).not.toBeInTheDocument();
  });

  it('preserves the current scroll anchor when Show more reveals older messages', () => {
    const layout = installScrollableTimelineLayout();
    try {
      render(
        <ChatTimeline
          conversationId="c10"
          streamingMessageId={null}
          isWaitingForResponse={false}
          messages={buildMessages(260)}
        />,
      );

      const timeline = screen.getByTestId('chat-timeline');
      timeline.scrollTop = 180;
      layout.scrollTopAssignments.length = 0;

      fireEvent.click(screen.getByTestId('chat-timeline-show-more'));

      expect(layout.scrollTopAssignments).toContain(180);
      expect(screen.getByText('Showing 240 / 260')).toBeInTheDocument();
    } finally {
      layout.restore();
    }
  });

  it('preserves the scroll anchor on the final Show more expansion before the button disappears', () => {
    const layout = installScrollableTimelineLayout();
    try {
      render(
        <ChatTimeline
          conversationId="c11"
          streamingMessageId={null}
          isWaitingForResponse={false}
          messages={buildMessages(260)}
        />,
      );

      const timeline = screen.getByTestId('chat-timeline');

      timeline.scrollTop = 120;
      fireEvent.click(screen.getByTestId('chat-timeline-show-more'));

      layout.scrollTopAssignments.length = 0;
      timeline.scrollTop = 320;
      fireEvent.click(screen.getByTestId('chat-timeline-show-more'));

      expect(layout.scrollTopAssignments).toContain(320);
      expect(screen.queryByTestId('chat-timeline-show-more')).not.toBeInTheDocument();
    } finally {
      layout.restore();
    }
  });

  it('does not show progressive-loading controls when raw messages exceed 120 but folded timeline items do not', () => {
    const now = Date.now() / 1000;
    const messages = Array.from({ length: 80 }, (_, index) => {
      const offset = index * 2;
      return [
        {
          message_id: `u-fold-${index + 1}`,
          role: 'user' as const,
          content: `question ${index + 1}`,
          metadata: {},
          created_at: now + offset,
        },
        {
          message_id: `a-fold-${index + 1}`,
          role: 'assistant' as const,
          content: '',
          metadata: {},
          created_at: now + offset + 1,
        },
      ];
    }).flat();

    render(
      <ChatTimeline
        conversationId="c13"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={messages}
      />,
    );

    expect(screen.getAllByTestId('message-bubble')).toHaveLength(80);
    expect(screen.queryByTestId('chat-timeline-show-more')).not.toBeInTheDocument();
    expect(screen.queryByText(/Showing \d+ \/ \d+/)).not.toBeInTheDocument();
  });

  it('renders date dividers when messages cross calendar days', () => {
    const base = new Date('2026-03-09T10:00:00Z').getTime() / 1000;
    render(
      <ChatTimeline
        conversationId="c9"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={[
          {
            message_id: 'd1',
            role: 'user',
            content: 'day one',
            metadata: {},
            created_at: base,
          },
          {
            message_id: 'd2',
            role: 'assistant',
            content: 'day two',
            metadata: {},
            created_at: base + 86_400,
          },
        ]}
      />,
    );

    expect(screen.getAllByTestId('chat-date-divider')).toHaveLength(2);
    expect(screen.getByText('day one')).toBeInTheDocument();
    expect(screen.getByText('day two')).toBeInTheDocument();
  });

  it('keeps stable key across ids', () => {
    const first = [
      {
        message_id: 'tmp-1',
        ui_render_id: 'ui-user-1',
        role: 'user' as const,
        content: 'hello',
        metadata: {},
        created_at: 1,
      },
    ];

    const second = [
      {
        message_id: 'user-1',
        ui_render_id: 'ui-user-1',
        role: 'user' as const,
        content: 'hello',
        metadata: { persisted: true },
        created_at: 1,
      },
    ];

    const { rerender } = render(
      <ChatTimeline
        conversationId="c12"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={first}
      />,
    );

    const firstNode = screen.getByText('hello').closest('[data-testid="message-bubble"]');

    rerender(
      <ChatTimeline
        conversationId="c12"
        streamingMessageId={null}
        isWaitingForResponse={false}
        messages={second}
      />,
    );

    const secondNode = screen.getByText('hello').closest('[data-testid="message-bubble"]');
    expect(secondNode).toBe(firstNode);
  });
});
