import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ToolCallBubble } from '@/components/Chat/ToolCallBubble';

describe('ToolCallBubble', () => {
  it('renders tool name and status', () => {
    render(
      <ToolCallBubble
        message={{
          message_id: 'm1',
          role: 'tool',
          content: '{"ok":true}',
          name: 'web_search',
          tool_call_id: 'call-1',
          metadata: { tool_status: 'running' },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.getByText('web_search')).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
  });

  it('expands and shows payload', () => {
    render(
      <ToolCallBubble
        message={{
          message_id: 'm2',
          role: 'tool',
          content: '{"result":"ok"}',
          name: 'houyi_read_file',
          tool_call_id: 'call-2',
          metadata: { tool_status: 'ok' },
          created_at: Date.now() / 1000,
        }}
      />,
    );

    expect(screen.queryByText('{"result":"ok"}')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button'));
    expect(screen.getByText(/"result": "ok"/)).toBeInTheDocument();
  });

  it('keeps long meta chips within the bubble width', () => {
    render(
      <ToolCallBubble
        message={{
          message_id: 'm3',
          role: 'tool',
          name: 'houyi_read_file',
          content: JSON.stringify({ data: { path: '/tmp/file.txt' } }),
          metadata: {
            tool_status: 'ok',
            parallel_group_id: 'parallel-group-with-a-very-long-identifier-that-should-wrap-inside-the-bubble',
            round_index: 2,
            duration_ms: 1280,
          },
          created_at: 1,
        } as any}
      />,
    );

    fireEvent.click(screen.getByRole('button'));

    const metaSection = screen.getByText('Meta').parentElement;
    expect(metaSection).toHaveClass('min-w-0', 'max-w-full');
    const wrappedMetaValue = screen.getAllByText(/parallel-group-with-a-very-long-identifier/i)
      .find((node) => node.className.includes('break-all'));
    expect(wrappedMetaValue).toBeDefined();
    expect(wrappedMetaValue).toHaveClass('min-w-0', 'break-all');
  });

  it('surfaces concise error summary for failed tools', () => {
    render(
      <ToolCallBubble
        message={{
          message_id: 'm4',
          role: 'tool',
          name: 'houyi_find_files',
          content: JSON.stringify({ error: 'permission_denied', message: 'Workspace access denied for /private/repo' }),
          metadata: { tool_status: 'error' },
          created_at: 1,
        } as any}
      />,
    );

    expect(screen.getByText('Error: Workspace access denied for /private/repo')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button'));
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getAllByText('Workspace access denied for /private/repo').length).toBeGreaterThan(0);
  });
});
