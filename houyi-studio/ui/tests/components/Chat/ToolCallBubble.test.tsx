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
    expect(screen.getByText('{"result":"ok"}')).toBeInTheDocument();
  });
});
