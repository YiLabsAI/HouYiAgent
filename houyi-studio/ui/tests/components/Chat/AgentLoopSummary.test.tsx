import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AgentLoopSummary } from '@/components/Chat/AgentLoopSummary';

describe('AgentLoopSummary', () => {
  it('does not render when summary is empty', () => {
    const { container } = render(
      <AgentLoopSummary rounds={0} toolCalls={0} traceId={null} usage={null} />,
    );
    expect(container.innerHTML).toBe('');
  });

  it('renders summary and toggles expanded details', () => {
    render(
      <AgentLoopSummary
        rounds={2}
        toolCalls={3}
        traceId="trace-123"
        usage={{ prompt_tokens: 34, completion_tokens: 54, total_tokens: 88 }}
      />,
    );

    expect(screen.getByText('Agent Loop')).toBeInTheDocument();
    expect(screen.getByText(/2 rounds/)).toBeInTheDocument();
    expect(screen.getByText('In 34')).toBeInTheDocument();
    expect(screen.getByText('Out 54')).toBeInTheDocument();
    expect(screen.getByText('Total 88')).toBeInTheDocument();
    expect(screen.queryByText(/Iterations:/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button'));
    expect(screen.getByText(/Iterations: 2/)).toBeInTheDocument();
    expect(screen.getByText(/Tool calls: 3/)).toBeInTheDocument();
    expect(screen.getByText('trace-123')).toBeInTheDocument();
  });

  it('invokes onOpenTrace from details link', () => {
    const onOpenTrace = vi.fn();
    render(
      <AgentLoopSummary
        rounds={1}
        toolCalls={1}
        traceId="trace-xyz"
        usage={null}
        onOpenTrace={onOpenTrace}
      />,
    );

    fireEvent.click(screen.getByRole('button'));
    fireEvent.click(screen.getByText('details'));
    expect(onOpenTrace).toHaveBeenCalledWith('trace-xyz');
  });
});
