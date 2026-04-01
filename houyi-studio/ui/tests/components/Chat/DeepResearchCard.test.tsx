import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DeepResearchCard } from '@/components/Chat/DeepResearchCard';

describe('DeepResearchCard', () => {
  it('renders query text', () => {
    render(<DeepResearchCard query="AI trends" />);
    expect(screen.getByText(/AI trends/)).toBeInTheDocument();
  });

  it('shows Deep Research header', () => {
    render(<DeepResearchCard query="q" />);
    expect(screen.getByText('Deep Research')).toBeInTheDocument();
  });

  it('shows pending status', () => {
    render(<DeepResearchCard query="q" />);
    expect(screen.getByText('Queued')).toBeInTheDocument();
  });

  it('shows running status', () => {
    render(<DeepResearchCard query="q" status="running" />);
    expect(screen.getByText('Researching...')).toBeInTheDocument();
  });

  it('shows completed status', () => {
    render(<DeepResearchCard query="q" status="completed" />);
    expect(screen.getByText('Complete')).toBeInTheDocument();
  });

  it('shows failed status', () => {
    render(<DeepResearchCard query="q" status="failed" />);
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('shows summary when provided', () => {
    render(<DeepResearchCard query="q" summary="Found 10 sources" />);
    expect(screen.getByText('Found 10 sources')).toBeInTheDocument();
  });

  it('hides summary when absent', () => {
    render(<DeepResearchCard query="test query without summary phrase" />);
    expect(screen.queryByText('Found 10 sources')).not.toBeInTheDocument();
  });

  it('Open in Workspace button', () => {
    render(
      <DeepResearchCard
        query="q"
        sessionId="sess-1"
        status="completed"
        onOpenWorkspace={vi.fn()}
      />,
    );
    expect(screen.getByRole('button', { name: /Open in Workspace/i })).toBeInTheDocument();
  });

  it('clicking workspace button calls handler', () => {
    const onOpenWorkspace = vi.fn();
    render(
      <DeepResearchCard
        query="q"
        sessionId="sess-99"
        status="completed"
        onOpenWorkspace={onOpenWorkspace}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Open in Workspace/i }));
    expect(onOpenWorkspace).toHaveBeenCalledWith('sess-99');
  });

  it('no workspace button without sessionId', () => {
    render(<DeepResearchCard query="q" status="completed" onOpenWorkspace={vi.fn()} />);
    expect(screen.queryByRole('button', { name: /Open in Workspace/i })).not.toBeInTheDocument();
  });

  it('no workspace button when not completed', () => {
    render(
      <DeepResearchCard
        query="q"
        sessionId="sess-1"
        status="running"
        onOpenWorkspace={vi.fn()}
      />,
    );
    expect(screen.queryByRole('button', { name: /Open in Workspace/i })).not.toBeInTheDocument();
  });
});
