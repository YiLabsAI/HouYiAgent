import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ProgressPanel } from '@/components/Agent/DeepResearch/ProgressPanel';
import type { ResearchProgress, SSEEvent } from '@/stores/useResearchStore';

const baseProgress = (overrides: Partial<ResearchProgress> = {}): ResearchProgress => ({
  total_steps: 10,
  completed_steps: 3,
  current_step: 'Searching...',
  elapsed_seconds: 45,
  sub_question_progress: {},
  ...overrides,
});

describe('ProgressPanel', () => {
  it('shows initializing when no progress', () => {
    render(<ProgressPanel progress={null} events={[]} onCancel={vi.fn()} />);
    expect(screen.getByText('Initializing...')).toBeInTheDocument();
  });

  it('shows percentage and step counts', () => {
    render(
      <ProgressPanel
        progress={baseProgress({ total_steps: 10, completed_steps: 3 })}
        events={[]}
        onCancel={vi.fn()}
      />,
    );
    expect(screen.getByText('30%')).toBeInTheDocument();
    expect(screen.getByText('3 / 10 search steps')).toBeInTheDocument();
  });

  it('shows current step name', () => {
    render(
      <ProgressPanel
        progress={baseProgress({ current_step: 'Searching...' })}
        events={[]}
        onCancel={vi.fn()}
      />,
    );
    expect(screen.getByText('Searching...')).toBeInTheDocument();
  });

  it('shows elapsed time', () => {
    render(
      <ProgressPanel
        progress={baseProgress({ elapsed_seconds: 45 })}
        events={[]}
        onCancel={vi.fn()}
      />,
    );
    expect(screen.getByText('45s elapsed')).toBeInTheDocument();
  });

  it('renders events in activity log', () => {
    const events: SSEEvent[] = [
      { event_id: 'e1', event_type: 'research.step_started', sequence: 1, payload: { step: 'Q1' } },
      { event_id: 'e2', event_type: 'research.step_started', sequence: 2, payload: { step: 'Q2' } },
      { event_id: 'e3', event_type: 'research.step_started', sequence: 3, payload: { step: 'Q3' } },
    ];
    render(<ProgressPanel progress={baseProgress()} events={events} onCancel={vi.fn()} />);
    expect(screen.getByText('Researching: Q1')).toBeInTheDocument();
    expect(screen.getByText('Researching: Q2')).toBeInTheDocument();
    expect(screen.getByText('Researching: Q3')).toBeInTheDocument();
  });

  it('shows empty state', () => {
    render(<ProgressPanel progress={baseProgress()} events={[]} onCancel={vi.fn()} />);
    expect(screen.getByText('Waiting for events...')).toBeInTheDocument();
  });

  it('cancel button calls callback', () => {
    const onCancel = vi.fn();
    render(<ProgressPanel progress={baseProgress()} events={[]} onCancel={onCancel} />);
    fireEvent.click(screen.getByRole('button', { name: /Cancel Research/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('limits display to 20 events', () => {
    const events: SSEEvent[] = Array.from({ length: 25 }, (_, i) => ({
      event_id: `id-${i + 1}`,
      event_type: 'research.step_started',
      sequence: i + 1,
      payload: { step: `S${i + 1}` },
    }));
    render(<ProgressPanel progress={baseProgress()} events={events} onCancel={vi.fn()} />);
    const seqLabels = screen.getAllByText(/#\d+/);
    expect(seqLabels).toHaveLength(20);
    expect(screen.queryByText('#1')).not.toBeInTheDocument();
    expect(screen.getByText('#25')).toBeInTheDocument();
  });

  it('event labels for different types', () => {
    const events: SSEEvent[] = [
      {
        event_id: 'a',
        event_type: 'research.step_completed',
        sequence: 1,
        payload: { step: 'S1' },
      },
      {
        event_id: 'b',
        event_type: 'research.source_found',
        sequence: 2,
        payload: { title: 'My Source', url: 'https://example.com' },
      },
      {
        event_id: 'c',
        event_type: 'research.quality_evaluated',
        sequence: 3,
        payload: {},
      },
    ];
    render(<ProgressPanel progress={baseProgress()} events={events} onCancel={vi.fn()} />);
    expect(screen.getByText('Completed: S1')).toBeInTheDocument();
    expect(screen.getByText('Found: My Source')).toBeInTheDocument();
    expect(screen.getByText('Quality evaluation complete')).toBeInTheDocument();
  });
});
