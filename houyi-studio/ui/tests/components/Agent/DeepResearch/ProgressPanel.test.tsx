/**
 * Tests for ProgressPanel — progress bar, ThinkingTrajectory integration, cancel.
 */
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

  it('renders ThinkingTrajectory with events', () => {
    const events: SSEEvent[] = [
      { event_id: 'e1', event_type: 'research.step_started', sequence: 1, payload: { step_id: 'sq1', step: 'Q1' } },
      { event_id: 'e2', event_type: 'research.source_found', sequence: 2, payload: { question_id: 'sq1', title: 'Source A' } },
    ];
    render(<ProgressPanel progress={baseProgress()} events={events} onCancel={vi.fn()} />);
    expect(screen.getByText('Q1')).toBeInTheDocument();
    expect(screen.getByText('Thinking Trajectory')).toBeInTheDocument();
  });

  it('passes subQuestions to ThinkingTrajectory', () => {
    const events: SSEEvent[] = [
      { event_id: 'e1', event_type: 'research.step_started', sequence: 1, payload: { step_id: 'sq1', step: 'short' } },
    ];
    const subQuestions = [{
      question_id: 'sq1',
      question: 'Full question text here',
      priority: 5,
      search_strategy: 'web',
      expected_sources: 5,
      depends_on: [],
    }];
    render(
      <ProgressPanel progress={baseProgress()} events={events} subQuestions={subQuestions} onCancel={vi.fn()} />,
    );
    expect(screen.getByText('Full question text here')).toBeInTheDocument();
  });

  it('shows empty state via ThinkingTrajectory', () => {
    render(<ProgressPanel progress={baseProgress()} events={[]} onCancel={vi.fn()} />);
    expect(screen.getByText('Waiting for research events...')).toBeInTheDocument();
  });

  it('cancel button calls callback', () => {
    const onCancel = vi.fn();
    render(<ProgressPanel progress={baseProgress()} events={[]} onCancel={onCancel} />);
    fireEvent.click(screen.getByRole('button', { name: /Cancel Research/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('hides cancel when terminated (error)', () => {
    render(
      <ProgressPanel progress={baseProgress()} events={[]} onCancel={vi.fn()} error="Something failed" />,
    );
    expect(screen.queryByRole('button', { name: /Cancel Research/i })).not.toBeInTheDocument();
    expect(screen.getByText('Stopped')).toBeInTheDocument();
  });
});
