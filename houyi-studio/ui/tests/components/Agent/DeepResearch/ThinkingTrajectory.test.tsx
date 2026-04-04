/**
 * Tests for ThinkingTrajectory — SSE event grouping by sub-question,
 * expand/collapse, source counts, search queries, and pipeline events.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ThinkingTrajectory } from '@/components/Agent/DeepResearch/ThinkingTrajectory';
import type { SSEEvent, SubQuestion } from '@/stores/useResearchStore';

const sq = (id: string, question: string): SubQuestion => ({
  question_id: id,
  question,
  priority: 5,
  search_strategy: 'web',
  expected_sources: 5,
  depends_on: [],
});

const evt = (
  id: string,
  type: string,
  seq: number,
  payload: Record<string, unknown> = {},
): SSEEvent => ({ event_id: id, event_type: type, sequence: seq, payload });

describe('ThinkingTrajectory', () => {
  it('shows waiting message when no events', () => {
    render(<ThinkingTrajectory events={[]} />);
    expect(screen.getByText('Waiting for research events...')).toBeInTheDocument();
  });

  it('groups events by question_id into collapsible cards', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'Question one' }),
      evt('e2', 'research.source_found', 2, { question_id: 'sq_1', title: 'Src A', url: 'https://a.com' }),
      evt('e3', 'research.step_completed', 3, { step_id: 'sq_1', step: 'Question one' }),
      evt('e4', 'research.step_started', 4, { step_id: 'sq_2', step: 'Question two' }),
    ];
    render(<ThinkingTrajectory events={events} />);
    expect(screen.getByText('Question one')).toBeInTheDocument();
    expect(screen.getByText('Question two')).toBeInTheDocument();
  });

  it('uses subQuestion labels when provided', () => {
    const subs = [sq('sq_1', 'What is the capital of France?')];
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'short' }),
    ];
    render(<ThinkingTrajectory events={events} subQuestions={subs} />);
    expect(screen.getByText('What is the capital of France?')).toBeInTheDocument();
    expect(screen.queryByText('short')).not.toBeInTheDocument();
  });

  it('displays source count badge with label per group', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'Q1' }),
      evt('e2', 'research.source_found', 2, { question_id: 'sq_1', title: 'S1', url: 'https://1.com' }),
      evt('e3', 'research.source_found', 3, { question_id: 'sq_1', title: 'S2', url: 'https://2.com' }),
      evt('e4', 'research.source_found', 4, { question_id: 'sq_1', title: 'S3', url: 'https://3.com' }),
    ];
    render(<ThinkingTrajectory events={events} />);
    expect(screen.getByText('3 sources')).toBeInTheDocument();
  });

  it('expands group on click to show source details', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'Q1' }),
      evt('e2', 'research.source_found', 2, {
        question_id: 'sq_1',
        title: 'Found source X',
        url: 'https://example.com/x',
        snippet: 'A snippet about X',
      }),
    ];
    render(<ThinkingTrajectory events={events} />);
    expect(screen.queryByText('Found source X')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Q1'));
    expect(screen.getByText('Found source X')).toBeInTheDocument();
    expect(screen.getByText('A snippet about X')).toBeInTheDocument();
  });

  it('expands group to show search queries', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'Q1' }),
      evt('e2', 'research.search_queries', 2, {
        question_id: 'sq_1',
        round: 1,
        queries: ['AI frameworks 2026', 'multi agent architecture'],
      }),
    ];
    render(<ThinkingTrajectory events={events} />);
    fireEvent.click(screen.getByText('Q1'));
    expect(screen.getByText('Round 1')).toBeInTheDocument();
    expect(screen.getByText('"AI frameworks 2026"')).toBeInTheDocument();
    expect(screen.getByText('"multi agent architecture"')).toBeInTheDocument();
  });

  it('renders pipeline events (non-question) separately', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.quality_evaluated', 1, {}),
      evt('e2', 'research.report_section', 2, { chunk: { title: 'Introduction' } }),
    ];
    render(<ThinkingTrajectory events={events} />);
    expect(screen.getByText('Pipeline')).toBeInTheDocument();
    expect(screen.getByText('Quality evaluation complete')).toBeInTheDocument();
    expect(screen.getByText('Writing: Introduction')).toBeInTheDocument();
  });

  it('shows completed status icon for finished group', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'Q1' }),
      evt('e2', 'research.step_completed', 2, { step_id: 'sq_1', step: 'Q1' }),
    ];
    const { container } = render(<ThinkingTrajectory events={events} />);
    const checkIcon = container.querySelector('.text-green-400');
    expect(checkIcon).toBeTruthy();
  });

  it('shows red icon (not spinner) for failed pipeline event', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.failed', 1, { error: 'Research timed out after 900s' }),
    ];
    const { container } = render(<ThinkingTrajectory events={events} />);
    expect(screen.getByText('Research timed out after 900s')).toBeInTheDocument();
    const redIcons = container.querySelectorAll('.text-red-400');
    expect(redIcons.length).toBeGreaterThan(0);
    const spinners = container.querySelectorAll('.animate-spin');
    expect(spinners.length).toBe(0);
  });

  it('does not show confusing sequence numbers in pipeline events', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.intermediate_report', 200, { question_id: 'sq_1' }),
    ];
    render(<ThinkingTrajectory events={events} />);
    expect(screen.getByText('Intermediate report: sq_1')).toBeInTheDocument();
    expect(screen.queryByText('#200')).not.toBeInTheDocument();
  });

  it('treats report_generation step events as pipeline, not question group', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'Q1' }),
      evt('e2', 'research.step_completed', 2, { step_id: 'sq_1', step: 'Q1' }),
      evt('e3', 'research.step_started', 3, { step_id: 'report_generation', step: 'Generating report...' }),
    ];
    render(<ThinkingTrajectory events={events} />);
    expect(screen.getByText('Q1')).toBeInTheDocument();
    // "Generating report..." should NOT appear as a question group card
    const allButtons = screen.getAllByRole('button');
    const labels = allButtons.map((b) => b.textContent);
    expect(labels.some((l) => l?.includes('Generating report'))).toBe(false);
  });

  it('forces searching questions to completed once report phase starts', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.step_started', 1, { step_id: 'sq_1', step: 'Q1' }),
      // step_completed missing — would stay "searching" without the fix
      evt('e2', 'research.intermediate_report', 2, { question_id: 'sq_1' }),
    ];
    const { container } = render(<ThinkingTrajectory events={events} />);
    // Should show green completed icon, not spinner
    const greenIcons = container.querySelectorAll('.text-green-400');
    expect(greenIcons.length).toBeGreaterThan(0);
    const questionSpinners = container.querySelectorAll('button .animate-spin');
    expect(questionSpinners.length).toBe(0);
  });

  it('shows pipeline phase label for research.pipeline_phase events', () => {
    const events: SSEEvent[] = [
      evt('e1', 'research.pipeline_phase', 1, { phase: 'report_generation' }),
    ];
    render(<ThinkingTrajectory events={events} />);
    expect(screen.getByText('Writing report sections...')).toBeInTheDocument();
  });
});
