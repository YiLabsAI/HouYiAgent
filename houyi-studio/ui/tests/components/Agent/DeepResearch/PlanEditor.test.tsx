/**
 * Tests for PlanEditor — plan display, outline, add row, confirm.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PlanEditor } from '@/components/Agent/DeepResearch/PlanEditor';

const mockEditPlan = vi.fn();

vi.mock('@/stores/useResearchStore', () => ({
  useResearchStore: vi.fn(() => ({ editPlan: mockEditPlan })),
}));

const makePlan = (overrides = {}) => ({
  query: 'AI research',
  version: 1,
  status: 'draft',
  sub_questions: [
    {
      question_id: 'q1',
      question: 'What is GPT?',
      priority: 3,
      search_strategy: 'web',
      expected_sources: 5,
      depends_on: [],
    },
    {
      question_id: 'q2',
      question: 'What is Claude?',
      priority: 2,
      search_strategy: 'academic',
      expected_sources: 3,
      depends_on: [],
    },
  ],
  outline: [{ title: 'Introduction', description: 'Overview', related_question_ids: ['q1'] }],
  ...overrides,
});

describe('PlanEditor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders plan version', () => {
    render(<PlanEditor plan={makePlan() as any} onConfirm={vi.fn()} loading={false} />);
    expect(screen.getByText('v1')).toBeInTheDocument();
  });

  it('renders sub-question count', () => {
    render(<PlanEditor plan={makePlan() as any} onConfirm={vi.fn()} loading={false} />);
    expect(screen.getByText('2 sub-questions')).toBeInTheDocument();
  });

  it('renders query topic', () => {
    render(<PlanEditor plan={makePlan() as any} onConfirm={vi.fn()} loading={false} />);
    expect(screen.getByText('AI research')).toBeInTheDocument();
  });

  it('renders each sub-question', () => {
    render(<PlanEditor plan={makePlan() as any} onConfirm={vi.fn()} loading={false} />);
    expect(screen.getByDisplayValue('What is GPT?')).toBeInTheDocument();
    expect(screen.getByDisplayValue('What is Claude?')).toBeInTheDocument();
  });

  it('renders outline sections', () => {
    render(<PlanEditor plan={makePlan() as any} onConfirm={vi.fn()} loading={false} />);
    expect(screen.getByText('Introduction')).toBeInTheDocument();
  });

  it('add input and button', () => {
    render(<PlanEditor plan={makePlan() as any} onConfirm={vi.fn()} loading={false} />);
    expect(screen.getByPlaceholderText('Add a sub-question...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Add$/i })).toBeInTheDocument();
  });

  it('confirm button calls onConfirm', () => {
    const onConfirm = vi.fn();
    render(<PlanEditor plan={makePlan() as any} onConfirm={onConfirm} loading={false} />);
    fireEvent.click(screen.getByRole('button', { name: /Confirm & Execute/i }));
    expect(onConfirm).toHaveBeenCalled();
  });

  it('confirm disabled with no questions', () => {
    const plan = makePlan({ sub_questions: [] });
    render(<PlanEditor plan={plan as any} onConfirm={vi.fn()} loading={false} />);
    expect(screen.getByRole('button', { name: /Confirm & Execute/i })).toBeDisabled();
  });

  it('confirm disabled when loading', () => {
    render(<PlanEditor plan={makePlan() as any} onConfirm={vi.fn()} loading />);
    expect(screen.getByRole('button', { name: /Confirm & Execute/i })).toBeDisabled();
  });
});
