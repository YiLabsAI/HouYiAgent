/**
 * Tests for DeepResearchWorkspace — phases, input, and actions.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DeepResearchWorkspace } from '@/components/Agent/DeepResearch/Workspace';

const mockStore: Record<string, any> = {
  phase: 'input',
  plan: null,
  progress: null,
  report: null,
  error: null,
  loading: false,
  createSession: vi.fn(),
  confirmAndExecute: vi.fn(),
  cancelSession: vi.fn(),
  reset: vi.fn(),
  events: [],
};

vi.mock('@/stores/useResearchStore', () => {
  const useResearchStore = vi.fn((selector?: (s: typeof mockStore) => unknown) =>
    selector ? selector(mockStore) : mockStore,
  );
  (useResearchStore as any).getState = () => mockStore;
  return { useResearchStore };
});

vi.mock('@/components/Agent/DeepResearch/PlanEditor', () => ({
  PlanEditor: (props: any) => (
    <div data-testid="plan-editor" data-version={props.plan?.version} />
  ),
}));
vi.mock('@/components/Agent/DeepResearch/ProgressPanel', () => ({
  ProgressPanel: () => <div data-testid="progress-panel" />,
}));
vi.mock('@/components/Agent/DeepResearch/ReportViewer', () => ({
  ReportViewer: () => <div data-testid="report-viewer" />,
}));

const samplePlan = {
  query: 'topic',
  version: 2,
  status: 'draft',
  sub_questions: [
    {
      question_id: 'q1',
      question: 'Q?',
      priority: 1,
      search_strategy: 'web',
      expected_sources: 1,
      depends_on: [] as string[],
    },
  ],
  outline: [] as { title: string; description: string; related_question_ids: string[] }[],
};

const sampleReport = {
  title: 'Report',
  sections: [{ title: 'S', content: 'c', citations: [] }],
  references: [],
  quality_score: null,
};

describe('DeepResearchWorkspace', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStore.phase = 'input';
    mockStore.plan = null;
    mockStore.progress = null;
    mockStore.report = null;
    mockStore.error = null;
    mockStore.loading = false;
    mockStore.createSession = vi.fn().mockResolvedValue(undefined);
    mockStore.confirmAndExecute = vi.fn();
    mockStore.cancelSession = vi.fn();
    mockStore.reset = vi.fn();
    mockStore.events = [];
  });

  it('renders input phase — textarea and Start Research disabled when empty', () => {
    render(<DeepResearchWorkspace />);
    expect(screen.getByPlaceholderText(/What would you like to research/)).toBeInTheDocument();
    const btn = screen.getByRole('button', { name: /Start Research/i });
    expect(btn).toBeDisabled();
  });

  it('enables button with text', () => {
    render(<DeepResearchWorkspace />);
    const textarea = screen.getByPlaceholderText(/What would you like to research/);
    fireEvent.change(textarea, { target: { value: '  hello  ' } });
    expect(screen.getByRole('button', { name: /Start Research/i })).not.toBeDisabled();
  });

  it('calls createSession on submit', async () => {
    render(<DeepResearchWorkspace />);
    fireEvent.change(screen.getByPlaceholderText(/What would you like to research/), {
      target: { value: '  my topic  ' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Research/i }));
    expect(mockStore.createSession).toHaveBeenCalledWith('my topic', { depth: 'standard' });
  });

  it('shows planning phase', () => {
    mockStore.phase = 'planning';
    mockStore.plan = samplePlan;
    render(<DeepResearchWorkspace />);
    expect(screen.getByTestId('plan-editor')).toBeInTheDocument();
    expect(screen.getByTestId('plan-editor')).toHaveAttribute('data-version', '2');
  });

  it('shows executing phase', () => {
    mockStore.phase = 'executing';
    render(<DeepResearchWorkspace />);
    expect(screen.getByTestId('progress-panel')).toBeInTheDocument();
  });

  it('shows report phase', () => {
    mockStore.phase = 'report';
    mockStore.report = sampleReport;
    render(<DeepResearchWorkspace />);
    expect(screen.getByTestId('report-viewer')).toBeInTheDocument();
  });

  it('shows error banner', () => {
    mockStore.error = 'Something went wrong';
    render(<DeepResearchWorkspace />);
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('New Research button shows in non-input', () => {
    mockStore.phase = 'planning';
    mockStore.plan = samplePlan;
    render(<DeepResearchWorkspace />);
    expect(screen.getByRole('button', { name: /New Research/i })).toBeInTheDocument();
  });

  it('New Research calls reset', () => {
    mockStore.phase = 'planning';
    mockStore.plan = samplePlan;
    render(<DeepResearchWorkspace />);
    fireEvent.click(screen.getByRole('button', { name: /New Research/i }));
    expect(mockStore.reset).toHaveBeenCalled();
  });

  it('empty query not submitted', () => {
    render(<DeepResearchWorkspace />);
    fireEvent.click(screen.getByRole('button', { name: /Start Research/i }));
    expect(mockStore.createSession).not.toHaveBeenCalled();
  });
});
