import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { ObsFullView } from '@/components/panels/ObsFullView';

const mockState = {
  spanStore: {
    exec_1: { spans: [] },
    exec_2: { spans: [] },
  },
  currentExecution: { execution_id: 'exec_1' },
  liveExecution: null,
  getSpanTree: vi.fn(() => null),
};

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: (selector?: (s: typeof mockState) => unknown) =>
    selector ? selector(mockState) : mockState,
}));

vi.mock('@/components/panels/ExecutionLineageTree', () => ({
  ExecutionLineageTree: ({
    executionIds,
    onSelect,
  }: {
    executionIds: string[];
    onSelect: (id: string) => void;
  }) => (
    <div>
      <div>Mock Execution Tree</div>
      {executionIds.map((id) => (
        <button key={id} type="button" onClick={() => onSelect(id)}>
          {id}
        </button>
      ))}
    </div>
  ),
}));

vi.mock('@/components/panels/TimelineWaterfall', () => ({
  TimelineWaterfall: ({ executionId }: { executionId: string }) => (
    <div>Waterfall: {executionId}</div>
  ),
}));

vi.mock('@/components/panels/MetricsPanel', () => ({
  MetricsPanel: ({ executionId }: { executionId?: string }) => (
    <div>Metrics: {executionId ?? 'none'}</div>
  ),
}));

describe('ObsFullView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing when closed', () => {
    const { container } = render(<ObsFullView isOpen={false} onClose={vi.fn()} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders in CenterStage and shows default execution details', () => {
    render(<ObsFullView isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/Observability/)).toBeInTheDocument();
    expect(screen.getByText('Mock Execution Tree')).toBeInTheDocument();
    expect(screen.getByText('Waterfall: exec_1')).toBeInTheDocument();
    expect(screen.getByText('Metrics: exec_1')).toBeInTheDocument();
  });

  it('updates selected execution when clicking in tree', () => {
    render(<ObsFullView isOpen={true} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'exec_2' }));
    expect(screen.getByText('Waterfall: exec_2')).toBeInTheDocument();
    expect(screen.getByText('Metrics: exec_2')).toBeInTheDocument();
  });

  it('closes on backdrop click', () => {
    const onClose = vi.fn();
    render(<ObsFullView isOpen={true} onClose={onClose} />);
    fireEvent.click(screen.getByTestId('center-stage-backdrop'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
