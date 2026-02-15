/**
 * Tests for BottomPanel after P3-15/P3-16 refactor.
 *
 * Covers:
 *   - New 5-tab structure (observability, checkpoints, context, logs, knowledge)
 *   - Checkpoints dual-view (list / compare)
 *   - Expand button (P3-17)
 *   - Collapsed summary text
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BottomPanel } from '@/components/BottomPanel';
import { useConsoleStore } from '@/stores/useConsoleStore';

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: vi.fn(),
}));

vi.mock('@/components/panels/ComparePanel', () => ({
  ComparePanel: () => <div data-testid="compare-panel">ComparePanel</div>,
}));

vi.mock('@/components/panels/KnowledgeResultsPanel', () => ({
  KnowledgeResultsPanel: () => <div data-testid="knowledge-panel">KnowledgeResultsPanel</div>,
}));

vi.mock('@/components/panels/LogsPanel', () => ({
  LogsPanel: () => <div data-testid="logs-panel">LogsPanel</div>,
}));

vi.mock('@/components/panels/MetricsPanel', () => ({
  MetricsPanel: () => <div data-testid="metrics-panel">MetricsPanel</div>,
}));

vi.mock('@/components/panels/ObsFullView', () => ({
  ObsFullView: () => <div data-testid="obs-full-view">ObsFullView</div>,
}));

vi.mock('@/components/panels/TimelineWaterfall', () => ({
  TimelineWaterfall: () => <div data-testid="timeline">TimelineWaterfall</div>,
}));

const mockConsoleStore = (overrides: Record<string, unknown> = {}) => {
  const state = {
    checkpoints: [],
    selectedCheckpointKey: null,
    loadCheckpoint: vi.fn(),
    exitCheckpointView: vi.fn(),
    currentExecution: null,
    liveExecution: null,
    checkpointExecution: null,
    viewMode: 'live',
    sendCommand: vi.fn(),
    sessionId: 'test-session',
    currentPlan: null,
    prepareRestoreFromCheckpoint: vi.fn(),
    clearCurrentExecutionOutputsForFreshReplay: vi.fn(),
    executionLineageMap: {},
    ...overrides,
  };
  const fn = useConsoleStore as unknown as ReturnType<typeof vi.fn>;
  fn.mockImplementation((selector?: (s: typeof state) => unknown) =>
    selector ? selector(state) : state,
  );
};

describe('BottomPanel', () => {
  beforeEach(() => {
    mockConsoleStore();
  });

  // Tab rendering
  it('renders all 5 tabs', () => {
    render(<BottomPanel isCollapsed={false} onToggleCollapse={vi.fn()} />);
    expect(screen.getByText('Observability')).toBeInTheDocument();
    expect(screen.getByText('Checkpoints')).toBeInTheDocument();
    expect(screen.getByText('Context')).toBeInTheDocument();
    expect(screen.getByText('Logs')).toBeInTheDocument();
    expect(screen.getByText('Knowledge')).toBeInTheDocument();
  });

  it('does NOT render old tabs (compare, skills, execution-replay)', () => {
    render(<BottomPanel isCollapsed={false} onToggleCollapse={vi.fn()} />);
    expect(screen.queryByText('Compare')).not.toBeInTheDocument();
    expect(screen.queryByText('Skills')).not.toBeInTheDocument();
    expect(screen.queryByText('Execution & Replay')).not.toBeInTheDocument();
  });

  // Collapsed
  it('shows summary text when collapsed', () => {
    render(<BottomPanel isCollapsed={true} onToggleCollapse={vi.fn()} />);
    expect(screen.getByText('Checkpoints')).toBeInTheDocument();
    expect(screen.getByText('Observability')).toBeInTheDocument();
    expect(screen.getByText('Context')).toBeInTheDocument();
  });

  // Tab switching
  it('switches to logs tab when clicked', () => {
    const onTabChange = vi.fn();
    render(
      <BottomPanel
        isCollapsed={false}
        onToggleCollapse={vi.fn()}
        activeTab="checkpoints"
        onTabChange={onTabChange}
      />,
    );
    fireEvent.click(screen.getByText('Logs'));
    expect(onTabChange).toHaveBeenCalledWith('logs');
  });

  // Expand button (P3-17)
  it('calls onExpandTab with current tab when expand button is clicked', () => {
    const onExpandTab = vi.fn();
    render(
      <BottomPanel
        isCollapsed={false}
        onToggleCollapse={vi.fn()}
        activeTab="observability"
        onExpandTab={onExpandTab}
      />,
    );
    fireEvent.click(screen.getByTitle('Expand to Center Stage'));
    expect(onExpandTab).toHaveBeenCalledWith('observability');
  });

  // Checkpoints: empty state
  it('shows empty state when no checkpoints', () => {
    render(
      <BottomPanel
        isCollapsed={false}
        onToggleCollapse={vi.fn()}
        activeTab="checkpoints"
      />,
    );
    expect(screen.getByText('No checkpoints created yet')).toBeInTheDocument();
    expect(screen.getByText(/Check any 2 checkpoints/)).toBeInTheDocument();
  });

  // Collapse toggle
  it('calls onToggleCollapse when collapse button is clicked', () => {
    const onToggle = vi.fn();
    render(<BottomPanel isCollapsed={false} onToggleCollapse={onToggle} />);
    fireEvent.click(screen.getByTitle('Collapse panel'));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it('calls onToggleCollapse when expand button is clicked in collapsed state', () => {
    const onToggle = vi.fn();
    render(<BottomPanel isCollapsed={true} onToggleCollapse={onToggle} />);
    fireEvent.click(screen.getByTitle('Expand panel'));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  // ─── Fusion: Checkpoints + Compare ────────────────────────────────

  const makeCheckpoints = (count: number) =>
    Array.from({ length: count }, (_, i) => ({
      checkpoint_id: `cp-${i}`,
      execution_id: 'exec-1',
      sequence_number: i + 1,
      created_at: new Date(Date.now() + i * 1000).toISOString(),
      trigger: 'auto',
      metadata: {},
    }));

  describe('Fusion: checkpoint selection & compare', () => {
    it('renders checkboxes on each checkpoint item', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(3),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      // First execution group auto-expands, so checkboxes should be visible
      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes).toHaveLength(3);
      checkboxes.forEach((cb) => {
        expect(cb).not.toBeChecked();
      });
    });

    it('checking 2 checkpoints auto-switches to compare view', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(3),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      const checkboxes = screen.getAllByRole('checkbox');
      // Check first two
      fireEvent.click(checkboxes[0]);
      fireEvent.click(checkboxes[1]);
      // Should now show ComparePanel
      expect(screen.getByTestId('compare-panel')).toBeInTheDocument();
    });

    it('shows "Back to List" button in compare view', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(3),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[0]);
      fireEvent.click(checkboxes[1]);
      expect(screen.getByText('Back to List')).toBeInTheDocument();
    });

    it('"Back to List" returns to list view and clears selection', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(3),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[0]);
      fireEvent.click(checkboxes[1]);
      // Now in compare view
      expect(screen.getByTestId('compare-panel')).toBeInTheDocument();
      // Click back
      fireEvent.click(screen.getByText('Back to List'));
      // Should be back to list, ComparePanel gone
      expect(screen.queryByTestId('compare-panel')).not.toBeInTheDocument();
      // Checkboxes should be unchecked (selection cleared)
      const cbs = screen.getAllByRole('checkbox');
      cbs.forEach((cb) => expect(cb).not.toBeChecked());
    });

    it('shows "Clear" button in header when checkpoints are selected', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(2),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[0]);
      expect(screen.getByTitle('Clear checkpoint selection')).toBeInTheDocument();
    });

    it('"Clear" button resets selection to empty', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(2),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[0]);
      // Clear button should be visible
      fireEvent.click(screen.getByTitle('Clear checkpoint selection'));
      // No checkbox should be checked
      const cbs = screen.getAllByRole('checkbox');
      cbs.forEach((cb) => expect(cb).not.toBeChecked());
    });

    it('shows selection hint when 1 checkpoint is checked', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(3),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[0]);
      expect(screen.getByText(/Select 1 more checkpoint to compare/)).toBeInTheDocument();
    });

    it('unchecking a checkbox removes it from selection', () => {
      mockConsoleStore({
        checkpoints: makeCheckpoints(3),
        executionLineageMap: {},
      });
      render(
        <BottomPanel
          isCollapsed={false}
          onToggleCollapse={vi.fn()}
          activeTab="checkpoints"
        />,
      );
      const checkboxes = screen.getAllByRole('checkbox');
      // Check first
      fireEvent.click(checkboxes[0]);
      expect(checkboxes[0]).toBeChecked();
      // Uncheck it
      fireEvent.click(checkboxes[0]);
      expect(checkboxes[0]).not.toBeChecked();
      // No "Select N more" hint should appear since nothing is checked
      expect(screen.queryByText(/Select.*more checkpoint/)).not.toBeInTheDocument();
    });
  });
});
