import { act } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { RightSidebar } from '@/components/RightSidebar';
import { useConsoleStore } from '@/stores/useConsoleStore';

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: vi.fn(),
}));

type StoreState = {
  selectedNodeId: string | null;
  nodes: any[];
  updateNode: (nodeId: string, updates: any) => void;
  viewMode: string;
  liveExecution: any;
  currentExecution: any;
  checkpointExecution: any;
};

const createStoreState = (overrides: Partial<StoreState> = {}): StoreState => ({
  selectedNodeId: 'tool_1',
  nodes: [
    {
      id: 'tool_1',
      type: 'tool',
      data: {
        label: 'tool_1',
        nodeType: 'tool',
        config: { tool_name: 'web_search' },
        inputs: { query: 'hello' },
        outputs: {},
        metadata: { label: 'tool_1' },
      },
    },
  ],
  updateNode: vi.fn(),
  viewMode: 'live',
  liveExecution: null,
  currentExecution: null,
  checkpointExecution: null,
  ...overrides,
});

describe('RightSidebar', () => {
  const mockedStore = useConsoleStore as unknown as ReturnType<typeof vi.fn>;

  it('should render with a selected node', () => {
    const state = createStoreState();
    mockedStore.mockImplementation((selector?: any) => (selector ? selector(state) : state));

    render(<RightSidebar isCollapsed={false} onToggleCollapse={vi.fn()} />);

    expect(screen.getByText('Properties')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter label')).toBeInTheDocument();
  });

  it('should update label and metadata when label changes', () => {
    const state = createStoreState();
    mockedStore.mockImplementation((selector?: any) => (selector ? selector(state) : state));

    render(<RightSidebar isCollapsed={false} onToggleCollapse={vi.fn()} />);

    const labelInput = screen.getByPlaceholderText('Enter label');
    act(() => {
      fireEvent.change(labelInput, { target: { value: 'web search tool' } });
      fireEvent.blur(labelInput);
    });

    expect(state.updateNode).toHaveBeenCalledTimes(1);
    const [nodeId, updates] = (state.updateNode as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(nodeId).toBe('tool_1');
    expect(updates.label).toBe('web search tool');
    expect(updates.metadata.label).toBe('web search tool');
  });

  it('should validate inputs JSON and prevent invalid save', () => {
    const state = createStoreState();
    mockedStore.mockImplementation((selector?: any) => (selector ? selector(state) : state));

    render(<RightSidebar isCollapsed={false} onToggleCollapse={vi.fn()} />);

    const inputsTab = screen.getByRole('button', { name: 'Inputs' });
    act(() => {
      fireEvent.click(inputsTab);
    });

    expect(screen.getByText('Inputs (runtime)')).toBeInTheDocument();
    expect(screen.getByText('Inputs (static mapping)')).toBeInTheDocument();
    expect(screen.getAllByText(/"query": "hello"/).length).toBeGreaterThan(0);
    expect(state.updateNode).not.toHaveBeenCalled();
  });

  it('should show execution inputs in inputs tab', () => {
    const state = createStoreState({
      currentExecution: {
        node_executions: {
          tool_1: {
            node_id: 'tool_1',
            status: 'completed',
            inputs: { query: 'from-exec', max_results: 3 },
            outputs: {},
            error: null,
            started_at: null,
            completed_at: null,
            streaming_output: '',
            metadata: {},
          },
        },
      },
    });
    mockedStore.mockImplementation((selector?: any) => (selector ? selector(state) : state));

    render(<RightSidebar isCollapsed={false} onToggleCollapse={vi.fn()} />);

    const inputsTab = screen.getByRole('button', { name: 'Inputs' });
    act(() => {
      fireEvent.click(inputsTab);
    });

    expect(screen.getByText(/"query": "from-exec"/)).toBeInTheDocument();
    expect(screen.getByText(/"max_results": 3/)).toBeInTheDocument();
  });

  it('should show web search content preview in outputs tab', () => {
    const state = createStoreState({
      currentExecution: {
        node_executions: {
          tool_1: {
            node_id: 'tool_1',
            status: 'completed',
            inputs: { query: 'from-exec', max_results: 3 },
            outputs: {
              type: 'tool_result',
              output: {
                results: [
                  {
                    title: 'Result',
                    url: 'https://example.com',
                    content: 'content preview',
                  },
                ],
              },
            },
            error: null,
            started_at: null,
            completed_at: null,
            streaming_output: '',
            metadata: { tool_name: 'web_search' },
          },
        },
      },
    });
    mockedStore.mockImplementation((selector?: any) => (selector ? selector(state) : state));

    render(<RightSidebar isCollapsed={false} onToggleCollapse={vi.fn()} />);

    const outputsTab = screen.getByRole('button', { name: 'Outputs' });
    act(() => {
      fireEvent.click(outputsTab);
    });

    expect(screen.getByText('Content (preview)')).toBeInTheDocument();
    expect(screen.getByText('content preview')).toBeInTheDocument();
  });
});
