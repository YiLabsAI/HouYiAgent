/**
 * Tests for AgentHub — hub title, agent cards, sessions, and navigation.
 */
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AgentHub } from '@/components/Agent/AgentHub';

const mockStore = {
  sessions: [] as any[],
  fetchSessions: vi.fn(),
  openSession: vi.fn(),
  deleteSession: vi.fn(),
  reset: vi.fn(),
  disconnectSSE: vi.fn(),
};

vi.mock('@/stores/useResearchStore', () => {
  const storeFn = vi.fn((selector?: (s: any) => unknown) =>
    selector ? selector(mockStore as any) : mockStore,
  );
  (storeFn as any).getState = () => mockStore;
  return { useResearchStore: storeFn };
});

vi.mock('@/components/Agent/DeepResearch/Workspace', () => ({
  DeepResearchWorkspace: () => <div data-testid="deep-research-workspace" />,
}));
vi.mock('@/components/Memory/MemoryInbox', () => ({
  MemoryInbox: () => <div data-testid="memory-inbox" />,
}));

describe('AgentHub', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.location.hash = '';
    mockStore.sessions = [];
    mockStore.fetchSessions = vi.fn();
    mockStore.openSession = vi.fn().mockResolvedValue(undefined);
    mockStore.deleteSession = vi.fn();
    mockStore.reset = vi.fn();
    mockStore.disconnectSSE = vi.fn();
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ types: [] }),
    }) as unknown as typeof fetch;
  });

  /** Flush fetch + setState from useEffect to avoid act() warnings. */
  async function flushHubEffects() {
    await waitFor(() => expect(mockStore.fetchSessions).toHaveBeenCalled());
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledWith('/api/agents/types'));
  }

  it('renders hub title and subtitle', async () => {
    render(<AgentHub />);
    expect(screen.getByRole('heading', { name: 'Agent Hub' })).toBeInTheDocument();
    expect(screen.getByText('Choose an agent to start a new session')).toBeInTheDocument();
    await flushHubEffects();
  });

  it('renders fallback cards when API fails', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('network')) as unknown as typeof fetch;
    render(<AgentHub />);
    expect(await screen.findByText('Deep Research')).toBeInTheDocument();
    expect(screen.getByText('Code Analyst')).toBeInTheDocument();
    expect(screen.getByText('Personal Office')).toBeInTheDocument();
    expect(screen.getByText('Data Analysis')).toBeInTheDocument();
  });

  it('renders API agent types', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        types: [
          {
            id: 'custom_a',
            name: 'Alpha Explorer',
            description: 'Test agent A',
            icon: '🧪',
            available: true,
          },
          {
            id: 'custom_b',
            name: 'Beta Helper',
            description: 'Test agent B',
            icon: '📎',
            available: false,
          },
        ],
      }),
    }) as unknown as typeof fetch;
    render(<AgentHub />);
    expect(await screen.findByText('Alpha Explorer')).toBeInTheDocument();
    expect(screen.getByText('Beta Helper')).toBeInTheDocument();
  });

  it('shows empty sessions message', async () => {
    mockStore.sessions = [];
    render(<AgentHub />);
    expect(screen.getByText('No recent sessions')).toBeInTheDocument();
    await flushHubEffects();
  });

  it('renders session list', async () => {
    mockStore.sessions = [
      {
        run_id: 'rr-1',
        query: 'quantum widgets',
        status: 'executing',
        created_at: '2026-01-01',
      },
    ];
    render(<AgentHub />);
    expect(screen.getByText('quantum widgets')).toBeInTheDocument();
    expect(screen.getByText(/Deep Research • Running/)).toBeInTheDocument();
    await flushHubEffects();
  });

  it('clicking Deep Research opens workspace', async () => {
    render(<AgentHub />);
    fireEvent.click(await screen.findByRole('button', { name: /Deep Research/i }));
    expect(screen.getByTestId('deep-research-workspace')).toBeInTheDocument();
    expect(mockStore.reset).toHaveBeenCalled();
  });

  it('clicking Memory Inbox opens inbox', async () => {
    render(<AgentHub />);
    await flushHubEffects();
    fireEvent.click(screen.getByRole('button', { name: /Memory Inbox/i }));
    expect(screen.getByTestId('memory-inbox')).toBeInTheDocument();
  });

  it('back button returns to hub and resets store', async () => {
    render(<AgentHub />);
    fireEvent.click(await screen.findByRole('button', { name: /Deep Research/i }));
    expect(screen.getByTestId('deep-research-workspace')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Agent Hub/i }));
    expect(screen.queryByTestId('deep-research-workspace')).not.toBeInTheDocument();
    expect(screen.getByText('Choose an agent to start a new session')).toBeInTheDocument();
    expect(mockStore.reset).toHaveBeenCalled();
    expect(mockStore.disconnectSSE).toHaveBeenCalled();
  });

  it('opening a session resets store before loading', async () => {
    mockStore.sessions = [
      { run_id: 'rr-nav', query: 'nav test', status: 'completed', created_at: '2026-01-01' },
    ];
    render(<AgentHub />);
    await flushHubEffects();
    fireEvent.click(screen.getByText('nav test'));
    expect(mockStore.reset).toHaveBeenCalled();
    expect(mockStore.openSession).toHaveBeenCalledWith('rr-nav');
  });

  it('popstate event syncs view from hash', async () => {
    render(<AgentHub />);
    await flushHubEffects();
    window.location.hash = '#/research';
    window.dispatchEvent(new PopStateEvent('popstate'));
    // Should switch to deep_research view
    await waitFor(() => {
      expect(screen.getByTestId('deep-research-workspace')).toBeInTheDocument();
    });
  });

  it('pagination shows when > 10 sessions', async () => {
    mockStore.sessions = Array.from({ length: 12 }, (_, i) => ({
      run_id: `rr-${i}`,
      query: `query ${i}`,
      status: 'completed',
      created_at: '2026-01-01',
    }));
    render(<AgentHub />);
    expect(await screen.findByRole('button', { name: 'Prev' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next' })).toBeInTheDocument();
    expect(screen.getByText('1/2')).toBeInTheDocument();
    await flushHubEffects();
  });

  it('delete confirmation flow', async () => {
    mockStore.sessions = [
      {
        run_id: 'rr-del-1',
        query: 'done topic',
        status: 'completed',
        created_at: '2026-01-01',
      },
    ];
    render(<AgentHub />);
    await flushHubEffects();

    const row = screen.getByText('done topic').closest('.group');
    expect(row).toBeTruthy();
    fireEvent.click(within(row as HTMLElement).getByTitle('Delete session'));
    expect(screen.getByRole('button', { name: 'Confirm' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(mockStore.deleteSession).toHaveBeenCalledWith('rr-del-1');
  });
});
