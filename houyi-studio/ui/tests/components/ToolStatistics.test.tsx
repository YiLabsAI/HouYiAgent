import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ToolStatistics } from '@/components/ToolStatistics';

// ─── Mocks ───────────────────────────────────────────────────────
const emptyStats = {
  totalCalls: 0,
  successfulCalls: 0,
  failedCalls: 0,
  toolsByName: {} as Record<string, { count: number; successful: number; failed: number }>,
  toolNodes: 0,
  totalNodes: 0,
};

let toolStatsMock = { ...emptyStats };

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: (selector: any) => {
    const state = { getToolStatistics: () => toolStatsMock };
    return selector(state);
  },
}));

const mockFetch = vi.fn();
globalThis.fetch = mockFetch as any;

describe('ToolStatistics', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    globalThis.fetch = mockFetch as any;
    toolStatsMock = { ...emptyStats };
  });

  // ─── Rendering ─────────────────────────────────────────────────

  it('returns null when no registered tools and no exec stats', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ tools: [] }),
    });
    const { container } = render(<ToolStatistics />);
    // Wait for the fetch effect to settle
    await waitFor(() => expect(mockFetch).toHaveBeenCalled());
    expect(container.querySelector('[data-testid="tool-statistics"]')).toBeNull();
  });

  it('renders registered tools count from /api/tools', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tools: [
          { name: 'calculator', description: 'Math ops' },
          { name: 'web_search', description: 'Search the web' },
        ],
      }),
    });

    render(<ToolStatistics />);

    await waitFor(() => {
      expect(screen.getByTestId('tool-registered-count')).toHaveTextContent('2 skills');
    });
  });

  it('renders execution stats when available', async () => {
    toolStatsMock = {
      ...emptyStats,
      totalCalls: 5,
      successfulCalls: 4,
      failedCalls: 1,
      toolsByName: {
        calculator: { count: 3, successful: 2, failed: 1 },
        web_search: { count: 2, successful: 2, failed: 0 },
      },
      toolNodes: 2,
      totalNodes: 4,
    };

    // No registered tools for this test
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ tools: [] }),
    });

    render(<ToolStatistics />);

    await waitFor(() => {
      expect(screen.getByTestId('tool-exec-calls')).toHaveTextContent('4');
    });
  });

  // ─── Dropdown ──────────────────────────────────────────────────

  it('opens dropdown on pill click', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tools: [{ name: 'calculator', description: 'Math operations' }],
      }),
    });

    render(<ToolStatistics />);

    await waitFor(() => {
      expect(screen.getByTestId('tool-registered-count')).toBeInTheDocument();
    });

    // Dropdown should not be visible yet
    expect(screen.queryByTestId('tool-statistics-dropdown')).not.toBeInTheDocument();

    // Click the pill
    fireEvent.click(screen.getByTestId('tool-statistics').querySelector('button')!);

    // Dropdown should now be visible
    expect(screen.getByTestId('tool-statistics-dropdown')).toBeInTheDocument();
    expect(screen.getByText('Registered Skills (1)')).toBeInTheDocument();
    expect(screen.getByText('calculator')).toBeInTheDocument();
    expect(screen.getByText('Math operations')).toBeInTheDocument();
  });

  it('shows execution statistics in dropdown when in graph mode', async () => {
    toolStatsMock = {
      ...emptyStats,
      totalCalls: 3,
      successfulCalls: 3,
      failedCalls: 0,
      toolsByName: {
        calculator: { count: 3, successful: 3, failed: 0 },
      },
      toolNodes: 1,
      totalNodes: 3,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ tools: [] }),
    });

    render(<ToolStatistics />);

    await waitFor(() => {
      expect(screen.getByTestId('tool-exec-calls')).toBeInTheDocument();
    });

    // Open dropdown
    fireEvent.click(screen.getByTestId('tool-statistics').querySelector('button')!);

    expect(screen.getByText('Execution Statistics')).toBeInTheDocument();
    expect(screen.getByText('3 ok')).toBeInTheDocument();
    // Verify totalCalls is rendered (inside "Calls: 3" text)
    const callsText = screen.getByText((_, el) =>
      el?.tagName === 'SPAN' && el?.textContent === 'Calls: 3'
    );
    expect(callsText).toBeInTheDocument();
  });

  it('closes dropdown on outside click', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tools: [{ name: 'calc', description: '' }],
      }),
    });

    render(<ToolStatistics />);

    await waitFor(() => {
      expect(screen.getByTestId('tool-registered-count')).toBeInTheDocument();
    });

    // Open dropdown
    fireEvent.click(screen.getByTestId('tool-statistics').querySelector('button')!);
    expect(screen.getByTestId('tool-statistics-dropdown')).toBeInTheDocument();

    // Click outside
    fireEvent.mouseDown(document.body);

    expect(screen.queryByTestId('tool-statistics-dropdown')).not.toBeInTheDocument();
  });

  // ─── Error resilience ─────────────────────────────────────────

  it('handles fetch failure gracefully', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    toolStatsMock = {
      ...emptyStats,
      totalCalls: 1,
      successfulCalls: 1,
      failedCalls: 0,
      toolsByName: { calc: { count: 1, successful: 1, failed: 0 } },
      toolNodes: 1,
      totalNodes: 1,
    };

    render(<ToolStatistics />);

    // Should still render exec stats even if fetch fails
    await waitFor(() => {
      expect(screen.getByTestId('tool-exec-calls')).toBeInTheDocument();
    });
  });

  it('shows both registered tools and exec stats together', async () => {
    toolStatsMock = {
      ...emptyStats,
      totalCalls: 2,
      successfulCalls: 1,
      failedCalls: 1,
      toolsByName: {
        calc: { count: 2, successful: 1, failed: 1 },
      },
      toolNodes: 1,
      totalNodes: 2,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tools: [
          { name: 'calc', description: 'Calculator' },
          { name: 'web', description: 'Web search' },
        ],
      }),
    });

    render(<ToolStatistics />);

    await waitFor(() => {
      expect(screen.getByTestId('tool-registered-count')).toHaveTextContent('2 skills');
      expect(screen.getByTestId('tool-exec-calls')).toHaveTextContent('1');
    });

    // Open dropdown — should have both sections
    fireEvent.click(screen.getByTestId('tool-statistics').querySelector('button')!);
    expect(screen.getByText('Execution Statistics')).toBeInTheDocument();
    expect(screen.getByText('Registered Skills (2)')).toBeInTheDocument();
  });
});
