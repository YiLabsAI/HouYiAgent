/**
 * Tests for Header component with primaryMode / onSetPrimaryMode props.
 *
 * Covers:
 *   - Mode indicator rendering (graph active, chat active)
 *   - Mode switch button callbacks
 *   - Conditional rendering of chat-specific and graph-specific toolbar items
 *   - Agent button disabled state
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Header } from '@/components/Header';
import { useConsoleStore } from '@/stores/useConsoleStore';

// Mock stores to avoid WebSocket and real state
vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: vi.fn(),
}));
vi.mock('@/stores/useThemeStore', () => ({
  useThemeStore: vi.fn(() => ({ theme: 'dark', setTheme: vi.fn() })),
  THEMES: [{ id: 'dark', label: 'Dark' }],
}));

const mockConsoleStore = (overrides: Record<string, unknown> = {}) => {
  const state = {
    connectionStatus: 'connected',
    getToolStatistics: () => ({
      totalCalls: 0,
      successfulCalls: 0,
      failedCalls: 0,
      toolsByName: {},
    }),
    ...overrides,
  };
  // useConsoleStore is called both as a hook (returns full state) and
  // with a selector: useConsoleStore(selector).  Handle both forms.
  const fn = (useConsoleStore as unknown as ReturnType<typeof vi.fn>);
  fn.mockImplementation((selector?: (s: typeof state) => unknown) =>
    selector ? selector(state) : state,
  );
};

describe('Header', () => {
  beforeEach(() => {
    mockConsoleStore();
  });

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  it('renders Graph button as active when primaryMode is graph', () => {
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={vi.fn()}
      />,
    );
    const graphBtn = screen.getByText('Graph').closest('button');
    expect(graphBtn?.className).toContain('bg-blue-600');
    const chatBtn = screen.getByText('Chat').closest('button');
    expect(chatBtn?.className).not.toContain('bg-blue-600');
  });

  it('renders Chat button as active when primaryMode is chat', () => {
    render(
      <Header
        primaryMode="chat"
        onSetPrimaryMode={vi.fn()}
      />,
    );
    const chatBtn = screen.getByText('Chat').closest('button');
    expect(chatBtn?.className).toContain('bg-blue-600');
    const graphBtn = screen.getByText('Graph').closest('button');
    expect(graphBtn?.className).not.toContain('bg-blue-600');
  });

  it('renders Agent button as disabled', () => {
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={vi.fn()}
      />,
    );
    const agentBtn = screen.getByText('Agent').closest('button');
    expect(agentBtn).toBeDisabled();
  });

  // ---------------------------------------------------------------------------
  // Mode switch callbacks
  // ---------------------------------------------------------------------------

  it('calls onSetPrimaryMode("graph") when Graph button is clicked', () => {
    const onSetPrimaryMode = vi.fn();
    render(
      <Header
        primaryMode="chat"
        onSetPrimaryMode={onSetPrimaryMode}
      />,
    );
    fireEvent.click(screen.getByText('Graph'));
    expect(onSetPrimaryMode).toHaveBeenCalledWith('graph');
    expect(onSetPrimaryMode).toHaveBeenCalledTimes(1);
  });

  it('calls onSetPrimaryMode("chat") when Chat button is clicked', () => {
    const onSetPrimaryMode = vi.fn();
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={onSetPrimaryMode}
      />,
    );
    fireEvent.click(screen.getByText('Chat'));
    expect(onSetPrimaryMode).toHaveBeenCalledWith('chat');
    expect(onSetPrimaryMode).toHaveBeenCalledTimes(1);
  });

  // ---------------------------------------------------------------------------
  // Conditional toolbar items
  // ---------------------------------------------------------------------------

  it('shows search/bookmark buttons in chat mode (settings unified to ActivityBar)', () => {
    render(
      <Header
        primaryMode="chat"
        onSetPrimaryMode={vi.fn()}
        onOpenSearch={vi.fn()}
        onOpenBookmarks={vi.fn()}
      />,
    );
    expect(screen.getByTitle('Search conversations (Cmd+K)')).toBeInTheDocument();
    expect(screen.getByTitle('Bookmarks')).toBeInTheDocument();
    // Settings button removed from Header — unified to ActivityBar gear
    expect(screen.queryByTitle('Global settings')).not.toBeInTheDocument();
  });

  it('hides search/bookmark buttons in graph mode', () => {
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={vi.fn()}
        onOpenSearch={vi.fn()}
        onOpenBookmarks={vi.fn()}
      />,
    );
    expect(screen.queryByTitle('Search conversations (Cmd+K)')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Bookmarks')).not.toBeInTheDocument();
  });

  it('shows timeline/checkpoints buttons in graph mode', () => {
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={vi.fn()}
        onOpenBottomPanel={vi.fn()}
      />,
    );
    expect(screen.getByTitle('Timeline')).toBeInTheDocument();
    expect(screen.getByTitle('Checkpoints')).toBeInTheDocument();
  });

  it('hides timeline/checkpoints buttons in chat mode', () => {
    render(
      <Header
        primaryMode="chat"
        onSetPrimaryMode={vi.fn()}
        onOpenBottomPanel={vi.fn()}
      />,
    );
    expect(screen.queryByTitle('Timeline')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Checkpoints')).not.toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Callback wiring
  // ---------------------------------------------------------------------------

  it('calls onOpenBottomPanel with correct tab for timeline and checkpoints', () => {
    const onOpenBottomPanel = vi.fn();
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={vi.fn()}
        onOpenBottomPanel={onOpenBottomPanel}
      />,
    );
    fireEvent.click(screen.getByTitle('Timeline'));
    expect(onOpenBottomPanel).toHaveBeenCalledWith('observability');

    fireEvent.click(screen.getByTitle('Checkpoints'));
    expect(onOpenBottomPanel).toHaveBeenCalledWith('checkpoints');
  });

  it('calls onOpenSearch when search button clicked in chat mode', () => {
    const onOpenSearch = vi.fn();
    render(
      <Header
        primaryMode="chat"
        onSetPrimaryMode={vi.fn()}
        onOpenSearch={onOpenSearch}
      />,
    );
    fireEvent.click(screen.getByTitle('Search conversations (Cmd+K)'));
    expect(onOpenSearch).toHaveBeenCalledTimes(1);
  });

  // ---------------------------------------------------------------------------
  // Connection status display
  // ---------------------------------------------------------------------------

  it('shows Live indicator when connected', () => {
    mockConsoleStore({ connectionStatus: 'connected' });
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={vi.fn()}
      />,
    );
    expect(screen.getByText('Live')).toBeInTheDocument();
  });

  it('shows Offline indicator when disconnected', () => {
    mockConsoleStore({ connectionStatus: 'disconnected' });
    render(
      <Header
        primaryMode="graph"
        onSetPrimaryMode={vi.fn()}
      />,
    );
    expect(screen.getByText('Offline')).toBeInTheDocument();
  });
});
