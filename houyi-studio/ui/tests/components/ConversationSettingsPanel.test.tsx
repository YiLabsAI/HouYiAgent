/**
 * Tests for ConversationSettingsPanel Tier 1 component (P3-08 / §7.5.2).
 *
 * Covers:
 *   - Model selector displays current model
 *   - Temperature slider displays current value
 *   - Deep Research toggle switches state
 *   - Quick stats (tool calls, max tools)
 *   - Full Settings action button callback
 *   - Web search provider conditional display
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ConversationSettingsPanel } from '@/components/panels/ConversationSettingsPanel';
import { useConsoleStore } from '@/stores/useConsoleStore';

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: vi.fn(),
}));

vi.mock('@/hooks/useAvailableModels', () => ({
  useAvailableModels: () => ({
    models: [
      { model: 'deepseek-ai/DeepSeek-V3', provider: 'siliconflow' },
      { model: 'gpt-4o', provider: 'openai' },
    ],
    modelIds: ['deepseek-ai/DeepSeek-V3', 'gpt-4o'],
    isLoading: false,
  }),
}));

const DEFAULT_RUN_SETTINGS = {
  model: 'deepseek-ai/DeepSeek-V3',
  temperature: 0.7,
  enable_tool_calls: true,
  tool_call_strategy: 'balanced',
  max_tool_calls: 10,
  web_search_provider: null as string | null,
};

const mockStore = (runSettingsOverrides: Record<string, unknown> = {}) => {
  const runSettings = { ...DEFAULT_RUN_SETTINGS, ...runSettingsOverrides };
  const updateRunSettings = vi.fn();
  const state: Record<string, unknown> = {
    runSettings,
    updateRunSettings,
  };
  const fn = useConsoleStore as unknown as ReturnType<typeof vi.fn>;
  fn.mockImplementation((selector?: (s: typeof state) => unknown) =>
    selector ? selector(state) : state,
  );
  return { updateRunSettings };
};

describe('ConversationSettingsPanel', () => {
  beforeEach(() => {
    mockStore();
  });

  // --- Model selector ---

  it('renders model selector with current model', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Model')).toBeInTheDocument();
    const select = screen.getByRole('combobox');
    expect(select).toHaveValue('deepseek-ai/DeepSeek-V3');
  });

  it('lists available models in dropdown', () => {
    render(<ConversationSettingsPanel />);
    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(2);
    expect(options[0]).toHaveTextContent('deepseek-ai/DeepSeek-V3');
    expect(options[1]).toHaveTextContent('gpt-4o');
  });

  // --- Temperature ---

  it('displays temperature label and value', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Temperature')).toBeInTheDocument();
    expect(screen.getByText('0.70')).toBeInTheDocument();
  });

  it('shows range labels (Precise / Creative)', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Precise')).toBeInTheDocument();
    expect(screen.getByText('Creative')).toBeInTheDocument();
  });

  // --- Deep Research toggle ---

  it('renders Deep Research toggle in off state by default', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Deep Research')).toBeInTheDocument();
    const toggle = screen.getByRole('switch');
    expect(toggle).toHaveAttribute('aria-checked', 'false');
  });

  it('toggles Deep Research on/off', () => {
    render(<ConversationSettingsPanel />);
    const toggle = screen.getByRole('switch');

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByText(/Deep Research mode active/)).toBeInTheDocument();

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    expect(screen.getByText(/Enable to perform thorough research/)).toBeInTheDocument();
  });

  // --- Quick stats ---

  it('shows tool calls enabled status', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Tool Calls')).toBeInTheDocument();
    expect(screen.getByText('Enabled')).toBeInTheDocument();
  });

  it('shows tool calls disabled when setting is false', () => {
    mockStore({ enable_tool_calls: false });
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Disabled')).toBeInTheDocument();
  });

  it('shows max tool calls count', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Max Tools')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument();
  });

  // --- Web search provider ---

  it('does NOT show web search when provider is null', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.queryByText('Web Search')).not.toBeInTheDocument();
  });

  it('shows web search provider when configured', () => {
    mockStore({ web_search_provider: 'tavily' });
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Web Search')).toBeInTheDocument();
    expect(screen.getByText('tavily')).toBeInTheDocument();
  });

  // --- Full Settings button ---

  it('calls onOpenFullSettings when Full Settings button clicked', () => {
    const onOpenFullSettings = vi.fn();
    render(<ConversationSettingsPanel onOpenFullSettings={onOpenFullSettings} />);
    fireEvent.click(screen.getByText('Full Settings...'));
    expect(onOpenFullSettings).toHaveBeenCalledTimes(1);
  });

  it('renders Full Settings button even without callback', () => {
    render(<ConversationSettingsPanel />);
    expect(screen.getByText('Full Settings...')).toBeInTheDocument();
  });
});
