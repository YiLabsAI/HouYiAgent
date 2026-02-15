import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { ConversationSettingsDrawer } from '@/components/Chat/ConversationSettingsDrawer';
import { invalidateModelCache } from '@/hooks/useAvailableModels';

// Mock useChatStore
const mockUpdateConversation = vi.fn().mockResolvedValue(undefined);
let mockActiveConversation: any = null;

vi.mock('@/stores/useChatStore', () => ({
  useChatStore: (selector: any) => {
    const state = {
      activeConversation: mockActiveConversation,
      updateConversation: mockUpdateConversation,
    };
    return selector(state);
  },
}));

vi.mock('@/constants/models', () => ({
  DEFAULT_MODEL: 'deepseek-v3',
  MODEL_OPTIONS: [
    { value: 'deepseek-v3', label: 'DeepSeek-V3' },
    { value: 'gpt-4', label: 'GPT-4' },
  ],
}));

const settingsResponse = {
  providers: [
    { id: 'p1', name: 'Test', enabled: true, models: ['deepseek-v3', 'gpt-4'] },
  ],
  defaults: { temperature: 0.7, max_tokens: 4096 },
};

const modelsResponse = {
  models: [
    { model: 'deepseek-v3', provider: 'Test' },
    { model: 'gpt-4', provider: 'Test' },
  ],
};

const mockFetch = vi.fn().mockImplementation((url: string) => {
  if (url.includes('/models')) {
    return Promise.resolve({ json: () => Promise.resolve(modelsResponse) });
  }
  return Promise.resolve({ json: () => Promise.resolve(settingsResponse) });
});
globalThis.fetch = mockFetch as any;

describe('ConversationSettingsDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    invalidateModelCache();
    // Restore URL-aware fetch mock after clearAllMocks
    globalThis.fetch = mockFetch.mockImplementation((url: string) => {
      if (url.includes('/models')) {
        return Promise.resolve({ json: () => Promise.resolve(modelsResponse) });
      }
      return Promise.resolve({ json: () => Promise.resolve(settingsResponse) });
    }) as any;
    mockActiveConversation = {
      conversation_id: 'c1',
      title: 'Test Chat',
      model: '',
      system_instructions: '',
    };
  });

  it('returns null when not open', () => {
    const { container } = render(
      <ConversationSettingsDrawer isOpen={false} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    expect(container.innerHTML).toBe('');
  });

  it('renders drawer with title when open', () => {
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    expect(screen.getByText('Conversation Settings')).toBeInTheDocument();
  });

  it('shows model selector and system prompt fields', () => {
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('System Prompt')).toBeInTheDocument();
  });

  it('shows inherited-from-global hint when fields are empty', () => {
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    const hints = screen.getAllByText('(inherited from global)');
    expect(hints.length).toBeGreaterThanOrEqual(3); // model + system prompt + temperature (+ maxTokens when visible)
  });

  it('calls onClose when backdrop clicked', () => {
    const onClose = vi.fn();
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={onClose} onOpenGlobalSettings={vi.fn()} />
    );
    // CenterStage backdrop uses data-testid
    const backdrop = screen.getByTestId('center-stage-backdrop');
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Save button is disabled when form is not dirty', () => {
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    const saveBtn = screen.getByText('Save');
    expect(saveBtn).toBeDisabled();
  });

  it('Save button becomes enabled after changing model', () => {
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'gpt-4' } });
    const saveBtn = screen.getByText('Save');
    expect(saveBtn).not.toBeDisabled();
  });

  it('calls updateConversation on Save', async () => {
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );

    // Wait for async fetch to populate model options
    await waitFor(() => {
      const select = screen.getByRole('combobox');
      expect(select.querySelectorAll('option').length).toBeGreaterThan(1);
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'gpt-4' } });

    const saveBtn = screen.getByText('Save');
    fireEvent.click(saveBtn);

    expect(mockUpdateConversation).toHaveBeenCalledWith('c1', expect.objectContaining({
      model: 'gpt-4',
    }));
  });

  it('calls updateConversation with empty strings on Reset', () => {
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    const resetBtn = screen.getByText(/Reset to Global/);
    fireEvent.click(resetBtn);

    expect(mockUpdateConversation).toHaveBeenCalledWith('c1', expect.objectContaining({
      model: '',
      system_instructions: '',
    }));
  });

  it('shows "No active conversation" when no conversation', () => {
    mockActiveConversation = null;
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={vi.fn()} onOpenGlobalSettings={vi.fn()} />
    );
    expect(screen.getByText('No active conversation')).toBeInTheDocument();
  });

  it('Global Settings link calls onClose then onOpenGlobalSettings', () => {
    const onClose = vi.fn();
    const onOpenGlobal = vi.fn();
    render(
      <ConversationSettingsDrawer isOpen={true} onClose={onClose} onOpenGlobalSettings={onOpenGlobal} />
    );
    fireEvent.click(screen.getByText('Global Settings...'));
    expect(onClose).toHaveBeenCalled();
    expect(onOpenGlobal).toHaveBeenCalled();
  });
});
