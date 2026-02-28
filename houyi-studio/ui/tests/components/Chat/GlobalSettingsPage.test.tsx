import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { GlobalSettingsPage } from '@/components/Chat/GlobalSettingsPage';

const { invalidateModelCacheMock } = vi.hoisted(() => ({
  invalidateModelCacheMock: vi.fn(),
}));

vi.mock('@/hooks/useAvailableModels', () => ({
  invalidateModelCache: invalidateModelCacheMock,
}));

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

const defaultSettings = {
  version: 1,
  providers: [
    {
      id: 'p1',
      name: 'SiliconFlow',
      api_key: 'sk-test',
      base_url: 'https://api.siliconflow.cn/v1',
      models: ['deepseek-v3'],
      enabled: true,
    },
  ],
  defaults: {
    model: 'deepseek-v3',
    system_instructions: 'You are helpful.',
    temperature: 0.7,
    max_tokens: 4096,
  },
  display: {
    user_name: 'Von',
    user_avatar: null,
    assistant_name: 'HouYi',
    assistant_avatar: null,
  },
  updated_at: 1000,
};

describe('GlobalSettingsPage', () => {
  beforeEach(() => {
    // resetAllMocks clears both call history AND the implementation queue
    // (clearAllMocks only clears call history, leaving unconsumed mockResolvedValueOnce entries)
    vi.resetAllMocks();
    global.fetch = mockFetch;
    invalidateModelCacheMock.mockReset();
  });

  it('returns null when not open', () => {
    const { container } = render(<GlobalSettingsPage isOpen={false} onClose={vi.fn()} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders title and loads settings on open', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => defaultSettings,
    });

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByText('Global Settings')).toBeInTheDocument();

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/chat/settings');
    });

    await waitFor(() => {
      expect(screen.getByText('LLM Providers')).toBeInTheDocument();
      expect(screen.getByText('Defaults')).toBeInTheDocument();
      expect(screen.getByText('Display')).toBeInTheDocument();
    });
  });

  it('shows provider name after loading', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => defaultSettings,
    });

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      // Provider name appears in the provider header row (may also appear in preset buttons)
      expect(screen.getAllByText('SiliconFlow').length).toBeGreaterThan(0);
    });
  });

  it('shows default model value', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => defaultSettings,
    });

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      const modelInput = screen.getByPlaceholderText('deepseek-ai/DeepSeek-V3');
      expect((modelInput as HTMLInputElement).value).toBe('deepseek-v3');
    });
  });

  it('calls onClose when Cancel clicked', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => defaultSettings,
    });

    const onClose = vi.fn();
    render(<GlobalSettingsPage isOpen={true} onClose={onClose} />);

    await waitFor(() => {
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls PUT on Save', async () => {
    mockFetch
      .mockResolvedValueOnce({ ok: true, json: async () => defaultSettings })  // GET
      .mockResolvedValueOnce({ ok: true, json: async () => defaultSettings }); // PUT

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    // Wait for settings to load (model input appears)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('deepseek-ai/DeepSeek-V3')).toBeInTheDocument();
    });

    // Make the form dirty by changing the default model
    const modelInput = screen.getByPlaceholderText('deepseek-ai/DeepSeek-V3');
    fireEvent.change(modelInput, { target: { value: 'gpt-4' } });

    fireEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/chat/settings', expect.objectContaining({
        method: 'PUT',
      }));
    });

    await waitFor(() => {
      expect(invalidateModelCacheMock).toHaveBeenCalledTimes(1);
    });
  });

  it('shows error when fetch fails', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load settings/)).toBeInTheDocument();
    });
  });

  it('shows Add Provider button', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => defaultSettings,
    });

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/Add Provider/)).toBeInTheDocument();
    });
  });

  it('shows display fields with correct values', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => defaultSettings,
    });

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      const userNameInputs = screen.getAllByDisplayValue('Von');
      expect(userNameInputs.length).toBeGreaterThan(0);
      const assistantNameInputs = screen.getAllByDisplayValue('HouYi');
      expect(assistantNameInputs.length).toBeGreaterThan(0);
    });
  });
});
