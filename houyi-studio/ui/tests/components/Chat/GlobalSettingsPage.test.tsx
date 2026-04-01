import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { GlobalSettingsPage } from '@/components/Chat/GlobalSettingsPage';

const { invalidateModelCacheMock } = vi.hoisted(() => ({
  invalidateModelCacheMock: vi.fn(),
}));

vi.mock('@/hooks/useAvailableModels', () => ({
  invalidateModelCache: invalidateModelCacheMock,
}));

const mockFetch = vi.fn();
global.fetch = mockFetch as any;

const memoryConfigResp = { ok: true, json: async () => ({ config: { enabled: true, auto_extract: true } }) };

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

function routedFetch(overrides?: Record<string, any>) {
  const settings = { ...defaultSettings, ...overrides };
  return (url: string, opts?: any) => {
    if (url === '/api/memory/config') return Promise.resolve(memoryConfigResp);
    if (typeof url === 'string' && url.startsWith('/api/memory/')) return Promise.resolve(memoryConfigResp);
    if (url === '/api/chat/settings')
      return Promise.resolve({ ok: true, json: async () => settings });
    return Promise.resolve({ ok: false, status: 404 });
  };
}

describe('GlobalSettingsPage', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetch.mockImplementation(routedFetch());
    global.fetch = mockFetch as any;
    invalidateModelCacheMock.mockReset();
  });

  it('returns null when not open', () => {
    const { container } = render(<GlobalSettingsPage isOpen={false} onClose={vi.fn()} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders title and loads settings on open', async () => {
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
    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getAllByText('SiliconFlow').length).toBeGreaterThan(0);
    });
  });

  it('shows default model value', async () => {
    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      const modelInput = screen.getByPlaceholderText('deepseek-ai/DeepSeek-V3');
      expect((modelInput as HTMLInputElement).value).toBe('deepseek-v3');
    });
  });

  it('calls onClose when Cancel clicked', async () => {
    const onClose = vi.fn();
    render(<GlobalSettingsPage isOpen={true} onClose={onClose} />);

    await waitFor(() => {
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls PUT on Save', async () => {
    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByPlaceholderText('deepseek-ai/DeepSeek-V3')).toBeInTheDocument();
    });

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
    mockFetch.mockImplementation((url: string) => {
      if (url === '/api/memory/config') return Promise.resolve(memoryConfigResp);
      return Promise.resolve({ ok: false, status: 500 });
    });

    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load settings/)).toBeInTheDocument();
    });
  });

  it('shows Add Provider button', async () => {
    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/Add Provider/)).toBeInTheDocument();
    });
  });

  it('shows display fields with correct values', async () => {
    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      const userNameInputs = screen.getAllByDisplayValue('Von');
      expect(userNameInputs.length).toBeGreaterThan(0);
      const assistantNameInputs = screen.getAllByDisplayValue('HouYi');
      expect(assistantNameInputs.length).toBeGreaterThan(0);
    });
  });

  it('shows max tokens explanation tooltip', async () => {
    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByTestId('global-max-tokens-info')).toBeInTheDocument();
    });

    fireEvent.mouseEnter(screen.getByTestId('global-max-tokens-info'));

    expect(screen.getByRole('tooltip')).toHaveTextContent(
      'Maximum response tokens to generate. Larger values allow longer answers, but leave less room for input context in each request. This does not change the model context window.',
    );
  });

  it('renders Memory section with toggles', async () => {
    render(<GlobalSettingsPage isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Memory')).toBeInTheDocument();
    });
  });
});
